
        # Get all neighbors data in a single list using distance based algorithm
        # This is used as a reference to map the icoxx data
        rdf_nb_lst = self.structure.get_neighbor_list(r=self.max_length)

        # Collect bond lengths, atom labels and bond strengths
        if site_index is not None:
            if site_index not in range(self.structure.num_sites):
                raise ValueError(f"{site_index} is not a valid site index for the structure")
            bond_labels = np.array(self.icoxxlist.icohpcollection._list_labels)
            indices = np.where(
                np.isin(
                    bond_labels,
                    list(
                        self.icoxxlist.icohpcollection.get_icohp_dict_of_site(
                            site_index, minbondlength=self.min_length, maxbondlength=self.max_length
                        ).keys()
                    ),
                )
            )[0]

            bond_lengths = np.array(self.icoxxlist.icohpcollection._list_length)[indices].tolist()
            atoms1 = np.array(self.icoxxlist.icohpcollection._list_atom1)[indices].tolist()
            atoms2 = np.array(self.icoxxlist.icohpcollection._list_atom2)[indices].tolist()
            icoxxs = np.array([sum(item.values()) for item in self.icoxxlist.icohpcollection._list_icohp])[
                indices
            ].tolist()
            trans = [list(item) for item in np.array(self.icoxxlist.icohpcollection._list_translation)[indices]]
            pairs = [[at1, at2] for at1, at2 in zip(atoms1, atoms2)]

            if self.n_electrons_scaling and self.grosspop:
                atom_1_index = np.array([int("".join(filter(str.isdigit, at1))) - 1 for at1 in atoms1])
                atom_2_index = np.array([int("".join(filter(str.isdigit, at2))) - 1 for at2 in atoms2])
                pair_n_electrons = np.array(
                    [
                        self.grosspop.list_dict_grosspop[at1]["Loewdin GP"]["total"]
                        + self.grosspop.list_dict_grosspop[at2]["Loewdin GP"]["total"]
                        for at1, at2 in zip(atom_1_index, atom_2_index)
                    ]
                )
                icoxxs = (np.array(icoxxs) / pair_n_electrons).tolist()

            relevant_site_indices = [
                ix for ix, (org, dest) in enumerate(zip(rdf_nb_lst[0], rdf_nb_lst[1])) if org == site_index
            ]
            rdf_distances = [round(dist, 5) for ix, dist in enumerate(rdf_nb_lst[3]) if ix in relevant_site_indices]
            rdf_trans = [[int(i) for i in img] for ix, img in enumerate(rdf_nb_lst[2]) if ix in relevant_site_indices]
            rdf_pairs = [
                [self.structure[org].species_string + str(org + 1), self.structure[dest].species_string + str(dest + 1)]
                for ix, (org, dest) in enumerate(zip(rdf_nb_lst[0], rdf_nb_lst[1]))
                if ix in relevant_site_indices
            ]
