// Function: sub_E02B90
// Address: 0xe02b90
//
const char *__fastcall sub_E02B90(unsigned int a1)
{
  const char *result; // rax

  switch ( a1 )
  {
    case 0u:
      return "DW_TAG_null";
    case 1u:
      return "DW_TAG_array_type";
    case 2u:
      return "DW_TAG_class_type";
    case 3u:
      return "DW_TAG_entry_point";
    case 4u:
      return "DW_TAG_enumeration_type";
    case 5u:
      return "DW_TAG_formal_parameter";
    case 6u:
    case 7u:
    case 9u:
    case 0xCu:
    case 0xEu:
    case 0x14u:
    case 0x3Eu:
      return 0;
    case 8u:
      return "DW_TAG_imported_declaration";
    case 0xAu:
      return "DW_TAG_label";
    case 0xBu:
      return "DW_TAG_lexical_block";
    case 0xDu:
      return "DW_TAG_member";
    case 0xFu:
      return "DW_TAG_pointer_type";
    case 0x10u:
      return "DW_TAG_reference_type";
    case 0x11u:
      return "DW_TAG_compile_unit";
    case 0x12u:
      return "DW_TAG_string_type";
    case 0x13u:
      return "DW_TAG_structure_type";
    case 0x15u:
      return "DW_TAG_subroutine_type";
    case 0x16u:
      return "DW_TAG_typedef";
    case 0x17u:
      return "DW_TAG_union_type";
    case 0x18u:
      return "DW_TAG_unspecified_parameters";
    case 0x19u:
      return "DW_TAG_variant";
    case 0x1Au:
      return "DW_TAG_common_block";
    case 0x1Bu:
      return "DW_TAG_common_inclusion";
    case 0x1Cu:
      return "DW_TAG_inheritance";
    case 0x1Du:
      return "DW_TAG_inlined_subroutine";
    case 0x1Eu:
      return "DW_TAG_module";
    case 0x1Fu:
      return "DW_TAG_ptr_to_member_type";
    case 0x20u:
      return "DW_TAG_set_type";
    case 0x21u:
      return "DW_TAG_subrange_type";
    case 0x22u:
      return "DW_TAG_with_stmt";
    case 0x23u:
      return "DW_TAG_access_declaration";
    case 0x24u:
      return "DW_TAG_base_type";
    case 0x25u:
      return "DW_TAG_catch_block";
    case 0x26u:
      return "DW_TAG_const_type";
    case 0x27u:
      return "DW_TAG_constant";
    case 0x28u:
      return "DW_TAG_enumerator";
    case 0x29u:
      return "DW_TAG_file_type";
    case 0x2Au:
      return "DW_TAG_friend";
    case 0x2Bu:
      return "DW_TAG_namelist";
    case 0x2Cu:
      return "DW_TAG_namelist_item";
    case 0x2Du:
      return "DW_TAG_packed_type";
    case 0x2Eu:
      return "DW_TAG_subprogram";
    case 0x2Fu:
      return "DW_TAG_template_type_parameter";
    case 0x30u:
      return "DW_TAG_template_value_parameter";
    case 0x31u:
      return "DW_TAG_thrown_type";
    case 0x32u:
      return "DW_TAG_try_block";
    case 0x33u:
      return "DW_TAG_variant_part";
    case 0x34u:
      return "DW_TAG_variable";
    case 0x35u:
      return "DW_TAG_volatile_type";
    case 0x36u:
      return "DW_TAG_dwarf_procedure";
    case 0x37u:
      return "DW_TAG_restrict_type";
    case 0x38u:
      return "DW_TAG_interface_type";
    case 0x39u:
      return "DW_TAG_namespace";
    case 0x3Au:
      return "DW_TAG_imported_module";
    case 0x3Bu:
      return "DW_TAG_unspecified_type";
    case 0x3Cu:
      return "DW_TAG_partial_unit";
    case 0x3Du:
      return "DW_TAG_imported_unit";
    case 0x3Fu:
      return "DW_TAG_condition";
    case 0x40u:
      return "DW_TAG_shared_type";
    case 0x41u:
      return "DW_TAG_type_unit";
    case 0x42u:
      return "DW_TAG_rvalue_reference_type";
    case 0x43u:
      return "DW_TAG_template_alias";
    case 0x44u:
      return "DW_TAG_coarray_type";
    case 0x45u:
      return "DW_TAG_generic_subrange";
    case 0x46u:
      return "DW_TAG_dynamic_type";
    case 0x47u:
      return "DW_TAG_atomic_type";
    case 0x48u:
      return "DW_TAG_call_site";
    case 0x49u:
      return "DW_TAG_call_site_parameter";
    case 0x4Au:
      return "DW_TAG_skeleton_unit";
    case 0x4Bu:
      return "DW_TAG_immutable_type";
    default:
      if ( a1 <= 0x420D )
      {
        if ( a1 > 0x41FF )
        {
          switch ( a1 )
          {
            case 0x4201u:
              result = "DW_TAG_SUN_function_template";
              break;
            case 0x4202u:
              result = "DW_TAG_SUN_class_template";
              break;
            case 0x4203u:
              result = "DW_TAG_SUN_struct_template";
              break;
            case 0x4204u:
              result = "DW_TAG_SUN_union_template";
              break;
            case 0x4205u:
              result = "DW_TAG_SUN_indirect_inheritance";
              break;
            case 0x4206u:
              result = "DW_TAG_SUN_codeflags";
              break;
            case 0x4207u:
              result = "DW_TAG_SUN_memop_info";
              break;
            case 0x4208u:
              result = "DW_TAG_SUN_omp_child_func";
              break;
            case 0x4209u:
              result = "DW_TAG_SUN_rtti_descriptor";
              break;
            case 0x420Au:
              result = "DW_TAG_SUN_dtor_info";
              break;
            case 0x420Bu:
              result = "DW_TAG_SUN_dtor";
              break;
            case 0x420Cu:
              result = "DW_TAG_SUN_f90_interface";
              break;
            case 0x420Du:
              result = "DW_TAG_SUN_fortran_vax_structure";
              break;
            default:
              result = "DW_TAG_APPLE_property";
              break;
          }
        }
        else if ( a1 == 16513 )
        {
          return "DW_TAG_MIPS_loop";
        }
        else
        {
          switch ( a1 )
          {
            case 0x4101u:
              result = "DW_TAG_format_label";
              break;
            case 0x4102u:
              result = "DW_TAG_function_template";
              break;
            case 0x4103u:
              result = "DW_TAG_class_template";
              break;
            case 0x4104u:
              result = "DW_TAG_GNU_BINCL";
              break;
            case 0x4105u:
              result = "DW_TAG_GNU_EINCL";
              break;
            case 0x4106u:
              result = "DW_TAG_GNU_template_template_param";
              break;
            case 0x4107u:
              result = "DW_TAG_GNU_template_parameter_pack";
              break;
            case 0x4108u:
              result = "DW_TAG_GNU_formal_parameter_pack";
              break;
            case 0x4109u:
              result = "DW_TAG_GNU_call_site";
              break;
            case 0x410Au:
              result = "DW_TAG_GNU_call_site_parameter";
              break;
            default:
              return 0;
          }
        }
        return result;
      }
      if ( a1 == 32775 )
        return "DW_TAG_GHS_template_templ_param";
      if ( a1 <= 0x8007 )
      {
        if ( a1 == 20753 )
          return "DW_TAG_ALTIUM_rom";
        if ( a1 <= 0x5111 )
        {
          if ( a1 == 20737 )
            return "DW_TAG_ALTIUM_circ_type";
          if ( a1 <= 0x5101 )
          {
            result = "DW_TAG_SUN_hi";
            if ( a1 == 17151 )
              return result;
            if ( a1 == 17152 )
              return "DW_TAG_LLVM_ptrauth_type";
          }
          else
          {
            result = "DW_TAG_ALTIUM_mwa_circ_type";
            if ( a1 == 20738 )
              return result;
            if ( a1 == 20739 )
              return "DW_TAG_ALTIUM_rev_carry_type";
          }
          return 0;
        }
        if ( a1 == 32773 )
          return "DW_TAG_GHS_using_namespace";
        if ( a1 > 0x8005 )
          return "DW_TAG_GHS_using_declaration";
        result = "DW_TAG_LLVM_annotation";
        if ( a1 != 24576 )
        {
          if ( a1 == 32772 )
            return "DW_TAG_GHS_namespace";
          return 0;
        }
      }
      else
      {
        if ( a1 > 0xB004 )
          return 0;
        if ( a1 <= 0xAFFF )
        {
          if ( a1 == 34663 )
            return "DW_TAG_UPC_relaxed";
          if ( a1 <= 0x8767 )
          {
            result = "DW_TAG_UPC_shared_type";
            if ( a1 == 34661 )
              return result;
            if ( a1 == 34662 )
              return "DW_TAG_UPC_strict_type";
          }
          else
          {
            result = "DW_TAG_PGI_kanji_type";
            if ( a1 == 40960 )
              return result;
            if ( a1 == 40992 )
              return "DW_TAG_PGI_interface_block";
          }
          return 0;
        }
        switch ( a1 )
        {
          case 0xB001u:
            result = "DW_TAG_BORLAND_Delphi_string";
            break;
          case 0xB002u:
            result = "DW_TAG_BORLAND_Delphi_dynamic_array";
            break;
          case 0xB003u:
            result = "DW_TAG_BORLAND_Delphi_set";
            break;
          case 0xB004u:
            result = "DW_TAG_BORLAND_Delphi_variant";
            break;
          default:
            result = "DW_TAG_BORLAND_property";
            break;
        }
      }
      return result;
  }
}
