// Function: sub_14E0540
// Address: 0x14e0540
//
const char *__fastcall sub_14E0540(unsigned int a1)
{
  const char *result; // rax

  switch ( a1 )
  {
    case 0u:
      result = "DW_TAG_null";
      break;
    case 1u:
      result = "DW_TAG_array_type";
      break;
    case 2u:
      result = "DW_TAG_class_type";
      break;
    case 3u:
      result = "DW_TAG_entry_point";
      break;
    case 4u:
      result = "DW_TAG_enumeration_type";
      break;
    case 5u:
      result = "DW_TAG_formal_parameter";
      break;
    case 6u:
    case 7u:
    case 9u:
    case 0xCu:
    case 0xEu:
    case 0x14u:
    case 0x3Eu:
      goto LABEL_7;
    case 8u:
      result = "DW_TAG_imported_declaration";
      break;
    case 0xAu:
      result = "DW_TAG_label";
      break;
    case 0xBu:
      result = "DW_TAG_lexical_block";
      break;
    case 0xDu:
      result = "DW_TAG_member";
      break;
    case 0xFu:
      result = "DW_TAG_pointer_type";
      break;
    case 0x10u:
      result = "DW_TAG_reference_type";
      break;
    case 0x11u:
      result = "DW_TAG_compile_unit";
      break;
    case 0x12u:
      result = "DW_TAG_string_type";
      break;
    case 0x13u:
      result = "DW_TAG_structure_type";
      break;
    case 0x15u:
      result = "DW_TAG_subroutine_type";
      break;
    case 0x16u:
      result = "DW_TAG_typedef";
      break;
    case 0x17u:
      result = "DW_TAG_union_type";
      break;
    case 0x18u:
      result = "DW_TAG_unspecified_parameters";
      break;
    case 0x19u:
      result = "DW_TAG_variant";
      break;
    case 0x1Au:
      result = "DW_TAG_common_block";
      break;
    case 0x1Bu:
      result = "DW_TAG_common_inclusion";
      break;
    case 0x1Cu:
      result = "DW_TAG_inheritance";
      break;
    case 0x1Du:
      result = "DW_TAG_inlined_subroutine";
      break;
    case 0x1Eu:
      result = "DW_TAG_module";
      break;
    case 0x1Fu:
      result = "DW_TAG_ptr_to_member_type";
      break;
    case 0x20u:
      result = "DW_TAG_set_type";
      break;
    case 0x21u:
      result = "DW_TAG_subrange_type";
      break;
    case 0x22u:
      result = "DW_TAG_with_stmt";
      break;
    case 0x23u:
      result = "DW_TAG_access_declaration";
      break;
    case 0x24u:
      result = "DW_TAG_base_type";
      break;
    case 0x25u:
      result = "DW_TAG_catch_block";
      break;
    case 0x26u:
      result = "DW_TAG_const_type";
      break;
    case 0x27u:
      result = "DW_TAG_constant";
      break;
    case 0x28u:
      result = "DW_TAG_enumerator";
      break;
    case 0x29u:
      result = "DW_TAG_file_type";
      break;
    case 0x2Au:
      result = "DW_TAG_friend";
      break;
    case 0x2Bu:
      result = "DW_TAG_namelist";
      break;
    case 0x2Cu:
      result = "DW_TAG_namelist_item";
      break;
    case 0x2Du:
      result = "DW_TAG_packed_type";
      break;
    case 0x2Eu:
      result = "DW_TAG_subprogram";
      break;
    case 0x2Fu:
      result = "DW_TAG_template_type_parameter";
      break;
    case 0x30u:
      result = "DW_TAG_template_value_parameter";
      break;
    case 0x31u:
      result = "DW_TAG_thrown_type";
      break;
    case 0x32u:
      result = "DW_TAG_try_block";
      break;
    case 0x33u:
      result = "DW_TAG_variant_part";
      break;
    case 0x34u:
      result = "DW_TAG_variable";
      break;
    case 0x35u:
      result = "DW_TAG_volatile_type";
      break;
    case 0x36u:
      result = "DW_TAG_dwarf_procedure";
      break;
    case 0x37u:
      result = "DW_TAG_restrict_type";
      break;
    case 0x38u:
      result = "DW_TAG_interface_type";
      break;
    case 0x39u:
      result = "DW_TAG_namespace";
      break;
    case 0x3Au:
      result = "DW_TAG_imported_module";
      break;
    case 0x3Bu:
      result = "DW_TAG_unspecified_type";
      break;
    case 0x3Cu:
      result = "DW_TAG_partial_unit";
      break;
    case 0x3Du:
      result = "DW_TAG_imported_unit";
      break;
    case 0x3Fu:
      result = "DW_TAG_condition";
      break;
    case 0x40u:
      result = "DW_TAG_shared_type";
      break;
    case 0x41u:
      result = "DW_TAG_type_unit";
      break;
    case 0x42u:
      result = "DW_TAG_rvalue_reference_type";
      break;
    case 0x43u:
      result = "DW_TAG_template_alias";
      break;
    case 0x44u:
      result = "DW_TAG_coarray_type";
      break;
    case 0x45u:
      result = "DW_TAG_generic_subrange";
      break;
    case 0x46u:
      result = "DW_TAG_dynamic_type";
      break;
    case 0x47u:
      result = "DW_TAG_atomic_type";
      break;
    case 0x48u:
      result = "DW_TAG_call_site";
      break;
    case 0x49u:
      result = "DW_TAG_call_site_parameter";
      break;
    case 0x4Au:
      result = "DW_TAG_skeleton_unit";
      break;
    case 0x4Bu:
      result = "DW_TAG_immutable_type";
      break;
    default:
      if ( a1 > 0x410A )
      {
        if ( a1 == 16896 )
        {
          result = "DW_TAG_APPLE_property";
        }
        else
        {
          switch ( a1 )
          {
            case 0xB000u:
              result = "DW_TAG_BORLAND_property";
              break;
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
              goto LABEL_7;
          }
        }
      }
      else if ( a1 <= 0x4100 )
      {
        result = "DW_TAG_MIPS_loop";
        if ( a1 != 16513 )
LABEL_7:
          result = 0;
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
            goto LABEL_7;
        }
      }
      break;
  }
  return result;
}
