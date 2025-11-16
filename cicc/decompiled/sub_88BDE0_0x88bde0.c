// Function: sub_88BDE0
// Address: 0x88bde0
//
int __fastcall sub_88BDE0(const char *a1)
{
  fprintf(qword_4F07510, "/* Target configuration: %s */\n", a1);
  fwrite("/* NOTE: For multiple target configurations, change _1 below as necessary. */\n", 1u, 0x4Eu, qword_4F07510);
  fprintf(qword_4F07510, "#define TARGET_CONFIGURATION_1 %s\n", a1);
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_ALIGNOF_DOUBLE", a1, "8");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_ALIGNOF_FLOAT", a1, "4");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_ALIGNOF_FLOAT128", a1, "16");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_ALIGNOF_FLOAT80", a1, "16");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_ALIGNOF_INT", a1, "4");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_ALIGNOF_INT128", a1, "16");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_ALIGNOF_LONG", a1, "8");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_ALIGNOF_LONG_DOUBLE", a1, "16");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_ALIGNOF_LONG_LONG", a1, "8");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_ALIGNOF_POINTER", a1, "8");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_ALIGNOF_PTR_TO_DATA_MEMBER", a1, "8");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_ALIGNOF_PTR_TO_MEMBER_FUNCTION", a1, "8");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_MICROSOFT_PTR_TO_MEMBER_SIZING", a1, "0");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_ALIGNOF_SHORT", a1, "2");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_ALIGNOF_VIRTUAL_FUNCTION_INFO", a1, "8");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_ALL_POINTERS_SAME_SIZE", a1, "1");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_BIT_FIELD_AFFECTS_UNION_ALIGNMENT", a1, "1");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_BIT_FIELD_CONTAINER_SIZE", a1, "(-1)");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_BOOL_INT_KIND", a1, "((an_integer_kind)ik_char)");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_C_BOOL_INT_KIND", a1, "((an_integer_kind)ik_unsigned_char)");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_CHAR16_T_INT_KIND", a1, "((an_integer_kind)ik_unsigned_short)");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_CHAR32_T_INT_KIND", a1, "((an_integer_kind)ik_unsigned_int)");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_CHAR_BIT", a1, "8");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_CHAR_CONSTANT_FIRST_CHAR_MOST_SIGNIFICANT", a1, "1");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_DBL_MANT_DIG", a1, "53");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_DBL_MAX_EXP", a1, "1024");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_DBL_MIN_EXP", a1, "(-1021)");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_DEFAULT_NEW_ALIGNMENT", a1, "16");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_DELTA_INT_KIND", a1, "((an_integer_kind)ik_long)");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_DUAL_ALIGNMENTS_FOR_BUILTIN_TYPES", a1, "0");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_ENUM_BIT_FIELDS_ARE_ALWAYS_UNSIGNED", a1, "0");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_ENUM_TYPES_CAN_BE_SMALLER_THAN_INT", a1, "0");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_FIELD_ALLOC_SEQUENCE_EQUALS_DECL_SEQUENCE", a1, "1");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_FLT_MANT_DIG", a1, "24");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_FLT_MAX_EXP", a1, "128");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_FLT_MIN_EXP", a1, "(-125)");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_FLT128_MANT_DIG", a1, "113");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_FLT128_MAX_EXP", a1, "(16384)");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_FLT128_MIN_EXP", a1, "(-16381)");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_FLT80_MANT_DIG", a1, "64");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_FLT80_MAX_EXP", a1, "(16384)");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_FLT80_MIN_EXP", a1, "(-16381)");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_FORCE_ONE_BIT_BIT_FIELD_TO_BE_UNSIGNED", a1, "0");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_HAS_SIGNED_CHARS", a1, "1");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_HOST_STRING_CHAR_BIT", a1, "8");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_IA64_ABI_USE_GUARD_ACQUIRE_RELEASE", a1, "0");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_IA64_ABI_USE_INT_STATIC_INIT_GUARD", a1, "0");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_IA64_ABI_USE_VARIANT_ARRAY_COOKIES", a1, "0");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_IA64_ABI_USE_VARIANT_PTR_TO_MEMBER_FUNCTION_REPR", a1, "0");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_IA64_ABI_VARIANT_CTORS_AND_DTORS_RETURN_THIS", a1, "0");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_IA64_ABI_VARIANT_KEY_FUNCTION", a1, "0");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_IA64_VTABLE_ENTRY_INT_KIND", a1, "((an_integer_kind)ik_long)");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_JMP_BUF_ELEMENTS_ARE_FLOAT", a1, "0");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_JMP_BUF_ELEMENT_FLOAT_KIND", a1, "((a_float_kind)fk_long_double)");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_JMP_BUF_ELEMENT_INT_KIND", a1, "((an_integer_kind)ik_long)");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_JMP_BUF_NUM_ELEMENTS", a1, "25");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_SETJMP_FUNC", a1, "\"_setjmp\"");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_LDBL_MANT_DIG", a1, "64");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_LDBL_MAX_EXP", a1, "16384");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_LDBL_MIN_EXP", a1, "(-16381)");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_LIBGCC_CMP_RETURN_MODE", a1, "((a_type_mode_kind)tmk_DI)");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_LIBGCC_SHIFT_COUNT_MODE", a1, "((a_type_mode_kind)tmk_DI)");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_LITTLE_ENDIAN", a1, "1");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_MAXIMUM_INTRINSIC_ALIGNMENT", a1, "16");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_MAX_BASE_CLASS_OFFSET", a1, "0");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_MAX_CLASS_OBJECT_SIZE", a1, "0");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_MICROSOFT_BIT_FIELD_ALLOCATION", a1, "0");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_MINIMUM_STRUCT_ALIGNMENT", a1, "1");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_NONNEGATIVE_ENUM_BIT_FIELD_IS_UNSIGNED", a1, "1");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_OPTIMIZE_EMPTY_BASE_CLASS_LAYOUT", a1, "1");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_PAD_BIT_FIELDS_LARGER_THAN_BASE_TYPE", a1, "1");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_PLAIN_INT_BIT_FIELD_IS_UNSIGNED", a1, "0");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_POINTER_MODE", a1, "((a_type_mode_kind)tmk_DI)");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_PTRDIFF_T_INT_KIND", a1, "((an_integer_kind)ik_long)");
  fprintf(
    qword_4F07510,
    "#define %s_%s %s\n",
    "TARG_REGION_NUMBER_INT_KIND",
    a1,
    "((an_integer_kind)ik_unsigned_short)");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_ETS_FLAG_TYPE_INT_KIND", a1, "((an_integer_kind)ik_unsigned_int)");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_REUSE_TAIL_PADDING", a1, "1");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_RIGHT_SHIFT_IS_ARITHMETIC", a1, "1");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_SIZEOF_DOUBLE", a1, "8");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_SIZEOF_FLOAT", a1, "4");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_SIZEOF_FLOAT128", a1, "16");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_SIZEOF_FLOAT80", a1, "16");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_SIZEOF_INT", a1, "4");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_SIZEOF_INT128", a1, "16");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_SIZEOF_LONG", a1, "8");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_SIZEOF_LONG_DOUBLE", a1, "16");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_SIZEOF_LONG_LONG", a1, "8");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_SIZEOF_LARGEST_ATOMIC", a1, "(8+8)");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_SIZEOF_POINTER", a1, "8");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_SIZEOF_PTR_TO_DATA_MEMBER", a1, "8");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_SIZEOF_PTR_TO_MEMBER_FUNCTION", a1, "(8+8)");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_SIZEOF_SHORT", a1, "2");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_SIZEOF_VIRTUAL_FUNCTION_INFO", a1, "8");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_SIZE_T_INT_KIND", a1, "((an_integer_kind)ik_unsigned_long)");
  fprintf(
    qword_4F07510,
    "#define %s_%s %s\n",
    "TARG_SIZE_T_MAX",
    a1,
    "((a_targ_size_t)(0x7fffffffffffffffL * 2UL + 1UL))");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_SSIZE_T_INT_KIND", a1, "((an_integer_kind)ik_long)");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_SUPPORTS_ARM32", a1, "0");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_SUPPORTS_ARM64", a1, "0");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_SUPPORTS_X86_64", a1, "1");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_TOO_LARGE_SHIFT_COUNT_IS_TAKEN_MODULO_SIZE", a1, "0");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_UNNAMED_BIT_FIELD_AFFECTS_STRUCT_ALIGNMENT", a1, "0");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_UNWIND_WORD_MODE", a1, "((a_type_mode_kind)tmk_DI)");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_USER_CONTROL_OF_STRUCT_PACKING_AFFECTS_BASE_CLASSES", a1, "1");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_USER_CONTROL_OF_STRUCT_PACKING_AFFECTS_BIT_FIELDS", a1, "1");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_VAR_HANDLE_INT_KIND", a1, "((an_integer_kind)ik_unsigned_short)");
  fprintf(
    qword_4F07510,
    "#define %s_%s %s\n",
    "TARG_VIRTUAL_FUNCTION_INDEX_INT_KIND",
    a1,
    "((an_integer_kind)ik_short)");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_WCHAR_T_INT_KIND", a1, "((an_integer_kind)ik_int)");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_WINT_T_INT_KIND", a1, "((an_integer_kind)ik_unsigned_int)");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_WORD_MODE", a1, "((a_type_mode_kind)tmk_DI)");
  fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_ZERO_WIDTH_BIT_FIELD_AFFECTS_STRUCT_ALIGNMENT", a1, "0");
  return fprintf(qword_4F07510, "#define %s_%s %s\n", "TARG_ZERO_WIDTH_BIT_FIELD_ALIGNMENT", a1, "(-1)");
}
