// Function: sub_706250
// Address: 0x706250
//
_DWORD *sub_706250()
{
  int v0; // eax
  int v1; // eax
  __int64 v2; // r13
  unsigned __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rax
  __m128i v8; // xmm2
  __m128i v9; // xmm3
  __m128i v10; // xmm0
  _QWORD *v11; // r15
  _QWORD *v12; // r12
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  __m128i v16; // xmm5
  __m128i v17; // xmm6
  __m128i v18; // xmm0
  __int64 v19; // r13
  __int64 v20; // rbx
  __int64 v21; // rax
  __int64 v22; // r13
  __int64 v23; // rsi
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // rax
  _QWORD *v38; // rdi
  __int64 v39; // rsi
  __int64 v40; // rdx
  __int64 v41; // rcx
  __int64 v42; // r8
  unsigned __int64 v43; // [rsp+10h] [rbp-70h]

  word_4F063FC[0] = 0;
  dword_4F063F8 = 0;
  *(_QWORD *)dword_4F07508 = *(_QWORD *)&dword_4F063F8;
  sub_8268E0();
  sub_8269D0();
  sub_5D29C0();
  sub_823EE0();
  sub_723990();
  sub_67ED40();
  sub_71AD50();
  sub_737E60();
  sub_77FA60();
  sub_666E90();
  sub_640900();
  sub_7CA870();
  sub_887B30();
  sub_867E30();
  sub_89F470();
  sub_8D05B0();
  sub_69A7E0();
  sub_6F0590();
  sub_875470();
  sub_603AA0();
  sub_7AB650();
  sub_8228E0();
  sub_858D80();
  sub_7F4E60();
  sub_7DABF0();
  sub_88B710();
  if ( !unk_4D03FE8 )
  {
    unk_4D03CAC = 1;
    dword_4D03C90 = 0;
  }
  sub_885C00(77, "auto");
  sub_885C00(78, "break");
  sub_885C00(79, "case");
  sub_885C00(80, "char");
  sub_885C00(82, "continue");
  sub_885C00(83, "default");
  sub_885C00(84, "do");
  sub_885C00(85, "double");
  sub_885C00(86, "else");
  sub_885C00(87, "enum");
  sub_885C00(88, "extern");
  sub_885C00(89, "float");
  sub_885C00(90, "for");
  sub_885C00(91, "goto");
  sub_885C00(92, "if");
  sub_885C00(93, "int");
  sub_885C00(94, "long");
  sub_885C00(95, "register");
  sub_885C00(96, "return");
  sub_885C00(97, "short");
  sub_885C00(99, "sizeof");
  sub_885C00(100, "static");
  sub_885C00(101, "struct");
  sub_885C00(102, "switch");
  sub_885C00(103, "typedef");
  sub_885C00(104, "union");
  sub_885C00(105, "unsigned");
  sub_885C00(106, "void");
  sub_885C00(108, "while");
  if ( dword_4F077C4 != 1 )
  {
    sub_885C00(81, "const");
    sub_885C00(98, "signed");
    sub_885C00(107, "volatile");
  }
  if ( unk_4D04398 )
  {
    sub_885C00(120, "_Bool");
    if ( dword_4F077C4 != 2 )
    {
      v0 = unk_4F07778;
      if ( unk_4F07778 <= 202310 )
      {
        if ( HIDWORD(qword_4F077B4) )
          goto LABEL_141;
        goto LABEL_9;
      }
      sub_885C00(120, "bool");
    }
  }
  if ( HIDWORD(qword_4F077B4) )
  {
LABEL_141:
    sub_885C00(121, "_Complex");
    sub_705950(0x79u, "__complex");
    sub_705950(0x90u, "__real");
    sub_705950(0x91u, "__imag");
    sub_885C00(123, "__I__");
    goto LABEL_142;
  }
  if ( dword_4F077C4 == 2 )
    goto LABEL_12;
  v0 = unk_4F07778;
LABEL_9:
  if ( v0 <= 199900 )
    goto LABEL_10;
  sub_885C00(121, "_Complex");
  sub_885C00(122, "_Imaginary");
  sub_885C00(123, "__I__");
LABEL_142:
  if ( dword_4F077C4 == 2
    || unk_4F07778 <= 199900
    || (sub_885C00(154, "inline"), sub_885C00(109, "__generic"), dword_4F077C4 == 2)
    || unk_4F07778 <= 199900 )
  {
LABEL_10:
    if ( !HIDWORD(qword_4F077B4) )
      goto LABEL_12;
  }
  sub_885C00(261, "__builtin_complex");
LABEL_12:
  if ( unk_4D044C4 )
    sub_885C00(260, "_Noreturn");
  if ( dword_4F077C4 != 2 && unk_4F07778 > 201111 )
  {
LABEL_16:
    sub_885C00(262, "_Generic");
    if ( unk_4D044D0 )
      goto LABEL_17;
    goto LABEL_34;
  }
  if ( dword_4F077C0 )
  {
    if ( !(_DWORD)qword_4F077B4 )
    {
      if ( qword_4F077A8 > 0x9FC3u )
        goto LABEL_16;
      goto LABEL_33;
    }
  }
  else if ( !(_DWORD)qword_4F077B4 )
  {
    goto LABEL_33;
  }
  if ( qword_4F077A0 > 0x752Fu )
    goto LABEL_16;
LABEL_33:
  if ( unk_4D044D0 )
  {
LABEL_17:
    sub_885C00(263, "_Atomic");
    if ( dword_4F077C4 == 2 )
      goto LABEL_18;
LABEL_35:
    if ( !unk_4D043F0 )
      goto LABEL_36;
    goto LABEL_21;
  }
LABEL_34:
  if ( dword_4F077C4 != 2 )
    goto LABEL_35;
LABEL_18:
  if ( !(_DWORD)qword_4F077B4 )
    goto LABEL_39;
  if ( qword_4F077A0 <= 0x75F7u || !unk_4D043F0 )
    goto LABEL_25;
LABEL_21:
  sub_885C00(247, "_Alignof");
  if ( dword_4F077C4 == 2 || unk_4F07778 > 202310 && (sub_885C00(247, "alignof"), dword_4F077C4 == 2) )
  {
    if ( !(_DWORD)qword_4F077B4 )
      goto LABEL_39;
LABEL_25:
    if ( qword_4F077A0 <= 0x765Bu || !unk_4D04220 )
      goto LABEL_39;
    goto LABEL_27;
  }
LABEL_36:
  if ( unk_4D04220 )
  {
LABEL_27:
    sub_885C00(194, "_Thread_local");
    if ( dword_4F077C4 == 2 )
      goto LABEL_39;
    if ( unk_4F07778 > 202310 )
    {
      sub_885C00(194, "thread_local");
      if ( dword_4F077C4 == 2 )
        goto LABEL_39;
    }
  }
  if ( unk_4D043F4 )
  {
    sub_885C00(248, "_Alignas");
    if ( dword_4F077C4 != 2 && unk_4F07778 > 202310 )
      sub_885C00(248, "alignas");
  }
  if ( unk_4F07764 )
  {
    sub_885C00(184, "_Static_assert");
    if ( dword_4F077C4 != 2 && unk_4F07778 > 202310 )
      sub_885C00(184, "static_assert");
  }
LABEL_39:
  sub_885C00(124, "__NAN__");
  sub_885C00(125, "__INFINITY__");
  sub_885C00(111, "__ALIGNOF__");
  sub_885C00(111, "__alignof__");
  sub_885C00(112, "__INTADDR__");
  if ( unk_4D044C8 )
    sub_885C00(118, "restrict");
  if ( unk_4D044C0 )
    sub_705950(0x77u, "__restrict");
  sub_885C00(138, "__func__");
  sub_885C00(139, "__FUNCTION__");
  sub_885C00(140, "__PRETTY_FUNCTION__");
  if ( unk_4D04548 | unk_4D04558 )
  {
    sub_885C00(132, "__declspec");
    sub_885C00(132, "_declspec");
    if ( unk_4F06AD1 != 13 )
    {
      sub_885C00(133, "__int8");
      sub_885C00(133, "_int8");
    }
    if ( unk_4F06ACF != 13 )
    {
      sub_885C00(134, "__int16");
      sub_885C00(134, "_int16");
    }
    if ( unk_4F06ACD != 13 )
    {
      sub_885C00(135, "__int32");
      sub_885C00(135, "_int32");
    }
    if ( unk_4F06ACB != 13 )
    {
      sub_885C00(136, "__int64");
      sub_885C00(136, "_int64");
    }
  }
  if ( dword_4F077C4 == 2 )
  {
    if ( !(_DWORD)qword_4F077B4 )
    {
      if ( !HIDWORD(qword_4F077B4) )
      {
        if ( !dword_4D044B0 )
          goto LABEL_54;
        goto LABEL_254;
      }
      if ( qword_4F077A8 <= 0x222DFu )
      {
        if ( !dword_4D044B0 )
          goto LABEL_54;
        goto LABEL_136;
      }
    }
    unk_4F07794 = 1;
    sub_885C00(238, "__internal_alias_decl");
  }
  if ( dword_4D044B0 )
  {
    if ( !HIDWORD(qword_4F077B4) )
    {
      if ( !(_DWORD)qword_4F077B4 )
        goto LABEL_254;
      goto LABEL_218;
    }
    if ( (_DWORD)qword_4F077B4 )
    {
LABEL_218:
      if ( qword_4F077A0 > 0x2E62Fu )
      {
LABEL_219:
        sub_885C00(298, "__is_layout_compatible");
        sub_885C00(299, "__is_pointer_interconvertible_base_of");
        if ( HIDWORD(qword_4F077B4) && !(_DWORD)qword_4F077B4 && qword_4F077A8 > 0x1D4BFu )
        {
          sub_885C00(303, "__builtin_is_corresponding_member");
          sub_885C00(301, "__builtin_is_pointer_interconvertible_with_class");
        }
      }
LABEL_254:
      sub_885C00(197, "__has_nothrow_assign");
      sub_885C00(198, "__has_nothrow_constructor");
      sub_885C00(199, "__has_nothrow_copy");
      sub_885C00(200, "__has_trivial_assign");
      sub_885C00(201, "__has_trivial_constructor");
      sub_885C00(202, "__has_trivial_copy");
      sub_885C00(203, "__has_trivial_destructor");
      sub_885C00(205, "__has_virtual_destructor");
      sub_885C00(206, "__is_abstract");
      sub_885C00(207, "__is_base_of");
      sub_885C00(208, "__is_class");
      sub_885C00(209, "__is_convertible_to");
      sub_885C00(212, "__is_empty");
      sub_885C00(213, "__is_enum");
      sub_885C00(293, "__is_function");
      sub_885C00(215, "__is_pod");
      sub_885C00(216, "__is_polymorphic");
      sub_885C00(217, "__is_union");
      sub_885C00(218, "__is_trivial");
      sub_885C00(219, "__is_standard_layout");
      sub_885C00(220, "__is_trivially_copyable");
      sub_885C00(221, "__is_literal_type");
      if ( (_DWORD)qword_4F077B4 )
        goto LABEL_288;
      if ( !HIDWORD(qword_4F077B4) )
        goto LABEL_305;
      if ( qword_4F077A8 > 0x1FBCFu )
      {
LABEL_288:
        sub_885C00(210, "__is_convertible");
        if ( (_DWORD)qword_4F077B4 )
        {
LABEL_289:
          sub_885C00(305, "__is_array");
          sub_885C00(336, "__is_bounded_array");
          sub_885C00(311, "__is_const");
          sub_885C00(316, "__is_member_function_pointer");
          sub_885C00(317, "__is_member_object_pointer");
          sub_885C00(318, "__is_member_pointer");
          sub_885C00(319, "__is_object");
          sub_885C00(321, "__is_reference");
          if ( (_DWORD)qword_4F077B4 )
            goto LABEL_290;
          if ( !HIDWORD(qword_4F077B4) )
            goto LABEL_305;
LABEL_258:
          if ( qword_4F077A8 <= 0x249EFu )
            goto LABEL_259;
LABEL_290:
          sub_885C00(306, "__array_rank");
          sub_885C00(337, "__is_unbounded_array");
          sub_885C00(327, "__is_volatile");
          if ( (_DWORD)qword_4F077B4 )
          {
            sub_885C00(307, "__array_extent");
            sub_885C00(308, "__is_arithmetic");
            sub_885C00(309, "__is_complete_type");
            sub_885C00(310, "__is_compound");
            sub_885C00(312, "__is_floating_point");
            sub_885C00(313, "__is_fundamental");
            sub_885C00(314, "__is_integral");
            sub_885C00(315, "__is_lvalue_reference");
            sub_885C00(322, "__is_rvalue_reference");
            sub_885C00(320, "__is_pointer");
            sub_885C00(323, "__is_scalar");
            sub_885C00(325, "__is_unsigned");
            sub_885C00(326, "__is_void");
            sub_885C00(292, "__is_same_as");
            sub_885C00(288, "__reference_binds_to_temporary");
            sub_885C00(338, "__is_referenceable");
            sub_885C00(221, "__is_literal");
            sub_885C00(210, "__is_convertible");
            if ( qword_4F077A0 > 0x249EFu )
            {
              sub_885C00(356, "__is_trivially_relocatable");
              if ( qword_4F077A0 > 0x2980Fu )
              {
                sub_885C00(355, "__is_trivially_equality_comparable");
                if ( qword_4F077A0 > 0x2BF1Fu )
                  sub_885C00(289, "__reference_constructs_from_temporary");
              }
            }
            if ( HIDWORD(qword_4F077B4) )
            {
              if ( (_DWORD)qword_4F077B4 )
              {
                if ( qword_4F077A0 <= 0x2BF1Fu )
                  goto LABEL_308;
                goto LABEL_295;
              }
LABEL_259:
              if ( qword_4F077A8 <= 0x1FBCFu )
                goto LABEL_260;
LABEL_295:
              sub_885C00(289, "__reference_constructs_from_temporary");
LABEL_296:
              if ( !HIDWORD(qword_4F077B4) )
                goto LABEL_297;
LABEL_308:
              if ( (_DWORD)qword_4F077B4 )
              {
                if ( qword_4F077A0 <= 0x2E62Fu )
                {
LABEL_310:
                  if ( (_DWORD)qword_4F077B4 )
                  {
LABEL_265:
                    if ( qword_4F077A0 > 0x270FFu )
LABEL_266:
                      sub_885C00(214, "__is_scoped_enum");
LABEL_267:
                    sub_885C00(222, "__has_trivial_move_constructor");
                    sub_885C00(223, "__has_trivial_move_assign");
                    sub_885C00(224, "__has_nothrow_move_assign");
                    sub_885C00(227, "__is_constructible");
                    sub_885C00(228, "__is_nothrow_constructible");
                    sub_885C00(229, "__is_trivially_constructible");
                    sub_885C00(230, "__is_destructible");
                    sub_885C00(231, "__is_nothrow_destructible");
                    sub_885C00(232, "__is_trivially_destructible");
                    sub_885C00(270, "__is_assignable");
                    sub_885C00(233, "__is_nothrow_assignable");
                    sub_885C00(234, "__is_trivially_assignable");
                    sub_885C00(236, "__underlying_type");
                    sub_885C00(242, "__is_final");
                    sub_885C00(285, "__has_unique_object_representations");
                    sub_885C00(286, "__is_aggregate");
                    sub_885C00(304, "__edg_is_deducible");
                    goto LABEL_54;
                  }
LABEL_299:
                  if ( qword_4F077A8 <= 0x222DFu )
                    goto LABEL_267;
                  goto LABEL_266;
                }
                goto LABEL_262;
              }
LABEL_261:
              if ( qword_4F077A8 <= 0x1FBCFu )
                goto LABEL_299;
LABEL_262:
              sub_885C00(211, "__is_nothrow_convertible");
              sub_885C00(290, "__reference_converts_from_temporary");
LABEL_263:
              if ( !HIDWORD(qword_4F077B4) )
              {
                if ( !(_DWORD)qword_4F077B4 )
                  goto LABEL_267;
                goto LABEL_265;
              }
              goto LABEL_310;
            }
          }
          else if ( HIDWORD(qword_4F077B4) )
          {
            goto LABEL_259;
          }
LABEL_305:
          if ( !(_DWORD)qword_4F077B4 )
          {
LABEL_260:
            if ( HIDWORD(qword_4F077B4) )
              goto LABEL_261;
LABEL_297:
            if ( !(_DWORD)qword_4F077B4 )
            {
              if ( !HIDWORD(qword_4F077B4) )
                goto LABEL_267;
              goto LABEL_299;
            }
            if ( qword_4F077A0 <= 0x2E62Fu )
              goto LABEL_263;
            goto LABEL_262;
          }
          if ( qword_4F077A0 <= 0x2BF1Fu )
            goto LABEL_296;
          goto LABEL_295;
        }
        if ( !HIDWORD(qword_4F077B4) )
          goto LABEL_305;
      }
      if ( qword_4F077A8 <= 0x222DFu )
        goto LABEL_258;
      goto LABEL_289;
    }
LABEL_136:
    if ( qword_4F077A8 > 0x1D4BFu )
      goto LABEL_219;
    goto LABEL_254;
  }
LABEL_54:
  if ( dword_4F077C0 && (dword_4F077C4 == 2 || unk_4F07778 <= 199900) )
  {
    sub_885C00(154, "inline");
    v1 = qword_4F077B4;
    if ( HIDWORD(qword_4F077B4) )
      goto LABEL_60;
  }
  else
  {
    if ( dword_4F077BC )
      sub_885C00(188, "__null");
    v1 = qword_4F077B4;
    if ( HIDWORD(qword_4F077B4) )
    {
LABEL_60:
      if ( !v1 && qword_4F077A8 <= 0x9C3Fu )
        goto LABEL_95;
      goto LABEL_62;
    }
  }
  if ( !v1 )
  {
    if ( !dword_4F07738 )
      goto LABEL_64;
    goto LABEL_87;
  }
LABEL_62:
  sub_885C00(117, "__builtin_offsetof");
  if ( !HIDWORD(qword_4F077B4) )
  {
    if ( !dword_4F07738 )
      goto LABEL_64;
LABEL_87:
    sub_885C00(189, "typeof");
    goto LABEL_96;
  }
LABEL_95:
  sub_705950(0xBDu, "typeof");
LABEL_96:
  if ( dword_4F07738 )
  {
    sub_885C00(190, "typeof_unqual");
    if ( !HIDWORD(qword_4F077B4) )
      goto LABEL_65;
    goto LABEL_98;
  }
LABEL_64:
  if ( !HIDWORD(qword_4F077B4) )
    goto LABEL_65;
LABEL_98:
  sub_885C00(187, "__extension__");
  if ( !HIDWORD(qword_4F077B4) )
    goto LABEL_65;
  if ( dword_4F077C0 && qword_4F077A8 > 0x9FC3u )
    sub_885C00(186, "__auto_type");
  if ( dword_4F077BC && qword_4F077A8 > 0x76BFu )
    sub_705950(0x70u, "__offsetof");
  sub_705950(0x8Fu, "__builtin_types_compatible_p");
  if ( unk_4D04548 | unk_4D04558 || (v4 = qword_4F077A8, qword_4F077A8 > 0x9E97u) )
  {
    if ( unk_4D04290 )
      sub_885C00(239, "__int128");
    v4 = qword_4F077A8;
    if ( qword_4F077A8 > 0x9EFBu )
    {
      sub_885C00(249, "__bases");
      sub_885C00(250, "__direct_bases");
      v4 = qword_4F077A8;
    }
  }
  if ( dword_4F077BC && !(_DWORD)qword_4F077B4 )
  {
    if ( v4 <= 0x1116F )
      goto LABEL_115;
    sub_885C00(292, "__is_same_as");
    v4 = qword_4F077A8;
    if ( dword_4F077BC )
    {
      if ( !(_DWORD)qword_4F077B4 )
      {
        if ( qword_4F077A8 <= 0x1387Fu )
          goto LABEL_115;
        sub_885C00(287, "__integer_pack");
        v4 = qword_4F077A8;
        if ( dword_4F077BC && !(_DWORD)qword_4F077B4 && qword_4F077A8 > 0x1869Fu )
        {
          sub_885C00(291, "__is_same");
          v4 = qword_4F077A8;
        }
      }
    }
  }
  if ( v4 > 0x15F8F )
    sub_705950(0x128u, "__builtin_has_attribute");
LABEL_115:
  if ( dword_4F077C0 )
  {
    if ( (_DWORD)qword_4F077B4 || qword_4F077A8 <= 0x1116Fu )
      goto LABEL_117;
  }
  else if ( !dword_4F077BC || (_DWORD)qword_4F077B4 || qword_4F077A8 <= 0x1FBCFu )
  {
    goto LABEL_117;
  }
  sub_885C00(331, "_Float32");
  sub_885C00(332, "_Float32x");
  sub_885C00(333, "_Float64");
  sub_885C00(334, "_Float64x");
  sub_885C00(335, "_Float128");
LABEL_117:
  sub_705950(0x9Au, "__inline");
  sub_705950(0x95u, "__asm");
  sub_705950(0x51u, "__const");
  sub_705950(0x62u, "__signed");
  sub_705950(0x6Bu, "__volatile");
  sub_885C00(111, "__alignof");
  if ( qword_4F077A8 >= (unsigned __int64)(dword_4F077C0 == 0 ? 0x64 : 0) + 40700 )
    sub_885C00(257, "__builtin_shuffle");
  if ( (_DWORD)qword_4F077B4 || HIDWORD(qword_4F077B4) && qword_4F077A8 > 0x1D4BFu )
    goto LABEL_122;
  if ( !unk_4D0455C )
  {
    if ( qword_4F077A8 <= 0x15F8Fu )
      goto LABEL_124;
    goto LABEL_123;
  }
  if ( unk_4D04600 > 0x2E693u && unk_4D045F8 )
  {
LABEL_122:
    sub_885C00(258, "__builtin_shufflevector");
    if ( !(_DWORD)qword_4F077B4 && qword_4F077A8 <= 0x15F8Fu )
    {
      if ( !unk_4D0455C )
        goto LABEL_124;
      goto LABEL_129;
    }
  }
  else if ( qword_4F077A8 <= 0x15F8Fu )
  {
LABEL_129:
    if ( unk_4D04600 <= 0x2E693u || !unk_4D045F8 )
      goto LABEL_124;
  }
LABEL_123:
  sub_885C00(259, "__builtin_convertvector");
LABEL_124:
  sub_885C00(273, "__edg_vector_type__");
  sub_885C00(274, "__edg_neon_vector_type__");
  sub_885C00(275, "__edg_neon_polyvector_type__");
  sub_885C00(276, "__edg_scalable_vector_type__");
LABEL_65:
  if ( (_DWORD)qword_4F077B4 || HIDWORD(qword_4F077B4) && qword_4F077A8 > 0x1116Fu )
  {
    sub_885C00(271, "__builtin_addressof");
    if ( (_DWORD)qword_4F077B4 )
    {
      if ( qword_4F077A0 > 0x15F8Fu )
        goto LABEL_92;
LABEL_69:
      if ( unk_4D0431C )
        goto LABEL_93;
      goto LABEL_70;
    }
  }
  if ( !dword_4F077BC || qword_4F077A8 <= 0x1ADAFu )
    goto LABEL_69;
LABEL_92:
  sub_885C30(297, "__builtin_bit_cast");
  if ( unk_4D0431C )
  {
LABEL_93:
    sub_885C00(264, "_Nullable");
    sub_885C00(265, "_Nonnull");
    sub_885C00(266, "_Null_unspecified");
    if ( !dword_4D043E0 )
      goto LABEL_71;
    goto LABEL_94;
  }
LABEL_70:
  if ( !dword_4D043E0 )
    goto LABEL_71;
LABEL_94:
  sub_705950(0x8Eu, "__attribute");
LABEL_71:
  if ( dword_4F077C4 | unk_4D04630 || !dword_4D04964 )
  {
    sub_885C00(149, "asm");
    if ( dword_4F077C4 == 2 )
    {
      sub_885C00(150, "catch");
      sub_885C00(151, "class");
      sub_885C00(153, "friend");
      sub_885C00(154, "inline");
      sub_885C00(174, "mutable");
      sub_885C00(156, "operator");
      sub_885C00(157, "private");
      sub_885C00(158, "protected");
      sub_885C00(159, "public");
      sub_885C00(160, "template");
      sub_885C00(161, "this");
      sub_885C00(162, "throw");
      sub_885C00(163, "try");
      sub_885C00(164, "virtual");
      sub_885C00(166, "const_cast");
      sub_885C00(177, "static_cast");
      sub_885C00(176, "reinterpret_cast");
      v5 = sub_885B80("delete", 6, 0, 0xFFFFFFFFLL);
      *(_BYTE *)(v5 + 90) |= 1u;
      *(_WORD *)(v5 + 88) = 152;
      v6 = sub_885B80("new", 3, 0, 0xFFFFFFFFLL);
      *(_BYTE *)(v6 + 90) |= 1u;
      *(_WORD *)(v6 + 88) = 155;
      if ( unk_4D04950 )
        sub_885C00(191, "overload");
      if ( unk_4D043A4 )
        sub_885C00(165, "wchar_t");
      if ( unk_4D041B4 )
        sub_885C00(128, "char8_t");
      if ( unk_4D043A0 )
      {
        sub_885C00(126, "char16_t");
        sub_885C00(127, "char32_t");
      }
      if ( (_DWORD)qword_4F077B4 )
      {
        sub_885C00(126, "__char16_t");
        sub_885C00(127, "__char32_t");
      }
      if ( unk_4D0439C )
      {
        sub_885C00(180, "bool");
        sub_885C00(181, "false");
        sub_885C00(182, "true");
      }
      if ( unk_4D04388 )
      {
        v27 = sub_885B80("and", 3, 0, 0xFFFFFFFFLL);
        *(_BYTE *)(v27 + 90) |= 1u;
        *(_WORD *)(v27 + 88) = 52;
        v28 = sub_885B80("and_eq", 6, 0, 0xFFFFFFFFLL);
        *(_BYTE *)(v28 + 90) |= 1u;
        *(_WORD *)(v28 + 88) = 64;
        v29 = sub_885B80("bitand", 6, 0, 0xFFFFFFFFLL);
        *(_BYTE *)(v29 + 90) |= 1u;
        *(_WORD *)(v29 + 88) = 33;
        v30 = sub_885B80("bitor", 5, 0, 0xFFFFFFFFLL);
        *(_BYTE *)(v30 + 90) |= 1u;
        *(_WORD *)(v30 + 88) = 51;
        v31 = sub_885B80("compl", 5, 0, 0xFFFFFFFFLL);
        *(_BYTE *)(v31 + 90) |= 1u;
        *(_WORD *)(v31 + 88) = 37;
        v32 = sub_885B80("not", 3, 0, 0xFFFFFFFFLL);
        *(_BYTE *)(v32 + 90) |= 1u;
        *(_WORD *)(v32 + 88) = 38;
        v33 = sub_885B80("not_eq", 6, 0, 0xFFFFFFFFLL);
        *(_BYTE *)(v33 + 90) |= 1u;
        *(_WORD *)(v33 + 88) = 48;
        v34 = sub_885B80("or", 2, 0, 0xFFFFFFFFLL);
        *(_BYTE *)(v34 + 90) |= 1u;
        *(_WORD *)(v34 + 88) = 53;
        v35 = sub_885B80("or_eq", 5, 0, 0xFFFFFFFFLL);
        *(_BYTE *)(v35 + 90) |= 1u;
        *(_WORD *)(v35 + 88) = 66;
        v36 = sub_885B80("xor", 3, 0, 0xFFFFFFFFLL);
        *(_BYTE *)(v36 + 90) |= 1u;
        *(_WORD *)(v36 + 88) = 50;
        v37 = sub_885B80("xor_eq", 6, 0, 0xFFFFFFFFLL);
        *(_BYTE *)(v37 + 90) |= 1u;
        *(_WORD *)(v37 + 88) = 65;
      }
      sub_885C00(167, "dynamic_cast");
      if ( unk_4D0484C )
      {
        sub_885C00(178, "typeid");
        if ( unk_4D04838 )
        {
LABEL_164:
          sub_885C00(175, "namespace");
          sub_885C00(179, "using");
          if ( unk_4D04830 )
          {
LABEL_165:
            sub_885C00(183, "typename");
            goto LABEL_166;
          }
LABEL_284:
          v15 = sub_885B80("typename", 8, 0, 0xFFFFFFFFLL);
          *(_WORD *)(v15 + 88) = 24;
          *(_DWORD *)(v15 + 92) = 560;
LABEL_166:
          if ( unk_4D04840 )
            sub_885C00(168, "explicit");
          if ( unk_4D04458 )
          {
            sub_885C00(170, "export");
          }
          else if ( unk_4D04274 )
          {
            sub_885C00(169, "export");
          }
          if ( unk_4F07764 )
            sub_885C00(184, "static_assert");
          if ( (_DWORD)qword_4F077B4 )
          {
            sub_885C00(184, "_Static_assert");
            sub_885C00(291, "__is_same");
          }
          if ( unk_4F07754 )
          {
            if ( dword_4F077BC )
              sub_885C00(185, "__decltype");
            if ( !unk_4F07744 )
              sub_885C00(185, "decltype");
          }
          if ( unk_4F0773C )
            sub_885C00(237, "nullptr");
          if ( (_DWORD)qword_4F077B4 )
            sub_885C00(237, "__nullptr");
          if ( dword_4D048B4 )
            sub_885C00(243, "noexcept");
          if ( word_4D04898 )
            sub_885C00(244, "constexpr");
          if ( unk_4D04894 )
            sub_885C00(245, "consteval");
          if ( unk_4D04890 )
            sub_885C00(246, "constinit");
          if ( dword_4F077BC && !(_DWORD)qword_4F077B4 && qword_4F077A8 > 0x1869Fu )
            sub_885C00(246, "__constinit");
          if ( unk_4D043F4 )
            sub_885C00(248, "alignas");
          if ( unk_4D043F0 )
          {
            if ( (_DWORD)qword_4F077B4 && (dword_4F077C4 != 2 || unk_4F07778 <= 201102 && !dword_4F07774) )
            {
LABEL_205:
              if ( qword_4F077A0 > 0x2BF1Fu )
                sub_885C00(284, "__datasizeof");
LABEL_207:
              if ( unk_4D04498 )
              {
                sub_885C00(267, "co_yield");
                sub_885C00(268, "co_return");
                sub_885C00(269, "co_await");
              }
              if ( unk_4D04494 )
              {
                sub_885C00(294, "requires");
                sub_885C00(295, "concept");
              }
              if ( !unk_4D04224 )
                goto LABEL_213;
              goto LABEL_212;
            }
            sub_885C00(247, "alignof");
          }
          if ( !(_DWORD)qword_4F077B4 )
            goto LABEL_207;
          goto LABEL_205;
        }
      }
      else
      {
        v7 = sub_885B80("typeid", 6, 0, 0xFFFFFFFFLL);
        *(_WORD *)(v7 + 88) = 24;
        *(_DWORD *)(v7 + 92) = 560;
        if ( unk_4D04838 )
          goto LABEL_164;
      }
      v13 = sub_885B80("namespace", 9, 0, 0xFFFFFFFFLL);
      *(_WORD *)(v13 + 88) = 24;
      *(_DWORD *)(v13 + 92) = 560;
      v14 = sub_885B80("using", 5, 0, 0xFFFFFFFFLL);
      *(_WORD *)(v14 + 88) = 24;
      *(_DWORD *)(v14 + 92) = 560;
      if ( unk_4D04830 )
        goto LABEL_165;
      goto LABEL_284;
    }
  }
  if ( !unk_4D04224 )
    goto LABEL_75;
LABEL_212:
  sub_885C00(192, "__thread");
LABEL_213:
  if ( dword_4F077C4 == 2 && unk_4D04220 )
    sub_885C00(193, "thread_local");
LABEL_75:
  sub_885C00(328, "__nv_is_extended_device_lambda_closure_type");
  sub_885C00(329, "__nv_is_extended_host_device_lambda_closure_type");
  sub_885C00(330, "__nv_is_extended_device_lambda_with_preserved_return_type");
  if ( (_DWORD)qword_4F077B4 && qword_4F077A0 > 0x270FFu )
  {
    sub_885C00(251, "__builtin_arm_ldrex");
    sub_885C00(252, "__builtin_arm_ldaex");
    sub_885C00(253, "__builtin_arm_addg");
    sub_885C00(254, "__builtin_arm_irg");
    sub_885C00(255, "__builtin_arm_ldg");
  }
  sub_885C00(272, "__edg_type__");
  sub_885C00(277, "__edg_size_type__");
  sub_885C00(278, "__edg_ptrdiff_type__");
  sub_885C00(279, "__edg_bool_type__");
  sub_885C00(280, "__edg_wchar_type__");
  sub_885C00(282, "__edg_opnd__");
  sub_885C00(281, "__edg_throw__");
  sub_822270(dest);
  v2 = unk_4D03FF0;
  *(_QWORD *)(v2 + 8) = sub_729740(unk_4F066A8);
  sub_860100(0);
  unk_4F07288 = *(_QWORD *)(unk_4D03FF0 + 8LL);
  unk_4F07290 = 0;
  unk_4F072C0 = 0;
  unk_4F07300 = 0;
  unk_4F07320 = 0;
  if ( dword_4F077C4 == 2 )
  {
    sub_87FD50();
    sub_87FCF0();
    if ( dword_4F077BC | dword_4D042AC )
    {
      v8 = _mm_loadu_si128(&xmmword_4F06660[1]);
      v9 = _mm_loadu_si128(&xmmword_4F06660[2]);
      v10 = _mm_loadu_si128(&xmmword_4F06660[3]);
      qword_4D04A00 = _mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
      unk_4D04A10 = v8;
      xmmword_4D04A20 = v9;
      unk_4D04A30 = v10;
      qword_4D04A08 = *(_QWORD *)&dword_4F077C8;
      sub_886510(&qword_4D04A00);
    }
    sub_897670();
    sub_8976D0();
    sub_87FD90();
    if ( dword_4F077BC )
    {
      v16 = _mm_loadu_si128(&xmmword_4F06660[1]);
      v17 = _mm_loadu_si128(&xmmword_4F06660[2]);
      v18 = _mm_loadu_si128(&xmmword_4F06660[3]);
      qword_4D04A00 = _mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
      unk_4D04A10 = v16;
      xmmword_4D04A20 = v17;
      unk_4D04A30 = v18;
      qword_4D04A08 = *(_QWORD *)&dword_4F077C8;
      sub_886F70(&qword_4D04A00);
    }
    v11 = &off_4B6D480;
    v12 = qword_4F06C80;
    do
    {
      if ( *v11 )
        *v12 = sub_736B10(9, *v11);
      ++v11;
      ++v12;
    }
    while ( v11 != (_QWORD *)algn_4B6D4D8 );
    qword_4F06CE0 = (_QWORD *)qword_4F06C80[0];
    if ( dword_4F077BC && qword_4D049B8 && dword_4F06900 )
    {
      if ( dword_4D042AC )
      {
        sub_736B60(qword_4F06C80[0], 0, &dword_4F077C8);
      }
      else
      {
        sub_8602E0(4, qword_4D049B8[11]);
        v38 = qword_4F06CE0;
        v39 = (unsigned int)dword_4F04C64;
        sub_736B60(qword_4F06CE0, (unsigned int)dword_4F04C64, &dword_4F077C8);
        sub_863FD0(v38, v39, v40, v41, v42);
      }
    }
    if ( dword_4D04818 )
    {
      v19 = sub_72BA30(byte_4F06A51[0]);
      v20 = sub_7259C0(2);
      *(_BYTE *)(v20 + 88) = *(_BYTE *)(v20 + 88) & 0x8F | 0x20;
      v21 = *(_QWORD *)(v19 + 128);
      *(_BYTE *)(v20 + 161) |= 0x1Cu;
      *(_QWORD *)(v20 + 128) = v21;
      *(_BYTE *)(v20 + 160) = byte_4F06A51[0];
      *(_QWORD *)(*(_QWORD *)(v20 + 176) + 8LL) = v19;
      v43 = _mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
      sub_878540("align_val_t", 0xBu);
      v22 = sub_87EBB0(6, v43);
      sub_877D80(v20, v22);
      *(_QWORD *)(v22 + 88) = v20;
      sub_8602E0(4, qword_4D049B8[11]);
      v23 = (unsigned int)dword_4F04C64;
      sub_736B60(v20, (unsigned int)dword_4F04C64, &dword_4F077C8);
      sub_863FD0(v20, v23, v24, v25, v26);
      *(_BYTE *)(v22 + 83) = ((dword_4F077BC == 0) << 6) | *(_BYTE *)(v22 + 83) & 0xBF;
      unk_4F06C60 = v20;
    }
    sub_8801C0(1, 0, 0);
    sub_8801C0(2, 0, 0);
    if ( unk_4D04478 )
    {
      sub_8801C0(2, 1, 0);
      if ( dword_4D04818 )
      {
        sub_8801C0(1, 0, 1);
        sub_8801C0(2, 1, 1);
        sub_8801C0(2, 0, 1);
      }
    }
    if ( unk_4D04844 )
    {
      sub_8801C0(3, 0, 0);
      sub_8801C0(4, 0, 0);
      if ( unk_4D04478 )
      {
        sub_8801C0(4, 1, 0);
        if ( dword_4D04818 )
        {
          sub_8801C0(3, 0, 1);
          sub_8801C0(4, 1, 1);
          sub_8801C0(4, 0, 1);
        }
      }
    }
  }
  sub_88A4A0();
  if ( unk_4D03FE8 )
    return (_DWORD *)sub_7C9DD0();
  unk_4D0493C = 0;
  dword_4D04944 = 0;
  return &dword_4D04944;
}
