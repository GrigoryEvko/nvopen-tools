// Function: sub_2095B00
// Address: 0x2095b00
//
__m128i *__fastcall sub_2095B00(__m128i *a1, __int64 a2, _QWORD *a3)
{
  unsigned int v4; // r8d
  __int16 v5; // ax
  __int64 v6; // rax
  _QWORD *v7; // rbx
  __int64 v8; // rax
  const char *v10; // rsi
  __int64 v11; // rax
  __int64 v12; // rdx
  __m128i v13; // xmm0
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rdx
  __m128i v18; // xmm0
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rdx
  __m128i v23; // xmm0
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rax
  __int64 v27; // rdx
  __m128i v28; // xmm0
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rax
  __int64 v32; // rdx
  __m128i v33; // xmm0
  __int64 v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rax
  __int64 v37; // rdx
  __m128i si128; // xmm0
  __int64 v39; // rax
  __int64 v40; // rdx
  __int64 v41; // rax
  __int64 v42; // rdx
  __m128i *v43; // r15
  const char *v44; // r14
  size_t v45; // rax
  size_t v46; // r13
  __int64 v47; // rax
  __int64 v48; // rdx
  __int64 v49; // rax
  __int64 v50[4]; // [rsp+0h] [rbp-70h] BYREF
  __m128i v51[5]; // [rsp+20h] [rbp-50h] BYREF

  v4 = *(unsigned __int16 *)(a2 + 24);
  v5 = *(_WORD *)(a2 + 24);
  switch ( (__int16)v4 )
  {
    case 1:
      sub_2094330(a1->m128i_i64, "EntryToken");
      return a1;
    case 2:
      sub_2094330(a1->m128i_i64, "TokenFactor");
      return a1;
    case 3:
      sub_2094330(a1->m128i_i64, "AssertSext");
      return a1;
    case 4:
      sub_2094330(a1->m128i_i64, "AssertZext");
      return a1;
    case 5:
      sub_2094330(a1->m128i_i64, "BasicBlock");
      return a1;
    case 6:
      sub_2094330(a1->m128i_i64, "ValueType");
      return a1;
    case 7:
      switch ( *(_DWORD *)(a2 + 84) )
      {
        case 0:
          sub_2094330(a1->m128i_i64, "setfalse");
          break;
        case 1:
          sub_2094330(a1->m128i_i64, "setoeq");
          break;
        case 2:
          sub_2094330(a1->m128i_i64, "setogt");
          break;
        case 3:
          sub_2094330(a1->m128i_i64, "setoge");
          break;
        case 4:
          sub_2094330(a1->m128i_i64, "setolt");
          break;
        case 5:
          sub_2094330(a1->m128i_i64, "setole");
          break;
        case 6:
          sub_2094330(a1->m128i_i64, "setone");
          break;
        case 7:
          sub_2094330(a1->m128i_i64, "seto");
          break;
        case 8:
          sub_2094330(a1->m128i_i64, "setuo");
          break;
        case 9:
          sub_2094330(a1->m128i_i64, "setueq");
          break;
        case 0xA:
          sub_2094330(a1->m128i_i64, "setugt");
          break;
        case 0xB:
          sub_2094330(a1->m128i_i64, "setuge");
          break;
        case 0xC:
          sub_2094330(a1->m128i_i64, "setult");
          break;
        case 0xD:
          sub_2094330(a1->m128i_i64, "setule");
          break;
        case 0xE:
          sub_2094330(a1->m128i_i64, "setune");
          break;
        case 0xF:
          sub_2094330(a1->m128i_i64, "settrue");
          break;
        case 0x10:
          sub_2094330(a1->m128i_i64, "setfalse2");
          break;
        case 0x11:
          sub_2094330(a1->m128i_i64, "seteq");
          break;
        case 0x12:
          sub_2094330(a1->m128i_i64, "setgt");
          break;
        case 0x13:
          sub_2094330(a1->m128i_i64, "setge");
          break;
        case 0x14:
          sub_2094330(a1->m128i_i64, "setlt");
          break;
        case 0x15:
          sub_2094330(a1->m128i_i64, "setle");
          break;
        case 0x16:
          sub_2094330(a1->m128i_i64, "setne");
          break;
        case 0x17:
          sub_2094330(a1->m128i_i64, "settrue2");
          break;
      }
      return a1;
    case 8:
      sub_2094330(a1->m128i_i64, "Register");
      return a1;
    case 9:
      sub_2094330(a1->m128i_i64, "RegisterMask");
      return a1;
    case 10:
      if ( (*(_BYTE *)(a2 + 26) & 8) != 0 )
        sub_2094330(a1->m128i_i64, "OpaqueConstant");
      else
        sub_2094330(a1->m128i_i64, "Constant");
      return a1;
    case 11:
      sub_2094330(a1->m128i_i64, "ConstantFP");
      return a1;
    case 12:
      sub_2094330(a1->m128i_i64, "GlobalAddress");
      return a1;
    case 13:
      sub_2094330(a1->m128i_i64, "GlobalTLSAddress");
      return a1;
    case 14:
      sub_2094330(a1->m128i_i64, "FrameIndex");
      return a1;
    case 15:
      sub_2094330(a1->m128i_i64, "JumpTable");
      return a1;
    case 16:
      sub_2094330(a1->m128i_i64, "ConstantPool");
      return a1;
    case 17:
      sub_2094330(a1->m128i_i64, "ExternalSymbol");
      return a1;
    case 18:
      sub_2094330(a1->m128i_i64, "BlockAddress");
      return a1;
    case 19:
      sub_2094330(a1->m128i_i64, "GLOBAL_OFFSET_TABLE");
      return a1;
    case 20:
      sub_2094330(a1->m128i_i64, "FRAMEADDR");
      return a1;
    case 21:
      sub_2094330(a1->m128i_i64, "RETURNADDR");
      return a1;
    case 22:
      sub_2094330(a1->m128i_i64, "ADDROFRETURNADDR");
      return a1;
    case 23:
      sub_2094330(a1->m128i_i64, "LOCAL_RECOVER");
      return a1;
    case 24:
      sub_2094330(a1->m128i_i64, "READ_REGISTER");
      return a1;
    case 25:
      sub_2094330(a1->m128i_i64, "WRITE_REGISTER");
      return a1;
    case 26:
      sub_2094330(a1->m128i_i64, "FRAME_TO_ARGS_OFFSET");
      return a1;
    case 27:
      sub_2094330(a1->m128i_i64, "EH_DWARF_CFA");
      return a1;
    case 28:
      sub_2094330(a1->m128i_i64, "EH_RETURN");
      return a1;
    case 29:
      sub_2094330(a1->m128i_i64, "EH_SJLJ_SETJMP");
      return a1;
    case 30:
      sub_2094330(a1->m128i_i64, "EH_SJLJ_LONGJMP");
      return a1;
    case 31:
      sub_2094330(a1->m128i_i64, "EH_SJLJ_SETUP_DISPATCH");
      return a1;
    case 32:
      if ( (*(_BYTE *)(a2 + 26) & 8) != 0 )
        sub_2094330(a1->m128i_i64, "OpaqueTargetConstant");
      else
        sub_2094330(a1->m128i_i64, "TargetConstant");
      return a1;
    case 33:
      sub_2094330(a1->m128i_i64, "TargetConstantFP");
      return a1;
    case 34:
      sub_2094330(a1->m128i_i64, "TargetGlobalAddress");
      return a1;
    case 35:
      sub_2094330(a1->m128i_i64, "TargetGlobalTLSAddress");
      return a1;
    case 36:
      sub_2094330(a1->m128i_i64, "TargetFrameIndex");
      return a1;
    case 37:
      sub_2094330(a1->m128i_i64, "TargetJumpTable");
      return a1;
    case 38:
      sub_2094330(a1->m128i_i64, "TargetConstantPool");
      return a1;
    case 39:
      sub_2094330(a1->m128i_i64, "TargetExternalSymbol");
      return a1;
    case 40:
      sub_2094330(a1->m128i_i64, "TargetBlockAddress");
      return a1;
    case 41:
      sub_2094330(a1->m128i_i64, "MCSymbol");
      return a1;
    case 42:
      sub_2094330(a1->m128i_i64, "TargetIndex");
      return a1;
    case 43:
    case 44:
    case 45:
      v6 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL * (v4 != 43)) + 88LL);
      v7 = *(_QWORD **)(v6 + 24);
      if ( *(_DWORD *)(v6 + 32) > 0x40u )
        v7 = (_QWORD *)*v7;
      if ( (unsigned int)v7 <= 0x1DC9 )
      {
        sub_15E1070(a1->m128i_i64, (int)v7, 0, 0);
      }
      else
      {
        v8 = (*(__int64 (__fastcall **)(_QWORD))(*(_QWORD *)*a3 + 32LL))(*a3);
        (*(void (__fastcall **)(__m128i *, __int64, _QWORD, _QWORD, _QWORD))(*(_QWORD *)v8 + 16LL))(
          a1,
          v8,
          (unsigned int)v7,
          0,
          0);
      }
      return a1;
    case 46:
      sub_2094330(a1->m128i_i64, "CopyToReg");
      return a1;
    case 47:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "CopyFromReg");
      a1->m128i_i64[1] = 11;
      return a1;
    case 48:
      sub_2094330(a1->m128i_i64, "undef");
      return a1;
    case 49:
      sub_2094330(a1->m128i_i64, "extract_element");
      return a1;
    case 50:
      sub_2094330(a1->m128i_i64, "build_pair");
      return a1;
    case 51:
      sub_2094330(a1->m128i_i64, "merge_values");
      return a1;
    case 52:
      sub_2094330(a1->m128i_i64, "add");
      return a1;
    case 53:
      a1[1].m128i_i8[2] = 98;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      a1[1].m128i_i16[0] = 30067;
      a1->m128i_i64[1] = 3;
      a1[1].m128i_i8[3] = 0;
      return a1;
    case 54:
      a1[1].m128i_i8[2] = 108;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      a1[1].m128i_i16[0] = 30061;
      a1->m128i_i64[1] = 3;
      a1[1].m128i_i8[3] = 0;
      return a1;
    case 55:
      sub_2094330(a1->m128i_i64, "sdiv");
      return a1;
    case 56:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "udiv");
      a1->m128i_i64[1] = 4;
      return a1;
    case 57:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "srem");
      a1->m128i_i64[1] = 4;
      return a1;
    case 58:
      sub_2094330(a1->m128i_i64, "urem");
      return a1;
    case 59:
      sub_2094330(a1->m128i_i64, "smul_lohi");
      return a1;
    case 60:
      sub_2094330(a1->m128i_i64, "umul_lohi");
      return a1;
    case 61:
      a1[1].m128i_i8[6] = 109;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      a1[1].m128i_i32[0] = 1986618483;
      a1[1].m128i_i16[2] = 25970;
      a1->m128i_i64[1] = 7;
      a1[1].m128i_i8[7] = 0;
      return a1;
    case 62:
      sub_2094330(a1->m128i_i64, "udivrem");
      return a1;
    case 63:
      sub_2094330(a1->m128i_i64, "carry_false");
      return a1;
    case 64:
      sub_2094330(a1->m128i_i64, "addc");
      return a1;
    case 65:
      sub_2094330(a1->m128i_i64, "subc");
      return a1;
    case 66:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "adde");
      a1->m128i_i64[1] = 4;
      return a1;
    case 67:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "sube");
      a1->m128i_i64[1] = 4;
      return a1;
    case 68:
      sub_2094330(a1->m128i_i64, "addcarry");
      return a1;
    case 69:
      sub_2094330(a1->m128i_i64, "subcarry");
      return a1;
    case 70:
      a1[1].m128i_i8[4] = 111;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      a1[1].m128i_i32[0] = 1684300147;
      a1->m128i_i64[1] = 5;
      a1[1].m128i_i8[5] = 0;
      return a1;
    case 71:
      sub_2094330(a1->m128i_i64, "uaddo");
      return a1;
    case 72:
      a1[1].m128i_i8[4] = 111;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      a1[1].m128i_i32[0] = 1651864435;
      a1->m128i_i64[1] = 5;
      a1[1].m128i_i8[5] = 0;
      return a1;
    case 73:
      a1[1].m128i_i8[4] = 111;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      a1[1].m128i_i32[0] = 1651864437;
      a1->m128i_i64[1] = 5;
      a1[1].m128i_i8[5] = 0;
      return a1;
    case 74:
      a1[1].m128i_i8[4] = 111;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      a1[1].m128i_i32[0] = 1819635059;
      a1->m128i_i64[1] = 5;
      a1[1].m128i_i8[5] = 0;
      return a1;
    case 75:
      sub_2094330(a1->m128i_i64, "umulo");
      return a1;
    case 76:
      sub_2094330(a1->m128i_i64, "fadd");
      return a1;
    case 77:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "fsub");
      a1->m128i_i64[1] = 4;
      return a1;
    case 78:
      sub_2094330(a1->m128i_i64, "fmul");
      return a1;
    case 79:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "fdiv");
      a1->m128i_i64[1] = 4;
      return a1;
    case 80:
      sub_2094330(a1->m128i_i64, "frem");
      return a1;
    case 81:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "strict_fadd");
      a1->m128i_i64[1] = 11;
      return a1;
    case 82:
      sub_2094330(a1->m128i_i64, "strict_fsub");
      return a1;
    case 83:
      sub_2094330(a1->m128i_i64, "strict_fmul");
      return a1;
    case 84:
      sub_2094330(a1->m128i_i64, "strict_fdiv");
      return a1;
    case 85:
      sub_2094330(a1->m128i_i64, "strict_frem");
      return a1;
    case 86:
      sub_2094330(a1->m128i_i64, "strict_fma");
      return a1;
    case 87:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "strict_fsqrt");
      a1->m128i_i64[1] = 12;
      return a1;
    case 88:
      sub_2094330(a1->m128i_i64, "strict_fpow");
      return a1;
    case 89:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "strict_fpowi");
      a1->m128i_i64[1] = 12;
      return a1;
    case 90:
      sub_2094330(a1->m128i_i64, "strict_fsin");
      return a1;
    case 91:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "strict_fcos");
      a1->m128i_i64[1] = 11;
      return a1;
    case 92:
      sub_2094330(a1->m128i_i64, "strict_fexp");
      return a1;
    case 93:
      sub_2094330(a1->m128i_i64, "strict_fexp2");
      return a1;
    case 94:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "strict_flog");
      a1->m128i_i64[1] = 11;
      return a1;
    case 95:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "strict_flog10");
      a1->m128i_i64[1] = 13;
      return a1;
    case 96:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "strict_flog2");
      a1->m128i_i64[1] = 12;
      return a1;
    case 97:
      sub_2094330(a1->m128i_i64, "strict_frint");
      return a1;
    case 98:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      v51[0].m128i_i64[0] = 17;
      v36 = sub_22409D0(a1, v51, 0);
      v37 = v51[0].m128i_i64[0];
      si128 = _mm_load_si128((const __m128i *)&xmmword_4308930);
      a1->m128i_i64[0] = v36;
      a1[1].m128i_i64[0] = v37;
      *(_BYTE *)(v36 + 16) = 116;
      *(__m128i *)v36 = si128;
      v39 = v51[0].m128i_i64[0];
      v40 = a1->m128i_i64[0];
      a1->m128i_i64[1] = v51[0].m128i_i64[0];
      *(_BYTE *)(v40 + v39) = 0;
      return a1;
    case 99:
      a1[1].m128i_i8[2] = 97;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      a1[1].m128i_i16[0] = 28006;
      a1->m128i_i64[1] = 3;
      a1[1].m128i_i8[3] = 0;
      return a1;
    case 100:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "fmad");
      a1->m128i_i64[1] = 4;
      return a1;
    case 101:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "fcopysign");
      a1->m128i_i64[1] = 9;
      return a1;
    case 102:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "fgetsign");
      a1->m128i_i64[1] = 8;
      return a1;
    case 103:
      sub_2094330(a1->m128i_i64, "fcanonicalize");
      return a1;
    case 104:
      sub_2094330(a1->m128i_i64, "BUILD_VECTOR");
      return a1;
    case 105:
      sub_2094330(a1->m128i_i64, "insert_vector_elt");
      return a1;
    case 106:
      sub_2094330(a1->m128i_i64, "extract_vector_elt");
      return a1;
    case 107:
      strcpy(a1[1].m128i_i8, "concat_vectors");
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      a1->m128i_i64[1] = 14;
      return a1;
    case 108:
      sub_2094330(a1->m128i_i64, "insert_subvector");
      return a1;
    case 109:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      v51[0].m128i_i64[0] = 17;
      v31 = sub_22409D0(a1, v51, 0);
      v32 = v51[0].m128i_i64[0];
      v33 = _mm_load_si128((const __m128i *)&xmmword_4308940);
      a1->m128i_i64[0] = v31;
      a1[1].m128i_i64[0] = v32;
      *(_BYTE *)(v31 + 16) = 114;
      *(__m128i *)v31 = v33;
      v34 = v51[0].m128i_i64[0];
      v35 = a1->m128i_i64[0];
      a1->m128i_i64[1] = v51[0].m128i_i64[0];
      *(_BYTE *)(v35 + v34) = 0;
      return a1;
    case 110:
      sub_2094330(a1->m128i_i64, "vector_shuffle");
      return a1;
    case 111:
      sub_2094330(a1->m128i_i64, "scalar_to_vector");
      return a1;
    case 112:
      sub_2094330(a1->m128i_i64, "mulhu");
      return a1;
    case 113:
      a1[1].m128i_i8[4] = 115;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      a1[1].m128i_i32[0] = 1751938413;
      a1->m128i_i64[1] = 5;
      a1[1].m128i_i8[5] = 0;
      return a1;
    case 114:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "smin");
      a1->m128i_i64[1] = 4;
      return a1;
    case 115:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "smax");
      a1->m128i_i64[1] = 4;
      return a1;
    case 116:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "umin");
      a1->m128i_i64[1] = 4;
      return a1;
    case 117:
      sub_2094330(a1->m128i_i64, "umax");
      return a1;
    case 118:
      sub_2094330(a1->m128i_i64, "and");
      return a1;
    case 119:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "or");
      a1->m128i_i64[1] = 2;
      return a1;
    case 120:
      sub_2094330(a1->m128i_i64, "xor");
      return a1;
    case 121:
      sub_2094330(a1->m128i_i64, "abs");
      return a1;
    case 122:
      a1[1].m128i_i8[2] = 108;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      a1[1].m128i_i16[0] = 26739;
      a1->m128i_i64[1] = 3;
      a1[1].m128i_i8[3] = 0;
      return a1;
    case 123:
      a1[1].m128i_i8[2] = 97;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      a1[1].m128i_i16[0] = 29299;
      a1->m128i_i64[1] = 3;
      a1[1].m128i_i8[3] = 0;
      return a1;
    case 124:
      sub_2094330(a1->m128i_i64, "srl");
      return a1;
    case 125:
      sub_2094330(a1->m128i_i64, "rotl");
      return a1;
    case 126:
      sub_2094330(a1->m128i_i64, "rotr");
      return a1;
    case 127:
      sub_2094330(a1->m128i_i64, "bswap");
      return a1;
    case 128:
      sub_2094330(a1->m128i_i64, "cttz");
      return a1;
    case 129:
      sub_2094330(a1->m128i_i64, "ctlz");
      return a1;
    case 130:
      sub_2094330(a1->m128i_i64, "ctpop");
      return a1;
    case 131:
      sub_2094330(a1->m128i_i64, "bitreverse");
      return a1;
    case 132:
      sub_2094330(a1->m128i_i64, "cttz_zero_undef");
      return a1;
    case 133:
      sub_2094330(a1->m128i_i64, "ctlz_zero_undef");
      return a1;
    case 134:
      sub_2094330(a1->m128i_i64, "select");
      return a1;
    case 135:
      a1[1].m128i_i32[0] = 1818588022;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      a1[1].m128i_i16[2] = 25445;
      a1[1].m128i_i8[6] = 116;
      a1->m128i_i64[1] = 7;
      a1[1].m128i_i8[7] = 0;
      return a1;
    case 136:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "select_cc");
      a1->m128i_i64[1] = 9;
      return a1;
    case 137:
      sub_2094330(a1->m128i_i64, "setcc");
      return a1;
    case 138:
      a1->m128i_i64[1] = 10;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "setcccarry");
      return a1;
    case 139:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "shl_parts");
      a1->m128i_i64[1] = 9;
      return a1;
    case 140:
      sub_2094330(a1->m128i_i64, "sra_parts");
      return a1;
    case 141:
      sub_2094330(a1->m128i_i64, "srl_parts");
      return a1;
    case 142:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "sign_extend");
      a1->m128i_i64[1] = 11;
      return a1;
    case 143:
      sub_2094330(a1->m128i_i64, "zero_extend");
      return a1;
    case 144:
      a1->m128i_i64[1] = 10;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "any_extend");
      return a1;
    case 145:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "truncate");
      a1->m128i_i64[1] = 8;
      return a1;
    case 146:
      sub_2094330(a1->m128i_i64, "sint_to_fp");
      return a1;
    case 147:
      a1->m128i_i64[1] = 10;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "uint_to_fp");
      return a1;
    case 148:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      v51[0].m128i_i64[0] = 17;
      v26 = sub_22409D0(a1, v51, 0);
      v27 = v51[0].m128i_i64[0];
      v28 = _mm_load_si128((const __m128i *)&xmmword_4308950);
      a1->m128i_i64[0] = v26;
      a1[1].m128i_i64[0] = v27;
      *(_BYTE *)(v26 + 16) = 103;
      *(__m128i *)v26 = v28;
      v29 = v51[0].m128i_i64[0];
      v30 = a1->m128i_i64[0];
      a1->m128i_i64[1] = v51[0].m128i_i64[0];
      *(_BYTE *)(v30 + v29) = 0;
      return a1;
    case 149:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      v51[0].m128i_i64[0] = 23;
      v21 = sub_22409D0(a1, v51, 0);
      v22 = v51[0].m128i_i64[0];
      v23 = _mm_load_si128((const __m128i *)&xmmword_4308960);
      a1->m128i_i64[0] = v21;
      a1[1].m128i_i64[0] = v22;
      *(_DWORD *)(v21 + 16) = 1852399474;
      *(_WORD *)(v21 + 20) = 25970;
      *(_BYTE *)(v21 + 22) = 103;
      *(__m128i *)v21 = v23;
      v24 = v51[0].m128i_i64[0];
      v25 = a1->m128i_i64[0];
      a1->m128i_i64[1] = v51[0].m128i_i64[0];
      *(_BYTE *)(v25 + v24) = 0;
      return a1;
    case 150:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      v51[0].m128i_i64[0] = 24;
      v16 = sub_22409D0(a1, v51, 0);
      v17 = v51[0].m128i_i64[0];
      v18 = _mm_load_si128((const __m128i *)&xmmword_4308970);
      a1->m128i_i64[0] = v16;
      a1[1].m128i_i64[0] = v17;
      *(_QWORD *)(v16 + 16) = 0x6765726E695F726FLL;
      *(__m128i *)v16 = v18;
      v19 = v51[0].m128i_i64[0];
      v20 = a1->m128i_i64[0];
      a1->m128i_i64[1] = v51[0].m128i_i64[0];
      *(_BYTE *)(v20 + v19) = 0;
      return a1;
    case 151:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      v51[0].m128i_i64[0] = 24;
      v11 = sub_22409D0(a1, v51, 0);
      v12 = v51[0].m128i_i64[0];
      v13 = _mm_load_si128((const __m128i *)&xmmword_4308980);
      a1->m128i_i64[0] = v11;
      a1[1].m128i_i64[0] = v12;
      *(_QWORD *)(v11 + 16) = 0x6765726E695F726FLL;
      *(__m128i *)v11 = v13;
      v14 = v51[0].m128i_i64[0];
      v15 = a1->m128i_i64[0];
      a1->m128i_i64[1] = v51[0].m128i_i64[0];
      *(_BYTE *)(v15 + v14) = 0;
      return a1;
    case 152:
      a1->m128i_i64[1] = 10;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "fp_to_sint");
      return a1;
    case 153:
      a1->m128i_i64[1] = 10;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "fp_to_uint");
      return a1;
    case 154:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "fp_round");
      a1->m128i_i64[1] = 8;
      return a1;
    case 155:
      sub_2094330(a1->m128i_i64, "flt_rounds");
      return a1;
    case 156:
      sub_2094330(a1->m128i_i64, "fp_round_inreg");
      return a1;
    case 157:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "fp_extend");
      a1->m128i_i64[1] = 9;
      return a1;
    case 158:
      sub_2094330(a1->m128i_i64, "bitcast");
      return a1;
    case 159:
      sub_2094330(a1->m128i_i64, "addrspacecast");
      return a1;
    case 160:
      sub_2094330(a1->m128i_i64, "fp16_to_fp");
      return a1;
    case 161:
      sub_2094330(a1->m128i_i64, "fp_to_fp16");
      return a1;
    case 162:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "fneg");
      a1->m128i_i64[1] = 4;
      return a1;
    case 163:
      sub_2094330(a1->m128i_i64, "fabs");
      return a1;
    case 164:
      a1[1].m128i_i8[4] = 116;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      a1[1].m128i_i32[0] = 1920037734;
      a1->m128i_i64[1] = 5;
      a1[1].m128i_i8[5] = 0;
      return a1;
    case 165:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "fsin");
      a1->m128i_i64[1] = 4;
      return a1;
    case 166:
      sub_2094330(a1->m128i_i64, "fcos");
      return a1;
    case 167:
      sub_2094330(a1->m128i_i64, "fpowi");
      return a1;
    case 168:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "fpow");
      a1->m128i_i64[1] = 4;
      return a1;
    case 169:
      sub_2094330(a1->m128i_i64, "flog");
      return a1;
    case 170:
      sub_2094330(a1->m128i_i64, "flog2");
      return a1;
    case 171:
      sub_2094330(a1->m128i_i64, "flog10");
      return a1;
    case 172:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "fexp");
      a1->m128i_i64[1] = 4;
      return a1;
    case 173:
      sub_2094330(a1->m128i_i64, "fexp2");
      return a1;
    case 174:
      a1[1].m128i_i8[4] = 108;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      a1[1].m128i_i32[0] = 1768252262;
      a1->m128i_i64[1] = 5;
      a1[1].m128i_i8[5] = 0;
      return a1;
    case 175:
      strcpy(a1[1].m128i_i8, "ftrunc");
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      a1->m128i_i64[1] = 6;
      return a1;
    case 176:
      a1[1].m128i_i8[4] = 116;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      a1[1].m128i_i32[0] = 1852404326;
      a1->m128i_i64[1] = 5;
      a1[1].m128i_i8[5] = 0;
      return a1;
    case 177:
      sub_2094330(a1->m128i_i64, "fnearbyint");
      return a1;
    case 178:
      strcpy(a1[1].m128i_i8, "fround");
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      a1->m128i_i64[1] = 6;
      return a1;
    case 179:
      sub_2094330(a1->m128i_i64, "ffloor");
      return a1;
    case 180:
      a1[1].m128i_i32[0] = 1852403046;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      a1[1].m128i_i16[2] = 30062;
      a1[1].m128i_i8[6] = 109;
      a1->m128i_i64[1] = 7;
      a1[1].m128i_i8[7] = 0;
      return a1;
    case 181:
      a1[1].m128i_i32[0] = 2019650918;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      a1[1].m128i_i16[2] = 30062;
      a1[1].m128i_i8[6] = 109;
      a1->m128i_i64[1] = 7;
      a1[1].m128i_i8[7] = 0;
      return a1;
    case 182:
      a1[1].m128i_i32[0] = 1852403046;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      a1[1].m128i_i16[2] = 24942;
      a1[1].m128i_i8[6] = 110;
      a1->m128i_i64[1] = 7;
      a1[1].m128i_i8[7] = 0;
      return a1;
    case 183:
      a1[1].m128i_i32[0] = 2019650918;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      a1[1].m128i_i16[2] = 24942;
      a1[1].m128i_i8[6] = 110;
      a1->m128i_i64[1] = 7;
      a1[1].m128i_i8[7] = 0;
      return a1;
    case 184:
      sub_2094330(a1->m128i_i64, "fsincos");
      return a1;
    case 185:
      sub_2094330(a1->m128i_i64, "load");
      return a1;
    case 186:
      sub_2094330(a1->m128i_i64, "store");
      return a1;
    case 187:
      sub_2094330(a1->m128i_i64, "dynamic_stackalloc");
      return a1;
    case 188:
      sub_2094330(a1->m128i_i64, "br");
      return a1;
    case 189:
      sub_2094330(a1->m128i_i64, "brind");
      return a1;
    case 190:
      sub_2094330(a1->m128i_i64, "br_jt");
      return a1;
    case 191:
      sub_2094330(a1->m128i_i64, "brcond");
      return a1;
    case 192:
      sub_2094330(a1->m128i_i64, "br_cc");
      return a1;
    case 193:
      sub_2094330(a1->m128i_i64, "inlineasm");
      return a1;
    case 194:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "eh_label");
      a1->m128i_i64[1] = 8;
      return a1;
    case 197:
      sub_2094330(a1->m128i_i64, "catchret");
      return a1;
    case 198:
      sub_2094330(a1->m128i_i64, "cleanupret");
      return a1;
    case 199:
      sub_2094330(a1->m128i_i64, "stacksave");
      return a1;
    case 200:
      sub_2094330(a1->m128i_i64, "stackrestore");
      return a1;
    case 201:
      sub_2094330(a1->m128i_i64, "callseq_start");
      return a1;
    case 202:
      sub_2094330(a1->m128i_i64, "fake_callseq_start");
      return a1;
    case 203:
      sub_2094330(a1->m128i_i64, "callseq_end");
      return a1;
    case 204:
      sub_2094330(a1->m128i_i64, "vaarg");
      return a1;
    case 205:
      sub_2094330(a1->m128i_i64, "vacopy");
      return a1;
    case 206:
      sub_2094330(a1->m128i_i64, "vaend");
      return a1;
    case 207:
      sub_2094330(a1->m128i_i64, "vastart");
      return a1;
    case 208:
      sub_2094330(a1->m128i_i64, "SrcValue");
      return a1;
    case 209:
      sub_2094330(a1->m128i_i64, "MDNode");
      return a1;
    case 210:
      sub_2094330(a1->m128i_i64, "PCMarker");
      return a1;
    case 211:
      sub_2094330(a1->m128i_i64, "ReadCycleCounter");
      return a1;
    case 212:
      sub_2094330(a1->m128i_i64, "handlenode");
      return a1;
    case 213:
      sub_2094330(a1->m128i_i64, "init_trampoline");
      return a1;
    case 214:
      sub_2094330(a1->m128i_i64, "adjust_trampoline");
      return a1;
    case 215:
      sub_2094330(a1->m128i_i64, "trap");
      return a1;
    case 216:
      sub_2094330(a1->m128i_i64, "debugtrap");
      return a1;
    case 217:
      sub_2094330(a1->m128i_i64, "Prefetch");
      return a1;
    case 218:
      sub_2094330(a1->m128i_i64, "AtomicFence");
      return a1;
    case 219:
      sub_2094330(a1->m128i_i64, "AtomicLoad");
      return a1;
    case 220:
      sub_2094330(a1->m128i_i64, "AtomicStore");
      return a1;
    case 221:
      sub_2094330(a1->m128i_i64, "AtomicCmpSwap");
      return a1;
    case 222:
      sub_2094330(a1->m128i_i64, "AtomicCmpSwapWithSuccess");
      return a1;
    case 223:
      sub_2094330(a1->m128i_i64, "AtomicSwap");
      return a1;
    case 224:
      sub_2094330(a1->m128i_i64, "AtomicLoadAdd");
      return a1;
    case 225:
      sub_2094330(a1->m128i_i64, "AtomicLoadSub");
      return a1;
    case 226:
      sub_2094330(a1->m128i_i64, "AtomicLoadAnd");
      return a1;
    case 227:
      sub_2094330(a1->m128i_i64, "AtomicLoadClr");
      return a1;
    case 228:
      sub_2094330(a1->m128i_i64, "AtomicLoadOr");
      return a1;
    case 229:
      sub_2094330(a1->m128i_i64, "AtomicLoadXor");
      return a1;
    case 230:
      sub_2094330(a1->m128i_i64, "AtomicLoadNand");
      return a1;
    case 231:
      sub_2094330(a1->m128i_i64, "AtomicLoadMin");
      return a1;
    case 232:
      sub_2094330(a1->m128i_i64, "AtomicLoadMax");
      return a1;
    case 233:
      sub_2094330(a1->m128i_i64, "AtomicLoadUMin");
      return a1;
    case 234:
      sub_2094330(a1->m128i_i64, "AtomicLoadUMax");
      return a1;
    case 235:
      sub_2094330(a1->m128i_i64, "masked_load");
      return a1;
    case 236:
      sub_2094330(a1->m128i_i64, "masked_store");
      return a1;
    case 237:
      sub_2094330(a1->m128i_i64, "masked_gather");
      return a1;
    case 238:
      sub_2094330(a1->m128i_i64, "masked_scatter");
      return a1;
    case 239:
      sub_2094330(a1->m128i_i64, "lifetime.start");
      return a1;
    case 240:
      sub_2094330(a1->m128i_i64, "lifetime.end");
      return a1;
    case 241:
      sub_2094330(a1->m128i_i64, "gc_transition.start");
      return a1;
    case 242:
      sub_2094330(a1->m128i_i64, "gc_transition.end");
      return a1;
    case 243:
      sub_2094330(a1->m128i_i64, "get.dynamic.area.offset");
      return a1;
    case 244:
      sub_2094330(a1->m128i_i64, "vecreduce_strict_fadd");
      return a1;
    case 245:
      sub_2094330(a1->m128i_i64, "vecreduce_strict_fmul");
      return a1;
    case 246:
      sub_2094330(a1->m128i_i64, "vecreduce_fadd");
      return a1;
    case 247:
      sub_2094330(a1->m128i_i64, "vecreduce_fmul");
      return a1;
    case 248:
      sub_2094330(a1->m128i_i64, "vecreduce_add");
      return a1;
    case 249:
      sub_2094330(a1->m128i_i64, "vecreduce_mul");
      return a1;
    case 250:
      sub_2094330(a1->m128i_i64, "vecreduce_and");
      return a1;
    case 251:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "vecreduce_or");
      a1->m128i_i64[1] = 12;
      return a1;
    case 252:
      sub_2094330(a1->m128i_i64, "vecreduce_xor");
      return a1;
    case 253:
      sub_2094330(a1->m128i_i64, "vecreduce_smax");
      return a1;
    case 254:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "vecreduce_smin");
      a1->m128i_i64[1] = 14;
      return a1;
    case 255:
      sub_2094330(a1->m128i_i64, "vecreduce_umax");
      return a1;
    case 256:
      sub_2094330(a1->m128i_i64, "vecreduce_umin");
      return a1;
    case 257:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "vecreduce_fmax");
      a1->m128i_i64[1] = 14;
      return a1;
    case 258:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "vecreduce_fmin");
      a1->m128i_i64[1] = 14;
      return a1;
    default:
      if ( v4 <= 0x102 )
      {
        sub_2094330(a1->m128i_i64, "<<Unknown DAG Node>>");
        return a1;
      }
      if ( v5 < 0 )
      {
        if ( !a3
          || (v41 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a3[4] + 16LL) + 40LL))(*(_QWORD *)(a3[4] + 16LL))) == 0
          || (v42 = (unsigned int)~*(__int16 *)(a2 + 24), *(_DWORD *)(v41 + 32) <= (unsigned int)v42) )
        {
          sub_155CE20(v50, *(unsigned __int16 *)(a2 + 24), 0);
          sub_95D570(v51, "<<Unknown Machine Node #", (__int64)v50);
          sub_94F930(a1, (__int64)v51, ">>");
          sub_2240A30(v51);
          sub_2240A30(v50);
          return a1;
        }
        v43 = a1 + 1;
        v44 = (const char *)(*(_QWORD *)(v41 + 24) + *(unsigned int *)(*(_QWORD *)(v41 + 16) + 4 * v42));
        if ( !v44 )
        {
          a1->m128i_i64[0] = (__int64)v43;
          a1->m128i_i64[1] = 0;
          a1[1].m128i_i8[0] = 0;
          return a1;
        }
        v45 = strlen(v44);
        a1->m128i_i64[0] = (__int64)v43;
        v51[0].m128i_i64[0] = v45;
        v46 = v45;
        if ( v45 > 0xF )
        {
          v49 = sub_22409D0(a1, v51, 0);
          a1->m128i_i64[0] = v49;
          v43 = (__m128i *)v49;
          a1[1].m128i_i64[0] = v51[0].m128i_i64[0];
        }
        else
        {
          if ( v45 == 1 )
          {
            a1[1].m128i_i8[0] = *v44;
LABEL_305:
            v47 = v51[0].m128i_i64[0];
            v48 = a1->m128i_i64[0];
            a1->m128i_i64[1] = v51[0].m128i_i64[0];
            *(_BYTE *)(v48 + v47) = 0;
            return a1;
          }
          if ( !v45 )
            goto LABEL_305;
        }
        memcpy(v43, v44, v46);
        goto LABEL_305;
      }
      if ( a3 )
      {
        v10 = (const char *)(*(__int64 (__fastcall **)(_QWORD, _QWORD))(*(_QWORD *)a3[2] + 1328LL))(
                              a3[2],
                              *(unsigned __int16 *)(a2 + 24));
        if ( v10 )
        {
          sub_2094330(a1->m128i_i64, v10);
        }
        else
        {
          sub_155CE20(v50, *(unsigned __int16 *)(a2 + 24), 0);
          sub_95D570(v51, "<<Unknown Target Node #", (__int64)v50);
          sub_94F930(a1, (__int64)v51, ">>");
          sub_2240A30(v51);
          sub_2240A30(v50);
        }
      }
      else
      {
        sub_155CE20(v50, (unsigned __int16)v5, 0);
        sub_95D570(v51, "<<Unknown Node #", (__int64)v50);
        sub_94F930(a1, (__int64)v51, ">>");
        sub_2240A30(v51);
        sub_2240A30(v50);
      }
      return a1;
  }
}
