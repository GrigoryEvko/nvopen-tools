// Function: sub_3418C90
// Address: 0x3418c90
//
__m128i *__fastcall sub_3418C90(__m128i *a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v5; // rsi
  __int64 v6; // rax
  _QWORD *v7; // rdi
  __int64 v9; // rax
  __int64 v10; // rdx
  __m128i v11; // xmm0
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rdx
  __m128i v16; // xmm0
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rdx
  __m128i v21; // xmm0
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rdx
  __m128i v26; // xmm0
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rax
  __int64 v30; // rdx
  __m128i v31; // xmm0
  __int64 v32; // rax
  __int64 v33; // rdx
  __int64 v34; // rax
  __int64 v35; // rdx
  __m128i v36; // xmm0
  __int64 v37; // rax
  __int64 v38; // rdx
  __int64 v39; // rax
  __int64 v40; // rdx
  __m128i v41; // xmm0
  __int64 v42; // rax
  __int64 v43; // rdx
  __int64 v44; // rax
  __int64 v45; // rdx
  __m128i v46; // xmm0
  __int64 v47; // rax
  __int64 v48; // rdx
  __int64 v49; // rax
  __int64 v50; // rdx
  __m128i v51; // xmm0
  __int64 v52; // rax
  __int64 v53; // rdx
  __int64 v54; // rax
  __int64 v55; // rdx
  __m128i v56; // xmm0
  __int64 v57; // rax
  __int64 v58; // rdx
  __int64 v59; // rax
  __int64 v60; // rdx
  __m128i v61; // xmm0
  __int64 v62; // rax
  __int64 v63; // rdx
  _OWORD *v64; // rax
  __int64 v65; // rdx
  __m128i si128; // xmm0
  __int64 v67; // rax
  __int64 v68; // rdx
  __int64 v69; // rax
  __int64 v70; // rdx
  __m128i v71; // xmm0
  __int64 v72; // rax
  __int64 v73; // rdx
  __int64 v74; // rax
  __int64 v75; // rdx
  __m128i v76; // xmm0
  __int64 v77; // rax
  __int64 v78; // rdx
  _OWORD *v79; // rax
  __int64 v80; // rdx
  __m128i v81; // xmm0
  __int64 v82; // rax
  __int64 v83; // rdx
  __int64 v84; // rax
  __int64 v85; // rdx
  __m128i v86; // xmm0
  __int64 v87; // rax
  __int64 v88; // rdx
  __int64 v89; // rax
  __int64 v90; // rdx
  __m128i v91; // xmm0
  __int64 v92; // rax
  __int64 v93; // rdx
  const char *v94; // rsi
  __int64 v95; // rdx
  char *v96; // rsi
  __int64 v97; // rax
  __int64 v98; // rdx
  __m128i *v99; // r15
  const char *v100; // r14
  size_t v101; // rax
  size_t v102; // r13
  __int64 v103; // rax
  __int64 v104; // rax
  __int64 v105; // rdx
  unsigned __int64 v106[4]; // [rsp+0h] [rbp-70h] BYREF
  __m128i v107[5]; // [rsp+20h] [rbp-50h] BYREF

  v5 = *(unsigned int *)(a2 + 24);
  switch ( (int)v5 )
  {
    case 0:
    case 335:
      if ( (unsigned int)v5 > 0x1F3 )
        goto LABEL_530;
      sub_34182A0(a1->m128i_i64, "<<Unknown DAG Node>>");
      return a1;
    case 1:
      sub_34182A0(a1->m128i_i64, "EntryToken");
      return a1;
    case 2:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "TokenFactor");
      a1->m128i_i64[1] = 11;
      return a1;
    case 3:
      sub_34182A0(a1->m128i_i64, "AssertSext");
      return a1;
    case 4:
      a1->m128i_i64[1] = 10;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "AssertZext");
      return a1;
    case 5:
      sub_34182A0(a1->m128i_i64, "AssertAlign");
      return a1;
    case 6:
      a1->m128i_i64[1] = 10;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "BasicBlock");
      return a1;
    case 7:
      sub_34182A0(a1->m128i_i64, "ValueType");
      return a1;
    case 8:
      switch ( *(_DWORD *)(a2 + 96) )
      {
        case 0:
          sub_34182A0(a1->m128i_i64, "setfalse");
          break;
        case 1:
          sub_34182A0(a1->m128i_i64, "setoeq");
          break;
        case 2:
          sub_34182A0(a1->m128i_i64, "setogt");
          break;
        case 3:
          sub_34182A0(a1->m128i_i64, "setoge");
          break;
        case 4:
          sub_34182A0(a1->m128i_i64, "setolt");
          break;
        case 5:
          sub_34182A0(a1->m128i_i64, "setole");
          break;
        case 6:
          sub_34182A0(a1->m128i_i64, "setone");
          break;
        case 7:
          sub_34182A0(a1->m128i_i64, "seto");
          break;
        case 8:
          sub_34182A0(a1->m128i_i64, "setuo");
          break;
        case 9:
          sub_34182A0(a1->m128i_i64, "setueq");
          break;
        case 0xA:
          sub_34182A0(a1->m128i_i64, "setugt");
          break;
        case 0xB:
          sub_34182A0(a1->m128i_i64, "setuge");
          break;
        case 0xC:
          sub_34182A0(a1->m128i_i64, "setult");
          break;
        case 0xD:
          sub_34182A0(a1->m128i_i64, "setule");
          break;
        case 0xE:
          sub_34182A0(a1->m128i_i64, "setune");
          break;
        case 0xF:
          sub_34182A0(a1->m128i_i64, "settrue");
          break;
        case 0x10:
          sub_34182A0(a1->m128i_i64, "setfalse2");
          break;
        case 0x11:
          sub_34182A0(a1->m128i_i64, "seteq");
          break;
        case 0x12:
          sub_34182A0(a1->m128i_i64, "setgt");
          break;
        case 0x13:
          sub_34182A0(a1->m128i_i64, "setge");
          break;
        case 0x14:
          sub_34182A0(a1->m128i_i64, "setlt");
          break;
        case 0x15:
          sub_34182A0(a1->m128i_i64, "setle");
          break;
        case 0x16:
          sub_34182A0(a1->m128i_i64, "setne");
          break;
        case 0x17:
          sub_34182A0(a1->m128i_i64, "settrue2");
          break;
        default:
          BUG();
      }
      return a1;
    case 9:
      a1->m128i_i64[1] = 8;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "Register");
      return a1;
    case 10:
      sub_34182A0(a1->m128i_i64, "RegisterMask");
      return a1;
    case 11:
      if ( (*(_BYTE *)(a2 + 32) & 8) != 0 )
        sub_34182A0(a1->m128i_i64, "OpaqueConstant");
      else
        sub_34182A0(a1->m128i_i64, "Constant");
      return a1;
    case 12:
      sub_34182A0(a1->m128i_i64, "ConstantFP");
      return a1;
    case 13:
      sub_34182A0(a1->m128i_i64, "GlobalAddress");
      return a1;
    case 14:
      v107[0].m128i_i64[0] = 16;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      v64 = (_OWORD *)sub_22409D0((__int64)a1, (unsigned __int64 *)v107, 0);
      v65 = v107[0].m128i_i64[0];
      si128 = _mm_load_si128((const __m128i *)&xmmword_44E0B30);
      a1->m128i_i64[0] = (__int64)v64;
      a1[1].m128i_i64[0] = v65;
      *v64 = si128;
      v67 = v107[0].m128i_i64[0];
      v68 = a1->m128i_i64[0];
      a1->m128i_i64[1] = v107[0].m128i_i64[0];
      *(_BYTE *)(v68 + v67) = 0;
      return a1;
    case 15:
      a1->m128i_i64[1] = 10;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "FrameIndex");
      return a1;
    case 16:
      sub_34182A0(a1->m128i_i64, "JumpTable");
      return a1;
    case 17:
      sub_34182A0(a1->m128i_i64, "ConstantPool");
      return a1;
    case 18:
      sub_34182A0(a1->m128i_i64, "ExternalSymbol");
      return a1;
    case 19:
      sub_34182A0(a1->m128i_i64, "BlockAddress");
      return a1;
    case 20:
      sub_34182A0(a1->m128i_i64, "PtrAuthGlobalAddress");
      return a1;
    case 21:
      sub_34182A0(a1->m128i_i64, "GLOBAL_OFFSET_TABLE");
      return a1;
    case 22:
      sub_34182A0(a1->m128i_i64, "FRAMEADDR");
      return a1;
    case 23:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "RETURNADDR");
      a1->m128i_i64[1] = 10;
      return a1;
    case 24:
      sub_34182A0(a1->m128i_i64, "ADDROFRETURNADDR");
      return a1;
    case 25:
      sub_34182A0(a1->m128i_i64, "SPONENTRY");
      return a1;
    case 26:
      sub_34182A0(a1->m128i_i64, "LOCAL_RECOVER");
      return a1;
    case 27:
      sub_34182A0(a1->m128i_i64, "READ_REGISTER");
      return a1;
    case 28:
      sub_34182A0(a1->m128i_i64, "WRITE_REGISTER");
      return a1;
    case 29:
      sub_34182A0(a1->m128i_i64, "FRAME_TO_ARGS_OFFSET");
      return a1;
    case 30:
      sub_34182A0(a1->m128i_i64, "EH_DWARF_CFA");
      return a1;
    case 31:
      sub_34182A0(a1->m128i_i64, "EH_RETURN");
      return a1;
    case 32:
      sub_34182A0(a1->m128i_i64, "EH_SJLJ_SETJMP");
      return a1;
    case 33:
      sub_34182A0(a1->m128i_i64, "EH_SJLJ_LONGJMP");
      return a1;
    case 34:
      sub_34182A0(a1->m128i_i64, "EH_SJLJ_SETUP_DISPATCH");
      return a1;
    case 35:
      if ( (*(_BYTE *)(a2 + 32) & 8) != 0 )
        sub_34182A0(a1->m128i_i64, "OpaqueTargetConstant");
      else
        sub_34182A0(a1->m128i_i64, "TargetConstant");
      return a1;
    case 36:
      sub_34182A0(a1->m128i_i64, "TargetConstantFP");
      return a1;
    case 37:
      sub_34182A0(a1->m128i_i64, "TargetGlobalAddress");
      return a1;
    case 38:
      sub_34182A0(a1->m128i_i64, "TargetGlobalTLSAddress");
      return a1;
    case 39:
      sub_34182A0(a1->m128i_i64, "TargetFrameIndex");
      return a1;
    case 40:
      sub_34182A0(a1->m128i_i64, "TargetJumpTable");
      return a1;
    case 41:
      sub_34182A0(a1->m128i_i64, "TargetConstantPool");
      return a1;
    case 42:
      sub_34182A0(a1->m128i_i64, "TargetExternalSymbol");
      return a1;
    case 43:
      sub_34182A0(a1->m128i_i64, "TargetBlockAddress");
      return a1;
    case 44:
      sub_34182A0(a1->m128i_i64, "MCSymbol");
      return a1;
    case 45:
      sub_34182A0(a1->m128i_i64, "TargetIndex");
      return a1;
    case 46:
    case 47:
    case 48:
      v6 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL * ((_DWORD)v5 != 46)) + 96LL);
      v7 = *(_QWORD **)(v6 + 24);
      if ( *(_DWORD *)(v6 + 32) > 0x40u )
        v7 = (_QWORD *)*v7;
      if ( (unsigned int)v7 <= 0x3EEF )
      {
        v96 = sub_B60B70((int)v7);
        a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
        if ( v96 )
        {
          sub_34181F0(a1->m128i_i64, v96, (__int64)&v96[v95]);
        }
        else
        {
          a1->m128i_i64[1] = 0;
          a1[1].m128i_i8[0] = 0;
        }
      }
      else
      {
        if ( a3 )
          BUG();
        sub_34182A0(a1->m128i_i64, "Unknown intrinsic");
      }
      return a1;
    case 49:
      sub_34182A0(a1->m128i_i64, "CopyToReg");
      return a1;
    case 50:
      sub_34182A0(a1->m128i_i64, "CopyFromReg");
      return a1;
    case 51:
      sub_34182A0(a1->m128i_i64, "undef");
      return a1;
    case 52:
      sub_34182A0(a1->m128i_i64, "freeze");
      return a1;
    case 53:
      sub_34182A0(a1->m128i_i64, "extract_element");
      return a1;
    case 54:
      a1->m128i_i64[1] = 10;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "build_pair");
      return a1;
    case 55:
      sub_34182A0(a1->m128i_i64, "merge_values");
      return a1;
    case 56:
      sub_34182A0(a1->m128i_i64, "add");
      return a1;
    case 57:
      sub_34182A0(a1->m128i_i64, "sub");
      return a1;
    case 58:
      sub_34182A0(a1->m128i_i64, "mul");
      return a1;
    case 59:
      sub_34182A0(a1->m128i_i64, "sdiv");
      return a1;
    case 60:
      sub_34182A0(a1->m128i_i64, "udiv");
      return a1;
    case 61:
      sub_34182A0(a1->m128i_i64, "srem");
      return a1;
    case 62:
      sub_34182A0(a1->m128i_i64, "urem");
      return a1;
    case 63:
      sub_34182A0(a1->m128i_i64, "smul_lohi");
      return a1;
    case 64:
      sub_34182A0(a1->m128i_i64, "umul_lohi");
      return a1;
    case 65:
      sub_34182A0(a1->m128i_i64, "sdivrem");
      return a1;
    case 66:
      sub_34182A0(a1->m128i_i64, "udivrem");
      return a1;
    case 67:
      sub_34182A0(a1->m128i_i64, "carry_false");
      return a1;
    case 68:
      sub_34182A0(a1->m128i_i64, "addc");
      return a1;
    case 69:
      strcpy(a1[1].m128i_i8, "subc");
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      a1->m128i_i64[1] = 4;
      return a1;
    case 70:
      sub_34182A0(a1->m128i_i64, "adde");
      return a1;
    case 71:
      sub_34182A0(a1->m128i_i64, "sube");
      return a1;
    case 72:
      sub_34182A0(a1->m128i_i64, "uaddo_carry");
      return a1;
    case 73:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "usubo_carry");
      a1->m128i_i64[1] = 11;
      return a1;
    case 74:
      sub_34182A0(a1->m128i_i64, "saddo_carry");
      return a1;
    case 75:
      sub_34182A0(a1->m128i_i64, "ssubo_carry");
      return a1;
    case 76:
      sub_34182A0(a1->m128i_i64, "saddo");
      return a1;
    case 77:
      sub_34182A0(a1->m128i_i64, "uaddo");
      return a1;
    case 78:
      sub_34182A0(a1->m128i_i64, "ssubo");
      return a1;
    case 79:
      sub_34182A0(a1->m128i_i64, "usubo");
      return a1;
    case 80:
      a1[1].m128i_i32[0] = 1819635059;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      a1[1].m128i_i8[4] = 111;
      a1->m128i_i64[1] = 5;
      a1[1].m128i_i8[5] = 0;
      return a1;
    case 81:
      sub_34182A0(a1->m128i_i64, "umulo");
      return a1;
    case 82:
      sub_34182A0(a1->m128i_i64, "saddsat");
      return a1;
    case 83:
      a1[1].m128i_i8[6] = 116;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      a1[1].m128i_i32[0] = 1684300149;
      a1[1].m128i_i16[2] = 24947;
      a1->m128i_i64[1] = 7;
      a1[1].m128i_i8[7] = 0;
      return a1;
    case 84:
      sub_34182A0(a1->m128i_i64, "ssubsat");
      return a1;
    case 85:
      a1[1].m128i_i8[6] = 116;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      a1[1].m128i_i32[0] = 1651864437;
      a1[1].m128i_i16[2] = 24947;
      a1->m128i_i64[1] = 7;
      a1[1].m128i_i8[7] = 0;
      return a1;
    case 86:
      sub_34182A0(a1->m128i_i64, "sshlsat");
      return a1;
    case 87:
      a1[1].m128i_i8[6] = 116;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      a1[1].m128i_i32[0] = 1818784629;
      a1[1].m128i_i16[2] = 24947;
      a1->m128i_i64[1] = 7;
      a1[1].m128i_i8[7] = 0;
      return a1;
    case 88:
      sub_34182A0(a1->m128i_i64, "smulfix");
      return a1;
    case 89:
      sub_34182A0(a1->m128i_i64, "umulfix");
      return a1;
    case 90:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "smulfixsat");
      a1->m128i_i64[1] = 10;
      return a1;
    case 91:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "umulfixsat");
      a1->m128i_i64[1] = 10;
      return a1;
    case 92:
      sub_34182A0(a1->m128i_i64, "sdivfix");
      return a1;
    case 93:
      sub_34182A0(a1->m128i_i64, "udivfix");
      return a1;
    case 94:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "sdivfixsat");
      a1->m128i_i64[1] = 10;
      return a1;
    case 95:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "udivfixsat");
      a1->m128i_i64[1] = 10;
      return a1;
    case 96:
      sub_34182A0(a1->m128i_i64, "fadd");
      return a1;
    case 97:
      sub_34182A0(a1->m128i_i64, "fsub");
      return a1;
    case 98:
      sub_34182A0(a1->m128i_i64, "fmul");
      return a1;
    case 99:
      sub_34182A0(a1->m128i_i64, "fdiv");
      return a1;
    case 100:
      sub_34182A0(a1->m128i_i64, "frem");
      return a1;
    case 101:
      sub_34182A0(a1->m128i_i64, "strict_fadd");
      return a1;
    case 102:
      sub_34182A0(a1->m128i_i64, "strict_fsub");
      return a1;
    case 103:
      sub_34182A0(a1->m128i_i64, "strict_fmul");
      return a1;
    case 104:
      sub_34182A0(a1->m128i_i64, "strict_fdiv");
      return a1;
    case 105:
      sub_34182A0(a1->m128i_i64, "strict_frem");
      return a1;
    case 106:
      sub_34182A0(a1->m128i_i64, "strict_fma");
      return a1;
    case 107:
      sub_34182A0(a1->m128i_i64, "strict_fsqrt");
      return a1;
    case 108:
      sub_34182A0(a1->m128i_i64, "strict_fpow");
      return a1;
    case 109:
      sub_34182A0(a1->m128i_i64, "strict_fpowi");
      return a1;
    case 110:
      sub_34182A0(a1->m128i_i64, "strict_fldexp");
      return a1;
    case 111:
      sub_34182A0(a1->m128i_i64, "strict_fsin");
      return a1;
    case 112:
      sub_34182A0(a1->m128i_i64, "strict_fcos");
      return a1;
    case 113:
      sub_34182A0(a1->m128i_i64, "strict_ftan");
      return a1;
    case 114:
      sub_34182A0(a1->m128i_i64, "strict_fasin");
      return a1;
    case 115:
      sub_34182A0(a1->m128i_i64, "strict_facos");
      return a1;
    case 116:
      sub_34182A0(a1->m128i_i64, "strict_fatan");
      return a1;
    case 117:
      sub_34182A0(a1->m128i_i64, "strict_fatan2");
      return a1;
    case 118:
      sub_34182A0(a1->m128i_i64, "strict_fsinh");
      return a1;
    case 119:
      sub_34182A0(a1->m128i_i64, "strict_fcosh");
      return a1;
    case 120:
      sub_34182A0(a1->m128i_i64, "strict_ftanh");
      return a1;
    case 121:
      sub_34182A0(a1->m128i_i64, "strict_fexp");
      return a1;
    case 122:
      sub_34182A0(a1->m128i_i64, "strict_fexp2");
      return a1;
    case 123:
      sub_34182A0(a1->m128i_i64, "strict_flog");
      return a1;
    case 124:
      sub_34182A0(a1->m128i_i64, "strict_flog10");
      return a1;
    case 125:
      sub_34182A0(a1->m128i_i64, "strict_flog2");
      return a1;
    case 126:
      sub_34182A0(a1->m128i_i64, "strict_frint");
      return a1;
    case 127:
      sub_34182A0(a1->m128i_i64, "strict_fnearbyint");
      return a1;
    case 128:
      sub_34182A0(a1->m128i_i64, "strict_fmaxnum");
      return a1;
    case 129:
      sub_34182A0(a1->m128i_i64, "strict_fminnum");
      return a1;
    case 130:
      sub_34182A0(a1->m128i_i64, "strict_fceil");
      return a1;
    case 131:
      sub_34182A0(a1->m128i_i64, "strict_ffloor");
      return a1;
    case 132:
      sub_34182A0(a1->m128i_i64, "strict_fround");
      return a1;
    case 133:
      sub_34182A0(a1->m128i_i64, "strict_froundeven");
      return a1;
    case 134:
      sub_34182A0(a1->m128i_i64, "strict_ftrunc");
      return a1;
    case 135:
      sub_34182A0(a1->m128i_i64, "strict_lround");
      return a1;
    case 136:
      sub_34182A0(a1->m128i_i64, "strict_llround");
      return a1;
    case 137:
      sub_34182A0(a1->m128i_i64, "strict_lrint");
      return a1;
    case 138:
      sub_34182A0(a1->m128i_i64, "strict_llrint");
      return a1;
    case 139:
      sub_34182A0(a1->m128i_i64, "strict_fmaximum");
      return a1;
    case 140:
      sub_34182A0(a1->m128i_i64, "strict_fminimum");
      return a1;
    case 141:
      sub_34182A0(a1->m128i_i64, "strict_fp_to_sint");
      return a1;
    case 142:
      sub_34182A0(a1->m128i_i64, "strict_fp_to_uint");
      return a1;
    case 143:
      sub_34182A0(a1->m128i_i64, "strict_sint_to_fp");
      return a1;
    case 144:
      sub_34182A0(a1->m128i_i64, "strict_uint_to_fp");
      return a1;
    case 145:
      sub_34182A0(a1->m128i_i64, "strict_fp_round");
      return a1;
    case 146:
      sub_34182A0(a1->m128i_i64, "strict_fp_extend");
      return a1;
    case 147:
      sub_34182A0(a1->m128i_i64, "strict_fsetcc");
      return a1;
    case 148:
      sub_34182A0(a1->m128i_i64, "strict_fsetccs");
      return a1;
    case 149:
      sub_34182A0(a1->m128i_i64, "fptrunc_round");
      return a1;
    case 150:
      sub_34182A0(a1->m128i_i64, "fma");
      return a1;
    case 151:
      sub_34182A0(a1->m128i_i64, "fmad");
      return a1;
    case 152:
      sub_34182A0(a1->m128i_i64, "fcopysign");
      return a1;
    case 153:
      sub_34182A0(a1->m128i_i64, "fgetsign");
      return a1;
    case 154:
      sub_34182A0(a1->m128i_i64, "fcanonicalize");
      return a1;
    case 155:
      sub_34182A0(a1->m128i_i64, "is_fpclass");
      return a1;
    case 156:
      sub_34182A0(a1->m128i_i64, "BUILD_VECTOR");
      return a1;
    case 157:
      sub_34182A0(a1->m128i_i64, "insert_vector_elt");
      return a1;
    case 158:
      sub_34182A0(a1->m128i_i64, "extract_vector_elt");
      return a1;
    case 159:
      sub_34182A0(a1->m128i_i64, "concat_vectors");
      return a1;
    case 160:
      sub_34182A0(a1->m128i_i64, "insert_subvector");
      return a1;
    case 161:
      sub_34182A0(a1->m128i_i64, "extract_subvector");
      return a1;
    case 162:
      sub_34182A0(a1->m128i_i64, "vector_deinterleave");
      return a1;
    case 163:
      sub_34182A0(a1->m128i_i64, "vector_interleave");
      return a1;
    case 164:
      sub_34182A0(a1->m128i_i64, "vector_reverse");
      return a1;
    case 165:
      sub_34182A0(a1->m128i_i64, "vector_shuffle");
      return a1;
    case 166:
      sub_34182A0(a1->m128i_i64, "vector_splice");
      return a1;
    case 167:
      sub_34182A0(a1->m128i_i64, "scalar_to_vector");
      return a1;
    case 168:
      sub_34182A0(a1->m128i_i64, "splat_vector");
      return a1;
    case 169:
      sub_34182A0(a1->m128i_i64, "splat_vector_parts");
      return a1;
    case 170:
      sub_34182A0(a1->m128i_i64, "step_vector");
      return a1;
    case 171:
      sub_34182A0(a1->m128i_i64, "vector_compress");
      return a1;
    case 172:
      sub_34182A0(a1->m128i_i64, "mulhu");
      return a1;
    case 173:
      sub_34182A0(a1->m128i_i64, "mulhs");
      return a1;
    case 174:
      sub_34182A0(a1->m128i_i64, "avgfloors");
      return a1;
    case 175:
      sub_34182A0(a1->m128i_i64, "avgflooru");
      return a1;
    case 176:
      sub_34182A0(a1->m128i_i64, "avgceils");
      return a1;
    case 177:
      sub_34182A0(a1->m128i_i64, "avgceilu");
      return a1;
    case 178:
      sub_34182A0(a1->m128i_i64, "abds");
      return a1;
    case 179:
      sub_34182A0(a1->m128i_i64, "abdu");
      return a1;
    case 180:
      sub_34182A0(a1->m128i_i64, "smin");
      return a1;
    case 181:
      sub_34182A0(a1->m128i_i64, "smax");
      return a1;
    case 182:
      sub_34182A0(a1->m128i_i64, "umin");
      return a1;
    case 183:
      sub_34182A0(a1->m128i_i64, "umax");
      return a1;
    case 184:
      sub_34182A0(a1->m128i_i64, "scmp");
      return a1;
    case 185:
      sub_34182A0(a1->m128i_i64, "ucmp");
      return a1;
    case 186:
      sub_34182A0(a1->m128i_i64, "and");
      return a1;
    case 187:
      sub_34182A0(a1->m128i_i64, "or");
      return a1;
    case 188:
      sub_34182A0(a1->m128i_i64, "xor");
      return a1;
    case 189:
      a1[1].m128i_i8[2] = 115;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      a1[1].m128i_i16[0] = 25185;
      a1->m128i_i64[1] = 3;
      a1[1].m128i_i8[3] = 0;
      return a1;
    case 190:
      sub_34182A0(a1->m128i_i64, "shl");
      return a1;
    case 191:
      sub_34182A0(a1->m128i_i64, "sra");
      return a1;
    case 192:
      sub_34182A0(a1->m128i_i64, "srl");
      return a1;
    case 193:
      sub_34182A0(a1->m128i_i64, "rotl");
      return a1;
    case 194:
      sub_34182A0(a1->m128i_i64, "rotr");
      return a1;
    case 195:
      sub_34182A0(a1->m128i_i64, "fshl");
      return a1;
    case 196:
      sub_34182A0(a1->m128i_i64, "fshr");
      return a1;
    case 197:
      a1[1].m128i_i32[0] = 1635218274;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      a1[1].m128i_i8[4] = 112;
      a1->m128i_i64[1] = 5;
      a1[1].m128i_i8[5] = 0;
      return a1;
    case 198:
      strcpy(a1[1].m128i_i8, "cttz");
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      a1->m128i_i64[1] = 4;
      return a1;
    case 199:
      sub_34182A0(a1->m128i_i64, "ctlz");
      return a1;
    case 200:
      sub_34182A0(a1->m128i_i64, "ctpop");
      return a1;
    case 201:
      sub_34182A0(a1->m128i_i64, "bitreverse");
      return a1;
    case 202:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "parity");
      a1->m128i_i64[1] = 6;
      return a1;
    case 203:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "cttz_zero_undef");
      a1->m128i_i64[1] = 15;
      return a1;
    case 204:
      sub_34182A0(a1->m128i_i64, "ctlz_zero_undef");
      return a1;
    case 205:
      sub_34182A0(a1->m128i_i64, "select");
      return a1;
    case 206:
      sub_34182A0(a1->m128i_i64, "vselect");
      return a1;
    case 207:
      sub_34182A0(a1->m128i_i64, "select_cc");
      return a1;
    case 208:
      sub_34182A0(a1->m128i_i64, "setcc");
      return a1;
    case 209:
      sub_34182A0(a1->m128i_i64, "setcccarry");
      return a1;
    case 210:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "shl_parts");
      a1->m128i_i64[1] = 9;
      return a1;
    case 211:
      sub_34182A0(a1->m128i_i64, "sra_parts");
      return a1;
    case 212:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "srl_parts");
      a1->m128i_i64[1] = 9;
      return a1;
    case 213:
      sub_34182A0(a1->m128i_i64, "sign_extend");
      return a1;
    case 214:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "zero_extend");
      a1->m128i_i64[1] = 11;
      return a1;
    case 215:
      sub_34182A0(a1->m128i_i64, "any_extend");
      return a1;
    case 216:
      a1->m128i_i64[1] = 8;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "truncate");
      return a1;
    case 217:
      sub_34182A0(a1->m128i_i64, "truncate_ssat_s");
      return a1;
    case 218:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "truncate_ssat_u");
      a1->m128i_i64[1] = 15;
      return a1;
    case 219:
      sub_34182A0(a1->m128i_i64, "truncate_usat_u");
      return a1;
    case 220:
      a1->m128i_i64[1] = 10;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "sint_to_fp");
      return a1;
    case 221:
      a1->m128i_i64[1] = 10;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "uint_to_fp");
      return a1;
    case 222:
      v107[0].m128i_i64[0] = 17;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      v74 = sub_22409D0((__int64)a1, (unsigned __int64 *)v107, 0);
      v75 = v107[0].m128i_i64[0];
      v76 = _mm_load_si128((const __m128i *)&xmmword_4308950);
      a1->m128i_i64[0] = v74;
      a1[1].m128i_i64[0] = v75;
      *(_BYTE *)(v74 + 16) = 103;
      *(__m128i *)v74 = v76;
      v77 = v107[0].m128i_i64[0];
      v78 = a1->m128i_i64[0];
      a1->m128i_i64[1] = v107[0].m128i_i64[0];
      *(_BYTE *)(v78 + v77) = 0;
      return a1;
    case 223:
      sub_34182A0(a1->m128i_i64, "any_extend_vector_inreg");
      return a1;
    case 224:
      v107[0].m128i_i64[0] = 24;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      v69 = sub_22409D0((__int64)a1, (unsigned __int64 *)v107, 0);
      v70 = v107[0].m128i_i64[0];
      v71 = _mm_load_si128((const __m128i *)&xmmword_4308970);
      a1->m128i_i64[0] = v69;
      a1[1].m128i_i64[0] = v70;
      *(_QWORD *)(v69 + 16) = 0x6765726E695F726FLL;
      *(__m128i *)v69 = v71;
      v72 = v107[0].m128i_i64[0];
      v73 = a1->m128i_i64[0];
      a1->m128i_i64[1] = v107[0].m128i_i64[0];
      *(_BYTE *)(v73 + v72) = 0;
      return a1;
    case 225:
      sub_34182A0(a1->m128i_i64, "zero_extend_vector_inreg");
      return a1;
    case 226:
      a1->m128i_i64[1] = 10;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "fp_to_sint");
      return a1;
    case 227:
      a1->m128i_i64[1] = 10;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "fp_to_uint");
      return a1;
    case 228:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "fp_to_sint_sat");
      a1->m128i_i64[1] = 14;
      return a1;
    case 229:
      sub_34182A0(a1->m128i_i64, "fp_to_uint_sat");
      return a1;
    case 230:
      a1->m128i_i64[1] = 8;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "fp_round");
      return a1;
    case 231:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "get_rounding");
      a1->m128i_i64[1] = 12;
      return a1;
    case 232:
      sub_34182A0(a1->m128i_i64, "set_rounding");
      return a1;
    case 233:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "fp_extend");
      a1->m128i_i64[1] = 9;
      return a1;
    case 234:
      a1[1].m128i_i8[6] = 116;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      a1[1].m128i_i32[0] = 1668573538;
      a1[1].m128i_i16[2] = 29537;
      a1->m128i_i64[1] = 7;
      a1[1].m128i_i8[7] = 0;
      return a1;
    case 235:
      sub_34182A0(a1->m128i_i64, "addrspacecast");
      return a1;
    case 236:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "fp16_to_fp");
      a1->m128i_i64[1] = 10;
      return a1;
    case 237:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "fp_to_fp16");
      a1->m128i_i64[1] = 10;
      return a1;
    case 238:
      sub_34182A0(a1->m128i_i64, "strict_fp16_to_fp");
      return a1;
    case 239:
      sub_34182A0(a1->m128i_i64, "strict_fp_to_fp16");
      return a1;
    case 240:
      a1->m128i_i64[1] = 10;
      strcpy(a1[1].m128i_i8, "bf16_to_fp");
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      return a1;
    case 241:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "fp_to_bf16");
      a1->m128i_i64[1] = 10;
      return a1;
    case 242:
      sub_34182A0(a1->m128i_i64, "strict_bf16_to_fp");
      return a1;
    case 243:
      sub_34182A0(a1->m128i_i64, "strict_fp_to_bf16");
      return a1;
    case 244:
      sub_34182A0(a1->m128i_i64, "fneg");
      return a1;
    case 245:
      sub_34182A0(a1->m128i_i64, "fabs");
      return a1;
    case 246:
      sub_34182A0(a1->m128i_i64, "fsqrt");
      return a1;
    case 247:
      sub_34182A0(a1->m128i_i64, "fcbrt");
      return a1;
    case 248:
      sub_34182A0(a1->m128i_i64, "fsin");
      return a1;
    case 249:
      sub_34182A0(a1->m128i_i64, "fcos");
      return a1;
    case 250:
      sub_34182A0(a1->m128i_i64, "ftan");
      return a1;
    case 251:
      sub_34182A0(a1->m128i_i64, "fasin");
      return a1;
    case 252:
      sub_34182A0(a1->m128i_i64, "facos");
      return a1;
    case 253:
      sub_34182A0(a1->m128i_i64, "fatan");
      return a1;
    case 254:
      sub_34182A0(a1->m128i_i64, "fsinh");
      return a1;
    case 255:
      sub_34182A0(a1->m128i_i64, "fcosh");
      return a1;
    case 256:
      sub_34182A0(a1->m128i_i64, "ftanh");
      return a1;
    case 257:
      sub_34182A0(a1->m128i_i64, "fpow");
      return a1;
    case 258:
      sub_34182A0(a1->m128i_i64, "fpowi");
      return a1;
    case 259:
      sub_34182A0(a1->m128i_i64, "fldexp");
      return a1;
    case 260:
      sub_34182A0(a1->m128i_i64, "fatan2");
      return a1;
    case 261:
      sub_34182A0(a1->m128i_i64, "ffrexp");
      return a1;
    case 262:
      sub_34182A0(a1->m128i_i64, "flog");
      return a1;
    case 263:
      sub_34182A0(a1->m128i_i64, "flog2");
      return a1;
    case 264:
      sub_34182A0(a1->m128i_i64, "flog10");
      return a1;
    case 265:
      sub_34182A0(a1->m128i_i64, "fexp");
      return a1;
    case 266:
      sub_34182A0(a1->m128i_i64, "fexp2");
      return a1;
    case 267:
      sub_34182A0(a1->m128i_i64, "fexp10");
      return a1;
    case 268:
      sub_34182A0(a1->m128i_i64, "fceil");
      return a1;
    case 269:
      sub_34182A0(a1->m128i_i64, "ftrunc");
      return a1;
    case 270:
      sub_34182A0(a1->m128i_i64, "frint");
      return a1;
    case 271:
      sub_34182A0(a1->m128i_i64, "fnearbyint");
      return a1;
    case 272:
      sub_34182A0(a1->m128i_i64, "fround");
      return a1;
    case 273:
      sub_34182A0(a1->m128i_i64, "froundeven");
      return a1;
    case 274:
      sub_34182A0(a1->m128i_i64, "ffloor");
      return a1;
    case 275:
      sub_34182A0(a1->m128i_i64, "lround");
      return a1;
    case 276:
      a1[1].m128i_i8[6] = 100;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      a1[1].m128i_i32[0] = 1869769836;
      a1[1].m128i_i16[2] = 28277;
      a1->m128i_i64[1] = 7;
      a1[1].m128i_i8[7] = 0;
      return a1;
    case 277:
      a1[1].m128i_i32[0] = 1852404332;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      a1[1].m128i_i8[4] = 116;
      a1->m128i_i64[1] = 5;
      a1[1].m128i_i8[5] = 0;
      return a1;
    case 278:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "llrint");
      a1->m128i_i64[1] = 6;
      return a1;
    case 279:
      sub_34182A0(a1->m128i_i64, "fminnum");
      return a1;
    case 280:
      sub_34182A0(a1->m128i_i64, "fmaxnum");
      return a1;
    case 281:
      sub_34182A0(a1->m128i_i64, "fminnum_ieee");
      return a1;
    case 282:
      sub_34182A0(a1->m128i_i64, "fmaxnum_ieee");
      return a1;
    case 283:
      sub_34182A0(a1->m128i_i64, "fminimum");
      return a1;
    case 284:
      sub_34182A0(a1->m128i_i64, "fmaximum");
      return a1;
    case 285:
      sub_34182A0(a1->m128i_i64, "fminimumnum");
      return a1;
    case 286:
      sub_34182A0(a1->m128i_i64, "fmaximumnum");
      return a1;
    case 287:
      sub_34182A0(a1->m128i_i64, "fsincos");
      return a1;
    case 288:
      sub_34182A0(a1->m128i_i64, "fsincospi");
      return a1;
    case 289:
      sub_34182A0(a1->m128i_i64, "fmodf");
      return a1;
    case 290:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "get_fpenv");
      a1->m128i_i64[1] = 9;
      return a1;
    case 291:
      sub_34182A0(a1->m128i_i64, "set_fpenv");
      return a1;
    case 292:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "reset_fpenv");
      a1->m128i_i64[1] = 11;
      return a1;
    case 293:
      sub_34182A0(a1->m128i_i64, "get_fpenv_mem");
      return a1;
    case 294:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "set_fpenv_mem");
      a1->m128i_i64[1] = 13;
      return a1;
    case 295:
      sub_34182A0(a1->m128i_i64, "get_fpmode");
      return a1;
    case 296:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "set_fpmode");
      a1->m128i_i64[1] = 10;
      return a1;
    case 297:
      sub_34182A0(a1->m128i_i64, "reset_fpmode");
      return a1;
    case 298:
      sub_34182A0(a1->m128i_i64, "load");
      return a1;
    case 299:
      a1[1].m128i_i32[0] = 1919906931;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      a1[1].m128i_i8[4] = 101;
      a1->m128i_i64[1] = 5;
      a1[1].m128i_i8[5] = 0;
      return a1;
    case 300:
      v107[0].m128i_i64[0] = 18;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      v59 = sub_22409D0((__int64)a1, (unsigned __int64 *)v107, 0);
      v60 = v107[0].m128i_i64[0];
      v61 = _mm_load_si128((const __m128i *)&xmmword_44E0B50);
      a1->m128i_i64[0] = v59;
      a1[1].m128i_i64[0] = v60;
      *(_WORD *)(v59 + 16) = 25455;
      *(__m128i *)v59 = v61;
      v62 = v107[0].m128i_i64[0];
      v63 = a1->m128i_i64[0];
      a1->m128i_i64[1] = v107[0].m128i_i64[0];
      *(_BYTE *)(v63 + v62) = 0;
      return a1;
    case 301:
      a1->m128i_i64[1] = 2;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "br");
      return a1;
    case 302:
      sub_34182A0(a1->m128i_i64, "brind");
      return a1;
    case 303:
      a1[1].m128i_i32[0] = 1784640098;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      a1[1].m128i_i8[4] = 116;
      a1->m128i_i64[1] = 5;
      a1[1].m128i_i8[5] = 0;
      return a1;
    case 304:
      v107[0].m128i_i64[0] = 21;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      v54 = sub_22409D0((__int64)a1, (unsigned __int64 *)v107, 0);
      v55 = v107[0].m128i_i64[0];
      v56 = _mm_load_si128((const __m128i *)&xmmword_44E0B40);
      a1->m128i_i64[0] = v54;
      a1[1].m128i_i64[0] = v55;
      *(_DWORD *)(v54 + 16) = 1179535711;
      *(_BYTE *)(v54 + 20) = 79;
      *(__m128i *)v54 = v56;
      v57 = v107[0].m128i_i64[0];
      v58 = a1->m128i_i64[0];
      a1->m128i_i64[1] = v107[0].m128i_i64[0];
      *(_BYTE *)(v58 + v57) = 0;
      return a1;
    case 305:
      sub_34182A0(a1->m128i_i64, "brcond");
      return a1;
    case 306:
      a1[1].m128i_i32[0] = 1667199586;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      a1[1].m128i_i8[4] = 99;
      a1->m128i_i64[1] = 5;
      a1[1].m128i_i8[5] = 0;
      return a1;
    case 307:
      sub_34182A0(a1->m128i_i64, "inlineasm");
      return a1;
    case 308:
      sub_34182A0(a1->m128i_i64, "inlineasm_br");
      return a1;
    case 309:
      sub_34182A0(a1->m128i_i64, "eh_label");
      return a1;
    case 310:
      sub_34182A0(a1->m128i_i64, "annotation_label");
      return a1;
    case 311:
      sub_34182A0(a1->m128i_i64, "catchret");
      return a1;
    case 312:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "cleanupret");
      a1->m128i_i64[1] = 10;
      return a1;
    case 313:
      sub_34182A0(a1->m128i_i64, "stacksave");
      return a1;
    case 314:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "stackrestore");
      a1->m128i_i64[1] = 12;
      return a1;
    case 315:
      sub_34182A0(a1->m128i_i64, "callseq_start");
      return a1;
    case 316:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "callseq_end");
      a1->m128i_i64[1] = 11;
      return a1;
    case 317:
      a1[1].m128i_i32[0] = 1918984566;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      a1[1].m128i_i8[4] = 103;
      a1->m128i_i64[1] = 5;
      a1[1].m128i_i8[5] = 0;
      return a1;
    case 318:
      sub_34182A0(a1->m128i_i64, "vacopy");
      return a1;
    case 319:
      a1[1].m128i_i32[0] = 1852137846;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      a1[1].m128i_i8[4] = 100;
      a1->m128i_i64[1] = 5;
      a1[1].m128i_i8[5] = 0;
      return a1;
    case 320:
      sub_34182A0(a1->m128i_i64, "vastart");
      return a1;
    case 321:
      a1->m128i_i64[1] = 10;
      strcpy(a1[1].m128i_i8, "call_setup");
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      return a1;
    case 322:
      sub_34182A0(a1->m128i_i64, "call_alloc");
      return a1;
    case 323:
      sub_34182A0(a1->m128i_i64, "SrcValue");
      return a1;
    case 324:
      strcpy(a1[1].m128i_i8, "MDNode");
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      a1->m128i_i64[1] = 6;
      return a1;
    case 325:
      a1->m128i_i64[1] = 8;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "PCMarker");
      return a1;
    case 326:
      sub_34182A0(a1->m128i_i64, "ReadCycleCounter");
      return a1;
    case 327:
      v107[0].m128i_i64[0] = 17;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      v49 = sub_22409D0((__int64)a1, (unsigned __int64 *)v107, 0);
      v50 = v107[0].m128i_i64[0];
      v51 = _mm_load_si128((const __m128i *)&xmmword_44E0B20);
      a1->m128i_i64[0] = v49;
      a1[1].m128i_i64[0] = v50;
      *(_BYTE *)(v49 + 16) = 114;
      *(__m128i *)v49 = v51;
      v52 = v107[0].m128i_i64[0];
      v53 = a1->m128i_i64[0];
      a1->m128i_i64[1] = v107[0].m128i_i64[0];
      *(_BYTE *)(v53 + v52) = 0;
      return a1;
    case 328:
      sub_34182A0(a1->m128i_i64, "handlenode");
      return a1;
    case 329:
      sub_34182A0(a1->m128i_i64, "init_trampoline");
      return a1;
    case 330:
      v107[0].m128i_i64[0] = 17;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      v44 = sub_22409D0((__int64)a1, (unsigned __int64 *)v107, 0);
      v45 = v107[0].m128i_i64[0];
      v46 = _mm_load_si128((const __m128i *)&xmmword_44E0B90);
      a1->m128i_i64[0] = v44;
      a1[1].m128i_i64[0] = v45;
      *(_BYTE *)(v44 + 16) = 101;
      *(__m128i *)v44 = v46;
      v47 = v107[0].m128i_i64[0];
      v48 = a1->m128i_i64[0];
      a1->m128i_i64[1] = v107[0].m128i_i64[0];
      *(_BYTE *)(v48 + v47) = 0;
      return a1;
    case 331:
      sub_34182A0(a1->m128i_i64, "trap");
      return a1;
    case 332:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "debugtrap");
      a1->m128i_i64[1] = 9;
      return a1;
    case 333:
      sub_34182A0(a1->m128i_i64, "ubsantrap");
      return a1;
    case 334:
      sub_34182A0(a1->m128i_i64, "Prefetch");
      return a1;
    case 336:
      sub_34182A0(a1->m128i_i64, "MemBarrier");
      return a1;
    case 337:
      sub_34182A0(a1->m128i_i64, "AtomicFence");
      return a1;
    case 338:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "AtomicLoad");
      a1->m128i_i64[1] = 10;
      return a1;
    case 339:
      sub_34182A0(a1->m128i_i64, "AtomicStore");
      return a1;
    case 340:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "AtomicCmpSwap");
      a1->m128i_i64[1] = 13;
      return a1;
    case 341:
      sub_34182A0(a1->m128i_i64, "AtomicCmpSwapWithSuccess");
      return a1;
    case 342:
      sub_34182A0(a1->m128i_i64, "AtomicSwap");
      return a1;
    case 343:
      sub_34182A0(a1->m128i_i64, "AtomicLoadAdd");
      return a1;
    case 344:
      sub_34182A0(a1->m128i_i64, "AtomicLoadSub");
      return a1;
    case 345:
      sub_34182A0(a1->m128i_i64, "AtomicLoadAnd");
      return a1;
    case 346:
      sub_34182A0(a1->m128i_i64, "AtomicLoadClr");
      return a1;
    case 347:
      sub_34182A0(a1->m128i_i64, "AtomicLoadOr");
      return a1;
    case 348:
      sub_34182A0(a1->m128i_i64, "AtomicLoadXor");
      return a1;
    case 349:
      sub_34182A0(a1->m128i_i64, "AtomicLoadNand");
      return a1;
    case 350:
      sub_34182A0(a1->m128i_i64, "AtomicLoadMin");
      return a1;
    case 351:
      sub_34182A0(a1->m128i_i64, "AtomicLoadMax");
      return a1;
    case 352:
      sub_34182A0(a1->m128i_i64, "AtomicLoadUMin");
      return a1;
    case 353:
      sub_34182A0(a1->m128i_i64, "AtomicLoadUMax");
      return a1;
    case 354:
      sub_34182A0(a1->m128i_i64, "AtomicLoadFAdd");
      return a1;
    case 355:
      sub_34182A0(a1->m128i_i64, "AtomicLoadFSub");
      return a1;
    case 356:
      sub_34182A0(a1->m128i_i64, "AtomicLoadFMax");
      return a1;
    case 357:
      sub_34182A0(a1->m128i_i64, "AtomicLoadFMin");
      return a1;
    case 358:
      sub_34182A0(a1->m128i_i64, "AtomicLoadUIncWrap");
      return a1;
    case 359:
      sub_34182A0(a1->m128i_i64, "AtomicLoadUDecWrap");
      return a1;
    case 360:
      v107[0].m128i_i64[0] = 18;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      v39 = sub_22409D0((__int64)a1, (unsigned __int64 *)v107, 0);
      v40 = v107[0].m128i_i64[0];
      v41 = _mm_load_si128((const __m128i *)&xmmword_44E0B10);
      a1->m128i_i64[0] = v39;
      a1[1].m128i_i64[0] = v40;
      *(_WORD *)(v39 + 16) = 25710;
      *(__m128i *)v39 = v41;
      v42 = v107[0].m128i_i64[0];
      v43 = a1->m128i_i64[0];
      a1->m128i_i64[1] = v107[0].m128i_i64[0];
      *(_BYTE *)(v43 + v42) = 0;
      return a1;
    case 361:
      sub_34182A0(a1->m128i_i64, "AtomicLoadUSubSat");
      return a1;
    case 362:
      sub_34182A0(a1->m128i_i64, "masked_load");
      return a1;
    case 363:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "masked_store");
      a1->m128i_i64[1] = 12;
      return a1;
    case 364:
      sub_34182A0(a1->m128i_i64, "masked_gather");
      return a1;
    case 365:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "masked_scatter");
      a1->m128i_i64[1] = 14;
      return a1;
    case 366:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "lifetime.start");
      a1->m128i_i64[1] = 14;
      return a1;
    case 367:
      sub_34182A0(a1->m128i_i64, "lifetime.end");
      return a1;
    case 368:
      a1->m128i_i64[1] = 8;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "fake_use");
      return a1;
    case 369:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      v107[0].m128i_i64[0] = 19;
      v34 = sub_22409D0((__int64)a1, (unsigned __int64 *)v107, 0);
      v35 = v107[0].m128i_i64[0];
      v36 = _mm_load_si128((const __m128i *)&xmmword_44E0B60);
      a1->m128i_i64[0] = v34;
      a1[1].m128i_i64[0] = v35;
      *(_WORD *)(v34 + 16) = 29281;
      *(_BYTE *)(v34 + 18) = 116;
      *(__m128i *)v34 = v36;
      v37 = v107[0].m128i_i64[0];
      v38 = a1->m128i_i64[0];
      a1->m128i_i64[1] = v107[0].m128i_i64[0];
      *(_BYTE *)(v38 + v37) = 0;
      return a1;
    case 370:
      sub_34182A0(a1->m128i_i64, "gc_transition.end");
      return a1;
    case 371:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      v107[0].m128i_i64[0] = 23;
      v29 = sub_22409D0((__int64)a1, (unsigned __int64 *)v107, 0);
      v30 = v107[0].m128i_i64[0];
      v31 = _mm_load_si128((const __m128i *)&xmmword_44E0B70);
      a1->m128i_i64[0] = v29;
      a1[1].m128i_i64[0] = v30;
      *(_DWORD *)(v29 + 16) = 1717989166;
      *(_WORD *)(v29 + 20) = 25971;
      *(_BYTE *)(v29 + 22) = 116;
      *(__m128i *)v29 = v31;
      v32 = v107[0].m128i_i64[0];
      v33 = a1->m128i_i64[0];
      a1->m128i_i64[1] = v107[0].m128i_i64[0];
      *(_BYTE *)(v33 + v32) = 0;
      return a1;
    case 372:
      sub_34182A0(a1->m128i_i64, "pseudoprobe");
      return a1;
    case 373:
      sub_34182A0(a1->m128i_i64, "vscale");
      return a1;
    case 374:
      v107[0].m128i_i64[0] = 18;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      v24 = sub_22409D0((__int64)a1, (unsigned __int64 *)v107, 0);
      v25 = v107[0].m128i_i64[0];
      v26 = _mm_load_si128((const __m128i *)&xmmword_44E0BA0);
      a1->m128i_i64[0] = v24;
      a1[1].m128i_i64[0] = v25;
      *(_WORD *)(v24 + 16) = 25700;
      *(__m128i *)v24 = v26;
      v27 = v107[0].m128i_i64[0];
      v28 = a1->m128i_i64[0];
      a1->m128i_i64[1] = v107[0].m128i_i64[0];
      *(_BYTE *)(v28 + v27) = 0;
      return a1;
    case 375:
      v107[0].m128i_i64[0] = 18;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      v19 = sub_22409D0((__int64)a1, (unsigned __int64 *)v107, 0);
      v20 = v107[0].m128i_i64[0];
      v21 = _mm_load_si128((const __m128i *)&xmmword_44E0BB0);
      a1->m128i_i64[0] = v19;
      a1[1].m128i_i64[0] = v20;
      *(_WORD *)(v19 + 16) = 27765;
      *(__m128i *)v19 = v21;
      v22 = v107[0].m128i_i64[0];
      v23 = a1->m128i_i64[0];
      a1->m128i_i64[1] = v107[0].m128i_i64[0];
      *(_BYTE *)(v23 + v22) = 0;
      return a1;
    case 376:
      sub_34182A0(a1->m128i_i64, "vecreduce_fadd");
      return a1;
    case 377:
      sub_34182A0(a1->m128i_i64, "vecreduce_fmul");
      return a1;
    case 378:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "vecreduce_fmax");
      a1->m128i_i64[1] = 14;
      return a1;
    case 379:
      sub_34182A0(a1->m128i_i64, "vecreduce_fmin");
      return a1;
    case 380:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      v107[0].m128i_i64[0] = 18;
      v14 = sub_22409D0((__int64)a1, (unsigned __int64 *)v107, 0);
      v15 = v107[0].m128i_i64[0];
      v16 = _mm_load_si128((const __m128i *)&xmmword_44E0BC0);
      a1->m128i_i64[0] = v14;
      a1[1].m128i_i64[0] = v15;
      *(_WORD *)(v14 + 16) = 28021;
      *(__m128i *)v14 = v16;
      v17 = v107[0].m128i_i64[0];
      v18 = a1->m128i_i64[0];
      a1->m128i_i64[1] = v107[0].m128i_i64[0];
      *(_BYTE *)(v18 + v17) = 0;
      return a1;
    case 381:
      sub_34182A0(a1->m128i_i64, "vecreduce_fminimum");
      return a1;
    case 382:
      sub_34182A0(a1->m128i_i64, "vecreduce_add");
      return a1;
    case 383:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "vecreduce_mul");
      a1->m128i_i64[1] = 13;
      return a1;
    case 384:
      sub_34182A0(a1->m128i_i64, "vecreduce_and");
      return a1;
    case 385:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "vecreduce_or");
      a1->m128i_i64[1] = 12;
      return a1;
    case 386:
      sub_34182A0(a1->m128i_i64, "vecreduce_xor");
      return a1;
    case 387:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "vecreduce_smax");
      a1->m128i_i64[1] = 14;
      return a1;
    case 388:
      sub_34182A0(a1->m128i_i64, "vecreduce_smin");
      return a1;
    case 389:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "vecreduce_umax");
      a1->m128i_i64[1] = 14;
      return a1;
    case 390:
      sub_34182A0(a1->m128i_i64, "vecreduce_umin");
      return a1;
    case 391:
      v107[0].m128i_i64[0] = 19;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      v9 = sub_22409D0((__int64)a1, (unsigned __int64 *)v107, 0);
      v10 = v107[0].m128i_i64[0];
      v11 = _mm_load_si128((const __m128i *)&xmmword_44E0BE0);
      a1->m128i_i64[0] = v9;
      a1[1].m128i_i64[0] = v10;
      *(_WORD *)(v9 + 16) = 27757;
      *(_BYTE *)(v9 + 18) = 97;
      *(__m128i *)v9 = v11;
      v12 = v107[0].m128i_i64[0];
      v13 = a1->m128i_i64[0];
      a1->m128i_i64[1] = v107[0].m128i_i64[0];
      *(_BYTE *)(v13 + v12) = 0;
      return a1;
    case 392:
      sub_34182A0(a1->m128i_i64, "partial_reduce_umla");
      return a1;
    case 393:
      a1->m128i_i64[1] = 8;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "stackmap");
      return a1;
    case 394:
      sub_34182A0(a1->m128i_i64, "patchpoint");
      return a1;
    case 395:
      sub_34182A0(a1->m128i_i64, "vp_add");
      return a1;
    case 396:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "vp_and");
      a1->m128i_i64[1] = 6;
      return a1;
    case 397:
      sub_34182A0(a1->m128i_i64, "vp_ashr");
      return a1;
    case 398:
      a1[1].m128i_i8[6] = 114;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      a1[1].m128i_i32[0] = 1818194038;
      a1[1].m128i_i16[2] = 26739;
      a1->m128i_i64[1] = 7;
      a1[1].m128i_i8[7] = 0;
      return a1;
    case 399:
      sub_34182A0(a1->m128i_i64, "vp_mul");
      return a1;
    case 400:
      a1[1].m128i_i32[0] = 1868525686;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      a1[1].m128i_i8[4] = 114;
      a1->m128i_i64[1] = 5;
      a1[1].m128i_i8[5] = 0;
      return a1;
    case 401:
      sub_34182A0(a1->m128i_i64, "vp_sdiv");
      return a1;
    case 402:
      strcpy(a1[1].m128i_i8, "vp_shl");
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      a1->m128i_i64[1] = 6;
      return a1;
    case 403:
      sub_34182A0(a1->m128i_i64, "vp_srem");
      return a1;
    case 404:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "vp_sub");
      a1->m128i_i64[1] = 6;
      return a1;
    case 405:
      sub_34182A0(a1->m128i_i64, "vp_udiv");
      return a1;
    case 406:
      a1[1].m128i_i8[6] = 109;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      a1[1].m128i_i32[0] = 1969188982;
      a1[1].m128i_i16[2] = 25970;
      a1->m128i_i64[1] = 7;
      a1[1].m128i_i8[7] = 0;
      return a1;
    case 407:
      sub_34182A0(a1->m128i_i64, "vp_xor");
      return a1;
    case 408:
      a1[1].m128i_i8[6] = 110;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      a1[1].m128i_i32[0] = 1935634550;
      a1[1].m128i_i16[2] = 26989;
      a1->m128i_i64[1] = 7;
      a1[1].m128i_i8[7] = 0;
      return a1;
    case 409:
      sub_34182A0(a1->m128i_i64, "vp_smax");
      return a1;
    case 410:
      a1[1].m128i_i32[0] = 1969188982;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      a1[1].m128i_i16[2] = 26989;
      a1[1].m128i_i8[6] = 110;
      a1->m128i_i64[1] = 7;
      a1[1].m128i_i8[7] = 0;
      return a1;
    case 411:
      sub_34182A0(a1->m128i_i64, "vp_umax");
      return a1;
    case 412:
      strcpy(a1[1].m128i_i8, "vp_abs");
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      a1->m128i_i64[1] = 6;
      return a1;
    case 413:
      sub_34182A0(a1->m128i_i64, "vp_bswap");
      return a1;
    case 414:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "vp_bitreverse");
      a1->m128i_i64[1] = 13;
      return a1;
    case 415:
      sub_34182A0(a1->m128i_i64, "vp_ctpop");
      return a1;
    case 416:
      a1[1].m128i_i32[0] = 1667199094;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      a1[1].m128i_i16[2] = 27764;
      a1[1].m128i_i8[6] = 122;
      a1->m128i_i64[1] = 7;
      a1[1].m128i_i8[7] = 0;
      return a1;
    case 417:
      sub_34182A0(a1->m128i_i64, "vp_ctlz_zero_undef");
      return a1;
    case 418:
      a1[1].m128i_i32[0] = 1667199094;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      a1[1].m128i_i16[2] = 29812;
      a1[1].m128i_i8[6] = 122;
      a1->m128i_i64[1] = 7;
      a1[1].m128i_i8[7] = 0;
      return a1;
    case 419:
      sub_34182A0(a1->m128i_i64, "vp_cttz_zero_undef");
      return a1;
    case 420:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "vp_cttz_elts");
      a1->m128i_i64[1] = 12;
      return a1;
    case 421:
      sub_34182A0(a1->m128i_i64, "vp_cttz_elts_zero_undef");
      return a1;
    case 422:
      a1[1].m128i_i32[0] = 1717530742;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      a1[1].m128i_i16[2] = 26739;
      a1[1].m128i_i8[6] = 108;
      a1->m128i_i64[1] = 7;
      a1[1].m128i_i8[7] = 0;
      return a1;
    case 423:
      sub_34182A0(a1->m128i_i64, "vp_fshr");
      return a1;
    case 424:
      sub_34182A0(a1->m128i_i64, "vp_sadd_sat");
      return a1;
    case 425:
      sub_34182A0(a1->m128i_i64, "vp_uadd_sat");
      return a1;
    case 426:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "vp_ssub_sat");
      a1->m128i_i64[1] = 11;
      return a1;
    case 427:
      sub_34182A0(a1->m128i_i64, "vp_usub_sat");
      return a1;
    case 428:
      a1[1].m128i_i8[6] = 100;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      a1[1].m128i_i32[0] = 1717530742;
      a1[1].m128i_i16[2] = 25697;
      a1->m128i_i64[1] = 7;
      a1[1].m128i_i8[7] = 0;
      return a1;
    case 429:
      sub_34182A0(a1->m128i_i64, "vp_fsub");
      return a1;
    case 430:
      a1[1].m128i_i8[6] = 108;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      a1[1].m128i_i32[0] = 1717530742;
      a1[1].m128i_i16[2] = 30061;
      a1->m128i_i64[1] = 7;
      a1[1].m128i_i8[7] = 0;
      return a1;
    case 431:
      sub_34182A0(a1->m128i_i64, "vp_fdiv");
      return a1;
    case 432:
      a1[1].m128i_i8[6] = 109;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      a1[1].m128i_i32[0] = 1717530742;
      a1[1].m128i_i16[2] = 25970;
      a1->m128i_i64[1] = 7;
      a1[1].m128i_i8[7] = 0;
      return a1;
    case 433:
      sub_34182A0(a1->m128i_i64, "vp_fneg");
      return a1;
    case 434:
      a1[1].m128i_i8[6] = 115;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      a1[1].m128i_i32[0] = 1717530742;
      a1[1].m128i_i16[2] = 25185;
      a1->m128i_i64[1] = 7;
      a1[1].m128i_i8[7] = 0;
      return a1;
    case 435:
      sub_34182A0(a1->m128i_i64, "vp_sqrt");
      return a1;
    case 436:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "vp_fma");
      a1->m128i_i64[1] = 6;
      return a1;
    case 437:
      sub_34182A0(a1->m128i_i64, "vp_fmuladd");
      return a1;
    case 438:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "vp_copysign");
      a1->m128i_i64[1] = 11;
      return a1;
    case 439:
      sub_34182A0(a1->m128i_i64, "vp_minnum");
      return a1;
    case 440:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "vp_maxnum");
      a1->m128i_i64[1] = 9;
      return a1;
    case 441:
      sub_34182A0(a1->m128i_i64, "vp_minimum");
      return a1;
    case 442:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "vp_maximum");
      a1->m128i_i64[1] = 10;
      return a1;
    case 443:
      sub_34182A0(a1->m128i_i64, "vp_ceil");
      return a1;
    case 444:
      a1->m128i_i64[1] = 8;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "vp_floor");
      return a1;
    case 445:
      sub_34182A0(a1->m128i_i64, "vp_round");
      return a1;
    case 446:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "vp_roundeven");
      a1->m128i_i64[1] = 12;
      return a1;
    case 447:
      sub_34182A0(a1->m128i_i64, "vp_roundtozero");
      return a1;
    case 448:
      a1[1].m128i_i32[0] = 1918857334;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      a1[1].m128i_i16[2] = 28265;
      a1[1].m128i_i8[6] = 116;
      a1->m128i_i64[1] = 7;
      a1[1].m128i_i8[7] = 0;
      return a1;
    case 449:
      sub_34182A0(a1->m128i_i64, "vp_nearbyint");
      return a1;
    case 450:
      a1->m128i_i64[1] = 8;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "vp_lrint");
      return a1;
    case 451:
      sub_34182A0(a1->m128i_i64, "vp_llrint");
      return a1;
    case 452:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "vp_fptoui");
      a1->m128i_i64[1] = 9;
      return a1;
    case 453:
      sub_34182A0(a1->m128i_i64, "vp_fptosi");
      return a1;
    case 454:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "vp_uitofp");
      a1->m128i_i64[1] = 9;
      return a1;
    case 455:
      sub_34182A0(a1->m128i_i64, "vp_sitofp");
      return a1;
    case 456:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "vp_fptrunc");
      a1->m128i_i64[1] = 10;
      return a1;
    case 457:
      sub_34182A0(a1->m128i_i64, "vp_fpext");
      return a1;
    case 458:
      a1->m128i_i64[1] = 8;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "vp_trunc");
      return a1;
    case 459:
      sub_34182A0(a1->m128i_i64, "vp_zext");
      return a1;
    case 460:
      a1[1].m128i_i8[6] = 116;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      a1[1].m128i_i32[0] = 1935634550;
      a1[1].m128i_i16[2] = 30821;
      a1->m128i_i64[1] = 7;
      a1[1].m128i_i8[7] = 0;
      return a1;
    case 461:
      sub_34182A0(a1->m128i_i64, "vp_ptrtoint");
      return a1;
    case 462:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "vp_inttoptr");
      a1->m128i_i64[1] = 11;
      return a1;
    case 463:
      sub_34182A0(a1->m128i_i64, "vp_setcc");
      return a1;
    case 464:
      sub_34182A0(a1->m128i_i64, "vp_is_fpclass");
      return a1;
    case 465:
      sub_34182A0(a1->m128i_i64, "vp_store");
      return a1;
    case 466:
      sub_34182A0(a1->m128i_i64, "experimental_vp_strided_store");
      return a1;
    case 467:
      sub_34182A0(a1->m128i_i64, "vp_scatter");
      return a1;
    case 468:
      sub_34182A0(a1->m128i_i64, "vp_load");
      return a1;
    case 469:
      sub_34182A0(a1->m128i_i64, "experimental_vp_strided_load");
      return a1;
    case 470:
      sub_34182A0(a1->m128i_i64, "vp_gather");
      return a1;
    case 471:
      sub_34182A0(a1->m128i_i64, "vp_reduce_add");
      return a1;
    case 472:
      sub_34182A0(a1->m128i_i64, "vp_reduce_mul");
      return a1;
    case 473:
      sub_34182A0(a1->m128i_i64, "vp_reduce_and");
      return a1;
    case 474:
      sub_34182A0(a1->m128i_i64, "vp_reduce_or");
      return a1;
    case 475:
      sub_34182A0(a1->m128i_i64, "vp_reduce_xor");
      return a1;
    case 476:
      sub_34182A0(a1->m128i_i64, "vp_reduce_smax");
      return a1;
    case 477:
      sub_34182A0(a1->m128i_i64, "vp_reduce_smin");
      return a1;
    case 478:
      sub_34182A0(a1->m128i_i64, "vp_reduce_umax");
      return a1;
    case 479:
      sub_34182A0(a1->m128i_i64, "vp_reduce_umin");
      return a1;
    case 480:
      sub_34182A0(a1->m128i_i64, "vp_reduce_fmax");
      return a1;
    case 481:
      sub_34182A0(a1->m128i_i64, "vp_reduce_fmin");
      return a1;
    case 482:
      sub_34182A0(a1->m128i_i64, "vp_reduce_fmaximum");
      return a1;
    case 483:
      sub_34182A0(a1->m128i_i64, "vp_reduce_fminimum");
      return a1;
    case 484:
      sub_34182A0(a1->m128i_i64, "vp_reduce_fadd");
      return a1;
    case 485:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "vp_reduce_fadd");
      a1->m128i_i64[1] = 14;
      return a1;
    case 486:
    case 487:
      sub_34182A0(a1->m128i_i64, "vp_reduce_fmul");
      return a1;
    case 488:
      sub_34182A0(a1->m128i_i64, "vp_select");
      return a1;
    case 489:
      sub_34182A0(a1->m128i_i64, "vp_merge");
      return a1;
    case 490:
      sub_34182A0(a1->m128i_i64, "experimental_vp_splice");
      return a1;
    case 491:
      sub_34182A0(a1->m128i_i64, "experimental_vp_reverse");
      return a1;
    case 492:
      sub_34182A0(a1->m128i_i64, "experimental_vp_splat");
      return a1;
    case 493:
      v107[0].m128i_i64[0] = 22;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      v89 = sub_22409D0((__int64)a1, (unsigned __int64 *)v107, 0);
      v90 = v107[0].m128i_i64[0];
      v91 = _mm_load_si128((const __m128i *)&xmmword_44E0B80);
      a1->m128i_i64[0] = v89;
      a1[1].m128i_i64[0] = v90;
      *(_DWORD *)(v89 + 16) = 1751346785;
      *(_WORD *)(v89 + 20) = 29295;
      *(__m128i *)v89 = v91;
      v92 = v107[0].m128i_i64[0];
      v93 = a1->m128i_i64[0];
      a1->m128i_i64[1] = v107[0].m128i_i64[0];
      *(_BYTE *)(v93 + v92) = 0;
      return a1;
    case 494:
      sub_34182A0(a1->m128i_i64, "convergencectrl_entry");
      return a1;
    case 495:
      v107[0].m128i_i64[0] = 20;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      v84 = sub_22409D0((__int64)a1, (unsigned __int64 *)v107, 0);
      v85 = v107[0].m128i_i64[0];
      v86 = _mm_load_si128((const __m128i *)&xmmword_44E0B80);
      a1->m128i_i64[0] = v84;
      a1[1].m128i_i64[0] = v85;
      *(_DWORD *)(v84 + 16) = 1886351212;
      *(__m128i *)v84 = v86;
      v87 = v107[0].m128i_i64[0];
      v88 = a1->m128i_i64[0];
      a1->m128i_i64[1] = v107[0].m128i_i64[0];
      *(_BYTE *)(v88 + v87) = 0;
      return a1;
    case 496:
      sub_34182A0(a1->m128i_i64, "convergencectrl_glue");
      return a1;
    case 497:
      sub_34182A0(a1->m128i_i64, "histogram");
      return a1;
    case 498:
      v107[0].m128i_i64[0] = 16;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      v79 = (_OWORD *)sub_22409D0((__int64)a1, (unsigned __int64 *)v107, 0);
      v80 = v107[0].m128i_i64[0];
      v81 = _mm_load_si128((const __m128i *)&xmmword_44E0BD0);
      a1->m128i_i64[0] = (__int64)v79;
      a1[1].m128i_i64[0] = v80;
      *v79 = v81;
      v82 = v107[0].m128i_i64[0];
      v83 = a1->m128i_i64[0];
      a1->m128i_i64[1] = v107[0].m128i_i64[0];
      *(_BYTE *)(v83 + v82) = 0;
      return a1;
    case 499:
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      strcpy(a1[1].m128i_i8, "clear_cache");
      a1->m128i_i64[1] = 11;
      return a1;
    default:
LABEL_530:
      if ( (v5 & 0x80000000) != 0LL )
      {
        if ( !a3
          || (v97 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(*(_QWORD *)(a3 + 40) + 16LL) + 128LL))(*(_QWORD *)(*(_QWORD *)(a3 + 40) + 16LL))) == 0
          || (v98 = (unsigned int)~*(_DWORD *)(a2 + 24), *(_DWORD *)(v97 + 48) <= (unsigned int)v98) )
        {
          sub_3418350((__int64 *)v106, *(unsigned int *)(a2 + 24), 0);
          sub_95D570(v107, "<<Unknown Machine Node #", (__int64)v106);
          sub_94F930(a1, (__int64)v107, ">>");
          sub_2240A30((unsigned __int64 *)v107);
          sub_2240A30(v106);
          return a1;
        }
        v99 = a1 + 1;
        v100 = (const char *)(*(_QWORD *)(v97 + 24) + *(unsigned int *)(*(_QWORD *)(v97 + 16) + 4 * v98));
        if ( v100 )
        {
          v101 = strlen(v100);
          a1->m128i_i64[0] = (__int64)v99;
          v107[0].m128i_i64[0] = v101;
          v102 = v101;
          if ( v101 <= 0xF )
          {
            if ( v101 == 1 )
            {
              a1[1].m128i_i8[0] = *v100;
              goto LABEL_549;
            }
            if ( !v101 )
              goto LABEL_549;
          }
          else
          {
            v103 = sub_22409D0((__int64)a1, (unsigned __int64 *)v107, 0);
            a1->m128i_i64[0] = v103;
            v99 = (__m128i *)v103;
            a1[1].m128i_i64[0] = v107[0].m128i_i64[0];
          }
          memcpy(v99, v100, v102);
        }
        else
        {
          a1->m128i_i64[0] = (__int64)v99;
          v107[0].m128i_i64[0] = 0;
        }
LABEL_549:
        v104 = v107[0].m128i_i64[0];
        v105 = a1->m128i_i64[0];
        a1->m128i_i64[1] = v107[0].m128i_i64[0];
        *(_BYTE *)(v105 + v104) = 0;
        return a1;
      }
      if ( a3 )
      {
        v94 = (const char *)(*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a3 + 16) + 2432LL))(*(_QWORD *)(a3 + 16));
        if ( v94 )
        {
          sub_34182A0(a1->m128i_i64, v94);
        }
        else
        {
          sub_3418350((__int64 *)v106, *(unsigned int *)(a2 + 24), 0);
          sub_95D570(v107, "<<Unknown Target Node #", (__int64)v106);
          sub_94F930(a1, (__int64)v107, ">>");
          sub_2240A30((unsigned __int64 *)v107);
          sub_2240A30(v106);
        }
      }
      else
      {
        sub_3418350((__int64 *)v106, v5, 0);
        sub_95D570(v107, "<<Unknown Node #", (__int64)v106);
        sub_94F930(a1, (__int64)v107, ">>");
        sub_2240A30((unsigned __int64 *)v107);
        sub_2240A30(v106);
      }
      return a1;
  }
}
