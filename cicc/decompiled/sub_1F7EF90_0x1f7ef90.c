// Function: sub_1F7EF90
// Address: 0x1f7ef90
//
__int64 __fastcall sub_1F7EF90(__int64 a1, __int64 *a2, double a3, double a4, double a5)
{
  __int64 *v5; // rax
  __int64 v6; // r12
  __int64 v7; // r14
  __int64 v8; // rax
  int v9; // edx
  char v10; // r13
  __int64 result; // rax
  __m128i v14; // xmm0
  __int64 v15; // rax
  char v16; // di
  __int64 v17; // rsi
  bool v18; // zf
  int v19; // eax
  const void **v20; // rdx
  int v21; // eax
  unsigned int v22; // eax
  unsigned int v23; // r13d
  __int64 v24; // r14
  char v25; // al
  __int64 v26; // rdx
  __int64 v27; // rdx
  void *v28; // rdi
  __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // r8
  __int64 v32; // r9
  unsigned int v33; // r14d
  __int64 v34; // rdx
  __int64 v35; // rcx
  __int64 v36; // r8
  __int64 v37; // r9
  __int64 v38; // rdx
  __int64 v39; // rcx
  __int64 v40; // r8
  __int64 v41; // r9
  unsigned int v42; // r14d
  __int64 v43; // rsi
  unsigned int v44; // r14d
  __int64 v45; // rdx
  __int64 v46; // rcx
  __int64 v47; // r8
  __int64 v48; // r9
  __int64 v49; // rsi
  unsigned int v50; // [rsp+18h] [rbp-78h]
  bool v51; // [rsp+20h] [rbp-70h]
  __int64 v52; // [rsp+20h] [rbp-70h]
  bool v53; // [rsp+28h] [rbp-68h]
  unsigned int v54; // [rsp+30h] [rbp-60h] BYREF
  const void **v55; // [rsp+38h] [rbp-58h]
  char v56[8]; // [rsp+40h] [rbp-50h] BYREF
  __int64 v57; // [rsp+48h] [rbp-48h]
  __int64 v58; // [rsp+50h] [rbp-40h] BYREF
  __int64 v59; // [rsp+58h] [rbp-38h]

  v5 = *(__int64 **)(a1 + 32);
  v6 = *v5;
  v7 = *((unsigned int *)v5 + 2);
  v8 = *(_QWORD *)(a1 + 40);
  v9 = *(unsigned __int16 *)(v6 + 24);
  v10 = *(_BYTE *)v8;
  v55 = *(const void ***)(v8 + 8);
  LOBYTE(v54) = v10;
  if ( (unsigned int)(v9 - 146) > 1 )
    return 0;
  v14 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(v6 + 32));
  v15 = *(_QWORD *)(**(_QWORD **)(v6 + 32) + 40LL) + 16LL * *(unsigned int *)(*(_QWORD *)(v6 + 32) + 8LL);
  v16 = *(_BYTE *)v15;
  v17 = *(_QWORD *)(v15 + 8);
  v53 = v9 == 146;
  v18 = *(_WORD *)(a1 + 24) == 152;
  v56[0] = v16;
  v57 = v17;
  v51 = v18;
  if ( v16 )
  {
    if ( (unsigned __int8)(v16 - 14) <= 0x5Fu )
    {
      switch ( v16 )
      {
        case 24:
        case 25:
        case 26:
        case 27:
        case 28:
        case 29:
        case 30:
        case 31:
        case 32:
        case 62:
        case 63:
        case 64:
        case 65:
        case 66:
        case 67:
          v16 = 3;
          break;
        case 33:
        case 34:
        case 35:
        case 36:
        case 37:
        case 38:
        case 39:
        case 40:
        case 68:
        case 69:
        case 70:
        case 71:
        case 72:
        case 73:
          v16 = 4;
          break;
        case 41:
        case 42:
        case 43:
        case 44:
        case 45:
        case 46:
        case 47:
        case 48:
        case 74:
        case 75:
        case 76:
        case 77:
        case 78:
        case 79:
          v16 = 5;
          break;
        case 49:
        case 50:
        case 51:
        case 52:
        case 53:
        case 54:
        case 80:
        case 81:
        case 82:
        case 83:
        case 84:
        case 85:
          v16 = 6;
          break;
        case 55:
          v16 = 7;
          break;
        case 86:
        case 87:
        case 88:
        case 98:
        case 99:
        case 100:
          v16 = 8;
          break;
        case 89:
        case 90:
        case 91:
        case 92:
        case 93:
        case 101:
        case 102:
        case 103:
        case 104:
        case 105:
          v16 = 9;
          break;
        case 94:
        case 95:
        case 96:
        case 97:
        case 106:
        case 107:
        case 108:
        case 109:
          v16 = 10;
          break;
        default:
          v16 = 2;
          break;
      }
    }
    goto LABEL_6;
  }
  if ( sub_1F58D20((__int64)v56) )
  {
    v10 = v54;
    LOBYTE(v58) = sub_1F596B0((__int64)v56);
    v16 = v58;
    v59 = v27;
    if ( (_BYTE)v58 )
    {
LABEL_6:
      v19 = sub_1F6C8D0(v16);
      goto LABEL_7;
    }
  }
  else
  {
    LOBYTE(v58) = 0;
    v59 = v17;
  }
  v19 = sub_1F58D40((__int64)&v58);
LABEL_7:
  v50 = v19 - v53;
  if ( v10 )
  {
    if ( (unsigned __int8)(v10 - 14) <= 0x5Fu )
    {
      switch ( v10 )
      {
        case 24:
        case 25:
        case 26:
        case 27:
        case 28:
        case 29:
        case 30:
        case 31:
        case 32:
        case 62:
        case 63:
        case 64:
        case 65:
        case 66:
        case 67:
          v10 = 3;
          break;
        case 33:
        case 34:
        case 35:
        case 36:
        case 37:
        case 38:
        case 39:
        case 40:
        case 68:
        case 69:
        case 70:
        case 71:
        case 72:
        case 73:
          v10 = 4;
          break;
        case 41:
        case 42:
        case 43:
        case 44:
        case 45:
        case 46:
        case 47:
        case 48:
        case 74:
        case 75:
        case 76:
        case 77:
        case 78:
        case 79:
          v10 = 5;
          break;
        case 49:
        case 50:
        case 51:
        case 52:
        case 53:
        case 54:
        case 80:
        case 81:
        case 82:
        case 83:
        case 84:
        case 85:
          v10 = 6;
          break;
        case 55:
          v10 = 7;
          break;
        case 86:
        case 87:
        case 88:
        case 98:
        case 99:
        case 100:
          v10 = 8;
          break;
        case 89:
        case 90:
        case 91:
        case 92:
        case 93:
        case 101:
        case 102:
        case 103:
        case 104:
        case 105:
          v10 = 9;
          break;
        case 94:
        case 95:
        case 96:
        case 97:
        case 106:
        case 107:
        case 108:
        case 109:
          v10 = 10;
          break;
        default:
          v10 = 2;
          break;
      }
      goto LABEL_20;
    }
    goto LABEL_9;
  }
  if ( !sub_1F58D20((__int64)&v54) )
  {
LABEL_9:
    v20 = v55;
    goto LABEL_10;
  }
  v10 = sub_1F596B0((__int64)&v54);
LABEL_10:
  LOBYTE(v58) = v10;
  v59 = (__int64)v20;
  if ( !v10 )
  {
    v21 = sub_1F58D40((__int64)&v58);
    goto LABEL_12;
  }
LABEL_20:
  v21 = sub_1F6C8D0(v10);
LABEL_12:
  v22 = v21 - v51;
  v23 = v50;
  if ( v22 <= v50 )
    v23 = v22;
  v24 = *(_QWORD *)(v6 + 40) + 16 * v7;
  v25 = *(_BYTE *)v24;
  v26 = *(_QWORD *)(v24 + 8);
  LOBYTE(v58) = v25;
  v59 = v26;
  if ( !v25 )
  {
    if ( !sub_1F58D20((__int64)&v58) )
    {
LABEL_69:
      ++*(_DWORD *)(a1 + 576);
      BUG();
    }
    v25 = sub_1F596B0((__int64)&v58);
LABEL_16:
    switch ( v25 )
    {
      case 8:
        goto LABEL_45;
      case 9:
        goto LABEL_44;
      case 10:
        goto LABEL_43;
      case 11:
        v28 = sub_16982A0();
        goto LABEL_32;
      case 12:
        v28 = sub_1698290();
        goto LABEL_32;
      case 13:
        v28 = sub_16982C0();
        goto LABEL_32;
      default:
        goto LABEL_69;
    }
  }
  if ( (unsigned __int8)(v25 - 14) > 0x5Fu )
    goto LABEL_16;
  switch ( v25 )
  {
    case 'V':
    case 'W':
    case 'X':
    case 'b':
    case 'c':
    case 'd':
LABEL_45:
      v28 = sub_1698260();
      goto LABEL_32;
    case 'Y':
    case 'Z':
    case '[':
    case '\\':
    case ']':
    case 'e':
    case 'f':
    case 'g':
    case 'h':
    case 'i':
LABEL_44:
      v28 = sub_1698270();
      goto LABEL_32;
    case '^':
    case '_':
    case '`':
    case 'a':
    case 'j':
    case 'k':
    case 'l':
    case 'm':
LABEL_43:
      v28 = sub_1698280();
LABEL_32:
      if ( (unsigned int)sub_16982D0((__int64)v28) < v23 )
        return 0;
      v33 = sub_1D159C0((__int64)&v54, v17, v29, v30, v31, v32);
      if ( v33 <= (unsigned int)sub_1D159C0((__int64)v56, v17, v34, v35, v36, v37) )
      {
        v44 = sub_1D159C0((__int64)&v54, v17, v38, v39, v40, v41);
        if ( v44 >= (unsigned int)sub_1D159C0((__int64)v56, v17, v45, v46, v47, v48) )
          return sub_1D32840(a2, v54, v55, v14.m128i_i64[0], v14.m128i_i64[1], *(double *)v14.m128i_i64, a4, a5);
        v49 = *(_QWORD *)(a1 + 72);
        v58 = v49;
        if ( v49 )
          sub_1623A60((__int64)&v58, v49, 2);
        LODWORD(v59) = *(_DWORD *)(a1 + 64);
        result = sub_1D309E0(a2, 145, (__int64)&v58, v54, v55, 0, *(double *)v14.m128i_i64, a4, a5, *(_OWORD *)&v14);
      }
      else
      {
        if ( !v53 || (v42 = 142, !v51) )
          v42 = 143;
        v43 = *(_QWORD *)(a1 + 72);
        v58 = v43;
        if ( v43 )
          sub_1623A60((__int64)&v58, v43, 2);
        LODWORD(v59) = *(_DWORD *)(a1 + 64);
        result = sub_1D309E0(a2, v42, (__int64)&v58, v54, v55, 0, *(double *)v14.m128i_i64, a4, a5, *(_OWORD *)&v14);
      }
      if ( v58 )
      {
        v52 = result;
        sub_161E7C0((__int64)&v58, v58);
        result = v52;
      }
      break;
    default:
      goto LABEL_69;
  }
  return result;
}
