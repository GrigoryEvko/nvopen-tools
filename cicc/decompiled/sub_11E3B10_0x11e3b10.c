// Function: sub_11E3B10
// Address: 0x11e3b10
//
__int64 __fastcall sub_11E3B10(__int64 a1, __int64 a2, __int64 a3, _DWORD *a4)
{
  _DWORD *v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx
  int v13; // r9d
  __int64 result; // rax
  unsigned __int8 v15; // r8
  unsigned int v16; // ecx
  __int64 *v17; // rdx
  __int64 v18; // rdi
  unsigned int v19; // r8d
  __int64 v20; // rax
  __int64 *v21; // rcx
  __int64 v22; // rsi
  __int64 v23; // rdi
  unsigned int v24; // r8d
  __int64 v25; // rax
  __int64 *v26; // rcx
  __int64 v27; // rsi
  __int64 v28; // rdi
  __int64 v29; // rax
  __int64 v30; // rax
  unsigned __int8 v31; // r8
  __int64 *v32; // rdx
  __int64 v33; // rdi
  __int64 v34; // [rsp+10h] [rbp-40h] BYREF
  __int64 v35[7]; // [rsp+18h] [rbp-38h] BYREF

  if ( !(_BYTE)qword_4F91328 )
    return 0;
  v34 = *(_QWORD *)(a2 + 72);
  v35[0] = sub_A747B0(&v34, -1, "memprof", 7u);
  v7 = (_DWORD *)sub_A72240(v35);
  if ( v8 == 4 && *v7 == 1684828003 )
  {
    v13 = (unsigned __int8)byte_4F91168;
    goto LABEL_13;
  }
  v34 = *(_QWORD *)(a2 + 72);
  v35[0] = sub_A747B0(&v34, -1, "memprof", 7u);
  v9 = sub_A72240(v35);
  if ( v10 != 7
    || *(_DWORD *)v9 != 1668575086
    || *(_WORD *)(v9 + 4) != 27759
    || (v13 = (unsigned __int8)dword_4F91088, *(_BYTE *)(v9 + 6) != 100) )
  {
    v34 = *(_QWORD *)(a2 + 72);
    v35[0] = sub_A747B0(&v34, -1, "memprof", 7u);
    v11 = sub_A72240(v35);
    if ( v12 == 3 && *(_WORD *)v11 == 28520 && *(_BYTE *)(v11 + 2) == 116 )
    {
      v13 = (unsigned __int8)byte_4F90FA8;
      goto LABEL_13;
    }
    return 0;
  }
LABEL_13:
  switch ( *a4 )
  {
    case '*':
      v15 = v13;
      if ( (unsigned __int8)v13 == dword_4F91088 )
        return 0;
      v16 = 43;
      v17 = *(__int64 **)(a1 + 24);
      v18 = *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
      return sub_11CBFA0(v18, a3, v17, v16, v15);
    case '+':
      if ( !(_BYTE)qword_4F91248 )
        return 0;
      v15 = v13;
      v16 = 43;
      v17 = *(__int64 **)(a1 + 24);
      v18 = *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
      return sub_11CBFA0(v18, a3, v17, v16, v15);
    case ',':
      if ( v13 == dword_4F91088 )
        return 0;
      goto LABEL_20;
    case '-':
      if ( !(_BYTE)qword_4F91248 )
        return 0;
LABEL_20:
      v19 = 45;
      v20 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
      v21 = *(__int64 **)(a1 + 24);
      v22 = 32 * (1 - v20);
      v23 = *(_QWORD *)(a2 - 32 * v20);
      return sub_11CC240(v23, *(_QWORD *)(a2 + v22), a3, v21, v19, v13);
    case '.':
      if ( v13 == dword_4F91088 )
        return 0;
      goto LABEL_25;
    case '/':
      if ( !(_BYTE)qword_4F91248 )
        return 0;
LABEL_25:
      v24 = 47;
      v25 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
      v26 = *(__int64 **)(a1 + 24);
      v27 = 32 * (1 - v25);
      v28 = *(_QWORD *)(a2 - 32 * v25);
      return sub_11CC520(v28, *(_QWORD *)(a2 + v27), a3, v26, v24, v13);
    case '0':
      if ( v13 == dword_4F91088 )
        return 0;
      return sub_11CC530(
               *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)),
               *(_QWORD *)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))),
               *(_QWORD *)(a2 + 32 * (2LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))),
               a3,
               *(__int64 **)(a1 + 24),
               0x31u,
               v13);
    case '1':
      if ( !(_BYTE)qword_4F91248 )
        return 0;
      return sub_11CC530(
               *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)),
               *(_QWORD *)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))),
               *(_QWORD *)(a2 + 32 * (2LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))),
               a3,
               *(__int64 **)(a1 + 24),
               0x31u,
               v13);
    case '6':
      v15 = v13;
      if ( (unsigned __int8)v13 == dword_4F91088 )
        return 0;
      v17 = *(__int64 **)(a1 + 24);
      v18 = *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
      goto LABEL_35;
    case '7':
      if ( !(_BYTE)qword_4F91248 )
        return 0;
      v15 = v13;
      v17 = *(__int64 **)(a1 + 24);
      v18 = *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
LABEL_35:
      v16 = 55;
      return sub_11CBFA0(v18, a3, v17, v16, v15);
    case '8':
      if ( v13 == dword_4F91088 )
        return 0;
      goto LABEL_39;
    case '9':
      if ( !(_BYTE)qword_4F91248 )
        return 0;
LABEL_39:
      v19 = 57;
      v29 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
      v21 = *(__int64 **)(a1 + 24);
      v22 = 32 * (1 - v29);
      v23 = *(_QWORD *)(a2 - 32 * v29);
      return sub_11CC240(v23, *(_QWORD *)(a2 + v22), a3, v21, v19, v13);
    case ':':
      if ( v13 == dword_4F91088 )
        return 0;
      goto LABEL_41;
    case ';':
      if ( !(_BYTE)qword_4F91248 )
        return 0;
LABEL_41:
      v24 = 59;
      v30 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
      v26 = *(__int64 **)(a1 + 24);
      v27 = 32 * (1 - v30);
      v28 = *(_QWORD *)(a2 - 32 * v30);
      return sub_11CC520(v28, *(_QWORD *)(a2 + v27), a3, v26, v24, v13);
    case '<':
      if ( v13 == dword_4F91088 )
        return 0;
      return sub_11CC530(
               *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)),
               *(_QWORD *)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))),
               *(_QWORD *)(a2 + 32 * (2LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))),
               a3,
               *(__int64 **)(a1 + 24),
               0x3Du,
               v13);
    case '=':
      if ( !(_BYTE)qword_4F91248 )
        return 0;
      return sub_11CC530(
               *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)),
               *(_QWORD *)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))),
               *(_QWORD *)(a2 + 32 * (2LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))),
               a3,
               *(__int64 **)(a1 + 24),
               0x3Du,
               v13);
    case '>':
      v31 = v13;
      if ( (unsigned __int8)v13 == dword_4F91088 )
        return 0;
      v32 = *(__int64 **)(a1 + 24);
      v33 = *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
      return sub_11CB9C0(v33, a3, v32, 0x3Fu, v31);
    case '?':
      if ( !(_BYTE)qword_4F91248 )
        return 0;
      v31 = v13;
      v32 = *(__int64 **)(a1 + 24);
      v33 = *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
      return sub_11CB9C0(v33, a3, v32, 0x3Fu, v31);
    case '@':
      if ( v13 == dword_4F91088 )
        return 0;
      goto LABEL_54;
    case 'A':
      if ( !(_BYTE)qword_4F91248 )
        return 0;
LABEL_54:
      result = sub_11CBC90(
                 *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)),
                 *(_QWORD *)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))),
                 a3,
                 *(__int64 **)(a1 + 24),
                 0x41u,
                 v13);
      break;
    default:
      return 0;
  }
  return result;
}
