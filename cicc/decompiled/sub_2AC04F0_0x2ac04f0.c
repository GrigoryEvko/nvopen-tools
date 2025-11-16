// Function: sub_2AC04F0
// Address: 0x2ac04f0
//
unsigned __int64 __fastcall sub_2AC04F0(
        __int64 a1,
        __int64 a2,
        unsigned __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6)
{
  unsigned __int64 v6; // rbx
  unsigned __int64 v7; // r12
  int v9; // r13d
  __int64 v11; // rdi
  __int64 v12; // r11
  char v13; // al
  __int64 *v14; // r10
  __int64 *v15; // r14
  unsigned __int64 v16; // rdx
  __int64 v17; // r10
  __int64 v18; // rax
  int v19; // edx
  bool v20; // zf
  int v21; // edx
  bool v22; // of
  __int64 *v23; // r11
  char v24; // al
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // r15
  int v28; // r15d
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rdx
  __int64 v32; // rdx
  _BYTE *v33; // r8
  _BYTE *v34; // r14
  _BYTE *v35; // r15
  _QWORD *v36; // rax
  __int64 v37; // r8
  __int64 v38; // rdx
  unsigned __int64 v39; // r9
  __int64 v40; // rax
  __int64 v41; // [rsp+8h] [rbp-108h]
  char v42; // [rsp+18h] [rbp-F8h]
  _QWORD *v43; // [rsp+18h] [rbp-F8h]
  __int64 v44; // [rsp+20h] [rbp-F0h]
  __int64 *v45; // [rsp+30h] [rbp-E0h]
  unsigned __int64 v46; // [rsp+30h] [rbp-E0h]
  unsigned __int64 v47; // [rsp+38h] [rbp-D8h]
  __int64 *v48; // [rsp+40h] [rbp-D0h]
  __int64 v49; // [rsp+40h] [rbp-D0h]
  __int64 v50; // [rsp+48h] [rbp-C8h]
  int v51; // [rsp+50h] [rbp-C0h]
  __int64 v52; // [rsp+50h] [rbp-C0h]
  __int64 v53; // [rsp+58h] [rbp-B8h]
  __int64 v54; // [rsp+68h] [rbp-A8h] BYREF
  _BYTE *v55; // [rsp+70h] [rbp-A0h] BYREF
  int v56; // [rsp+78h] [rbp-98h]
  _BYTE v57[32]; // [rsp+80h] [rbp-90h] BYREF
  _BYTE *v58; // [rsp+A0h] [rbp-70h] BYREF
  __int64 v59; // [rsp+A8h] [rbp-68h]
  _BYTE v60[96]; // [rsp+B0h] [rbp-60h] BYREF

  v6 = HIDWORD(a3);
  v53 = a3;
  if ( BYTE4(a3) )
    return 0;
  v9 = a3;
  if ( (_DWORD)a3 == 1 )
    return 0;
  v11 = *(_QWORD *)(a2 + 8);
  v58 = (_BYTE *)a3;
  if ( *(_BYTE *)(v11 + 8) == 15 )
  {
    v12 = (__int64)sub_E454C0(v11, a3, a3, a4, a5, a6);
    v13 = *(_BYTE *)(v12 + 8);
    v54 = v12;
    if ( v13 == 7 )
      goto LABEL_26;
  }
  else
  {
    v12 = sub_2AAEDF0(v11, a3);
    v13 = *(_BYTE *)(v12 + 8);
    v54 = v12;
    if ( v13 == 7 )
      goto LABEL_26;
  }
  if ( *(_BYTE *)a2 == 61 )
  {
    v52 = v12;
    if ( (unsigned __int8)sub_DFAB30(*(_QWORD *)(a1 + 448)) )
      goto LABEL_26;
    v12 = v52;
    v13 = *(_BYTE *)(v52 + 8);
  }
  if ( v13 != 15 )
  {
    v14 = &v54;
    v48 = (__int64 *)&v55;
    goto LABEL_11;
  }
  v14 = *(__int64 **)(v12 + 16);
  v48 = &v14[*(unsigned int *)(v12 + 12)];
  if ( v48 == v14 )
  {
LABEL_26:
    v46 = 0;
    goto LABEL_28;
  }
  v12 = *v14;
LABEL_11:
  v15 = v14 + 1;
  v16 = 0;
  v17 = v12;
  v42 = v6;
  v51 = 0;
  if ( (_DWORD)v53 )
    v16 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v53;
  v6 = 0;
  v47 = v16;
  while ( 1 )
  {
    LODWORD(v59) = v53;
    v23 = *(__int64 **)(a1 + 448);
    if ( (unsigned int)v53 <= 0x40 )
    {
      v58 = (_BYTE *)v47;
    }
    else
    {
      v44 = v17;
      v45 = *(__int64 **)(a1 + 448);
      sub_C43690((__int64)&v58, -1, 1);
      v17 = v44;
      v23 = v45;
    }
    v18 = sub_DFAAD0(v23, v17, (__int64)&v58, 1u, 0);
    v20 = v19 == 1;
    v21 = 1;
    if ( !v20 )
      v21 = v51;
    v22 = __OFADD__(v18, v6);
    v6 += v18;
    v51 = v21;
    if ( v22 )
    {
      v6 = 0x8000000000000000LL;
      if ( v18 > 0 )
        v6 = 0x7FFFFFFFFFFFFFFFLL;
    }
    if ( (unsigned int)v59 > 0x40 && v58 )
      j_j___libc_free_0_0((unsigned __int64)v58);
    if ( v48 == v15 )
      break;
    v17 = *v15++;
  }
  v46 = v6;
  LOBYTE(v6) = v42;
LABEL_28:
  v24 = *(_BYTE *)a2;
  if ( *(_BYTE *)a2 == 61 )
  {
    if ( !(unsigned __int8)sub_DFA780(*(_QWORD *)(a1 + 448)) )
      return v46;
    v24 = *(_BYTE *)a2;
  }
  if ( v24 != 62 )
    goto LABEL_30;
  if ( (unsigned __int8)sub_DFAB30(*(_QWORD *)(a1 + 448)) )
    return v46;
  v24 = *(_BYTE *)a2;
LABEL_30:
  if ( v24 == 85 )
  {
    if ( *(char *)(a2 + 7) < 0 )
    {
      v25 = sub_BD2BC0(a2);
      v27 = v25 + v26;
      if ( *(char *)(a2 + 7) >= 0 )
      {
        if ( (unsigned int)(v27 >> 4) )
          goto LABEL_71;
      }
      else if ( (unsigned int)((v27 - sub_BD2BC0(a2)) >> 4) )
      {
        if ( *(char *)(a2 + 7) < 0 )
        {
          v28 = *(_DWORD *)(sub_BD2BC0(a2) + 8);
          if ( *(char *)(a2 + 7) >= 0 )
            BUG();
          v29 = sub_BD2BC0(a2);
          v31 = -32 - 32LL * (unsigned int)(*(_DWORD *)(v29 + v30 - 4) - v28);
          goto LABEL_54;
        }
LABEL_71:
        BUG();
      }
    }
    v31 = -32;
LABEL_54:
    v50 = a2 + v31;
    v49 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
    goto LABEL_39;
  }
  v49 = sub_1168D40(a2);
  v50 = v32;
LABEL_39:
  v58 = v60;
  v59 = 0x600000000LL;
  sub_2ABFD70((__int64)&v55, a1, v49, v50, v53);
  v33 = v55;
  v34 = &v55[8 * v56];
  if ( v34 != v55 )
  {
    v35 = v55;
    do
    {
      LODWORD(v53) = v9;
      BYTE4(v53) = v6;
      v36 = sub_2AAEE30(*(_QWORD *)(*(_QWORD *)v35 + 8LL), v53);
      v38 = (unsigned int)v59;
      v39 = (unsigned int)v59 + 1LL;
      if ( v39 > HIDWORD(v59) )
      {
        v43 = v36;
        sub_C8D5F0((__int64)&v58, v60, (unsigned int)v59 + 1LL, 8u, v37, v39);
        v38 = (unsigned int)v59;
        v36 = v43;
      }
      v35 += 8;
      *(_QWORD *)&v58[8 * v38] = v36;
      LODWORD(v59) = v59 + 1;
    }
    while ( v34 != v35 );
    v33 = v55;
  }
  if ( v33 != v57 )
    _libc_free((unsigned __int64)v33);
  BYTE4(v53) = v6;
  LODWORD(v53) = v9;
  v41 = *(_QWORD *)(a1 + 448);
  sub_2ABFD70((__int64)&v55, a1, v49, v50, v53);
  v40 = sub_DFAB00(v41);
  v7 = v40 + v46;
  if ( __OFADD__(v40, v46) )
  {
    v7 = 0x7FFFFFFFFFFFFFFFLL;
    if ( v40 <= 0 )
      v7 = 0x8000000000000000LL;
  }
  if ( v55 != v57 )
    _libc_free((unsigned __int64)v55);
  if ( v58 != v60 )
    _libc_free((unsigned __int64)v58);
  return v7;
}
