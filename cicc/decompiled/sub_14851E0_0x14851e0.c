// Function: sub_14851E0
// Address: 0x14851e0
//
__int64 __fastcall sub_14851E0(_QWORD *a1, __int64 a2, __int64 a3, __m128i a4, __m128i a5)
{
  __int64 v5; // r15
  __int64 v6; // r14
  _BYTE *v8; // rsi
  __int64 v9; // rbx
  __int64 v10; // rsi
  unsigned int v11; // eax
  __int64 *v12; // rdi
  __int64 v13; // rdx
  __int64 v14; // rsi
  __int64 v15; // rsi
  __int64 v16; // rdx
  __int64 *v17; // rdi
  __int64 v18; // rax
  unsigned __int64 v19; // rsi
  unsigned int v20; // r10d
  bool v21; // cf
  __int64 *v22; // rax
  unsigned int v23; // edx
  unsigned __int64 v24; // rax
  unsigned int v25; // ecx
  __int64 v26; // rax
  __int64 v27; // rax
  _BYTE *v28; // rsi
  __int64 v29; // rbx
  _BYTE *i; // rdx
  __int64 v31; // rax
  __int64 *v32; // rdi
  __int64 v33; // r12
  unsigned __int64 v35; // rsi
  __int64 v36; // rax
  __int64 v37; // rax
  int v38; // eax
  __int64 v39; // rsi
  unsigned int v40; // eax
  __int64 v41; // [rsp+8h] [rbp-A8h]
  unsigned int v42; // [rsp+8h] [rbp-A8h]
  unsigned __int64 v43; // [rsp+20h] [rbp-90h] BYREF
  unsigned int v44; // [rsp+28h] [rbp-88h]
  __int64 *v45; // [rsp+30h] [rbp-80h] BYREF
  unsigned int v46; // [rsp+38h] [rbp-78h]
  __int64 *v47; // [rsp+40h] [rbp-70h] BYREF
  unsigned int v48; // [rsp+48h] [rbp-68h]
  __int64 v49; // [rsp+50h] [rbp-60h] BYREF
  unsigned int v50; // [rsp+58h] [rbp-58h]
  __int64 *v51; // [rsp+60h] [rbp-50h] BYREF
  __int64 v52; // [rsp+68h] [rbp-48h]
  _BYTE v53[64]; // [rsp+70h] [rbp-40h] BYREF

  v5 = a2;
  v6 = a3;
  if ( *(_WORD *)(a3 + 24) )
    goto LABEL_36;
  v8 = *(_BYTE **)(a2 + 32);
  v9 = *(_QWORD *)v8;
  if ( *(_WORD *)(*(_QWORD *)v8 + 24LL) )
    goto LABEL_36;
  if ( a3 == v9 )
  {
    v52 = 0x200000000LL;
    v36 = *(_QWORD *)(v5 + 40);
    v51 = (__int64 *)v53;
    sub_145C5B0((__int64)&v51, v8 + 8, &v8[8 * v36]);
    v37 = sub_147EE30(a1, &v51, 0, 0, a4, a5);
    v32 = v51;
    v33 = v37;
    if ( v51 == (__int64 *)v53 )
      return v33;
    goto LABEL_42;
  }
  v10 = *(_QWORD *)(v9 + 32);
  v11 = *(_DWORD *)(v10 + 32);
  v12 = *(__int64 **)(v10 + 24);
  v13 = 1LL << ((unsigned __int8)v11 - 1);
  if ( v11 > 0x40 )
  {
    v14 = v10 + 24;
    if ( (v12[(v11 - 1) >> 6] & v13) == 0 )
    {
      v46 = v11;
      sub_16A4FD0(&v45, v14);
      goto LABEL_7;
    }
    LODWORD(v52) = v11;
    sub_16A4FD0(&v51, v14);
    LOBYTE(v11) = v52;
    if ( (unsigned int)v52 > 0x40 )
    {
      sub_16A8F40(&v51);
      goto LABEL_48;
    }
    v35 = (unsigned __int64)v51;
LABEL_47:
    v51 = (__int64 *)(~v35 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v11));
LABEL_48:
    sub_16A7400(&v51);
    v46 = v52;
    v45 = v51;
    goto LABEL_7;
  }
  v35 = *(_QWORD *)(v10 + 24);
  if ( (v13 & (unsigned __int64)v12) != 0 )
  {
    LODWORD(v52) = v11;
    goto LABEL_47;
  }
  v46 = v11;
  v45 = v12;
LABEL_7:
  v15 = *(_QWORD *)(v6 + 32);
  v16 = *(unsigned int *)(v15 + 32);
  v17 = *(__int64 **)(v15 + 24);
  v18 = 1LL << ((unsigned __int8)v16 - 1);
  if ( (unsigned int)v16 > 0x40 )
  {
    v39 = v15 + 24;
    if ( (v17[(unsigned int)(v16 - 1) >> 6] & v18) == 0 )
    {
      v48 = v16;
      sub_16A4FD0(&v47, v39);
      v16 = v48;
LABEL_10:
      v20 = v46;
      v21 = v46 < (unsigned int)v16;
      if ( v46 <= (unsigned int)v16 )
        goto LABEL_11;
LABEL_56:
      sub_16A5C50(&v51, &v47, v20);
      if ( v48 > 0x40 && v47 )
        j_j___libc_free_0_0(v47);
      v22 = v51;
      LODWORD(v16) = v52;
      v20 = v46;
      v47 = v51;
      goto LABEL_13;
    }
    LODWORD(v52) = v16;
    sub_16A4FD0(&v51, v39);
    LOBYTE(v16) = v52;
    if ( (unsigned int)v52 > 0x40 )
    {
      sub_16A8F40(&v51);
      goto LABEL_55;
    }
    v19 = (unsigned __int64)v51;
  }
  else
  {
    v19 = *(_QWORD *)(v15 + 24);
    if ( (v18 & (unsigned __int64)v17) == 0 )
    {
      v48 = v16;
      v47 = v17;
      goto LABEL_10;
    }
    LODWORD(v52) = v16;
  }
  v51 = (__int64 *)(~v19 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v16));
LABEL_55:
  sub_16A7400(&v51);
  v16 = (unsigned int)v52;
  v20 = v46;
  v48 = v52;
  v47 = v51;
  v21 = v46 < (unsigned int)v52;
  if ( v46 > (unsigned int)v52 )
    goto LABEL_56;
LABEL_11:
  if ( v21 )
  {
    sub_16A5C50(&v51, &v45, v16);
    if ( v46 > 0x40 && v45 )
      j_j___libc_free_0_0(v45);
    v45 = v51;
    v40 = v52;
    LODWORD(v52) = 0;
    v46 = v40;
    sub_135E100((__int64 *)&v51);
    LODWORD(v16) = v48;
    v22 = v47;
    v20 = v46;
  }
  else
  {
    v22 = v47;
  }
LABEL_13:
  v49 = (__int64)v22;
  v50 = v16;
  v51 = v45;
  v48 = 0;
  LODWORD(v52) = v20;
  v46 = 0;
  sub_16A9A30(&v43, &v51, &v49);
  if ( (unsigned int)v52 > 0x40 && v51 )
    j_j___libc_free_0_0(v51);
  if ( v50 > 0x40 && v49 )
    j_j___libc_free_0_0(v49);
  if ( v48 > 0x40 && v47 )
    j_j___libc_free_0_0(v47);
  if ( v46 > 0x40 && v45 )
    j_j___libc_free_0_0(v45);
  v23 = v44;
  if ( v44 > 0x40 )
  {
    v42 = v44;
    v38 = sub_16A57B0(&v43);
    v23 = v42;
    v25 = v42 - v38;
  }
  else
  {
    if ( !v43 )
      goto LABEL_36;
    _BitScanReverse64(&v24, v43);
    v25 = 64 - (v24 ^ 0x3F);
  }
  if ( v25 > 1 )
  {
    sub_16A9D70(&v51, *(_QWORD *)(v9 + 32) + 24LL, &v43);
    v41 = sub_145CF40((__int64)a1, (__int64)&v51);
    sub_135E100((__int64 *)&v51);
    sub_16A9D70(&v51, *(_QWORD *)(v6 + 32) + 24LL, &v43);
    v6 = sub_145CF40((__int64)a1, (__int64)&v51);
    sub_135E100((__int64 *)&v51);
    v51 = (__int64 *)v53;
    v49 = v41;
    v52 = 0x200000000LL;
    sub_1458920((__int64)&v51, &v49);
    sub_145C5B0(
      (__int64)&v51,
      (_BYTE *)(*(_QWORD *)(v5 + 32) + 8LL),
      (_BYTE *)(*(_QWORD *)(v5 + 32) + 8LL * *(_QWORD *)(v5 + 40)));
    v26 = sub_147EE30(a1, &v51, 0, 0, a4, a5);
    v5 = v26;
    if ( *(_WORD *)(v26 + 24) != 5 )
    {
      v33 = sub_14857A0(a1, v26, v6);
      if ( v51 != (__int64 *)v53 )
        _libc_free((unsigned __int64)v51);
      sub_135E100((__int64 *)&v43);
      return v33;
    }
    if ( v51 != (__int64 *)v53 )
      _libc_free((unsigned __int64)v51);
    v23 = v44;
  }
  if ( v23 > 0x40 )
  {
    if ( v43 )
      j_j___libc_free_0_0(v43);
  }
LABEL_36:
  v27 = *(_QWORD *)(v5 + 40);
  if ( (_DWORD)v27 )
  {
    v28 = *(_BYTE **)(v5 + 32);
    v29 = 0;
    for ( i = v28; *(_QWORD *)i != v6; i += 8 )
    {
      if ( v29 == (_DWORD)v27 - 1 )
        return sub_1483CF0(a1, v5, v6, a4, a5);
      ++v29;
    }
    v52 = 0x200000000LL;
    v51 = (__int64 *)v53;
    sub_145C5B0((__int64)&v51, v28, i);
    sub_145C5B0(
      (__int64)&v51,
      (_BYTE *)(*(_QWORD *)(v5 + 32) + 8 * v29 + 8),
      (_BYTE *)(*(_QWORD *)(v5 + 32) + 8LL * *(_QWORD *)(v5 + 40)));
    v31 = sub_147EE30(a1, &v51, 0, 0, a4, a5);
    v32 = v51;
    v33 = v31;
    if ( v51 == (__int64 *)v53 )
      return v33;
LABEL_42:
    _libc_free((unsigned __int64)v32);
    return v33;
  }
  return sub_1483CF0(a1, v5, v6, a4, a5);
}
