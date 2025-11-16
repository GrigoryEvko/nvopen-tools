// Function: sub_BD45C0
// Address: 0xbd45c0
//
unsigned __int8 *__fastcall sub_BD45C0(
        unsigned __int8 *a1,
        __int64 a2,
        __int64 a3,
        char a4,
        char a5,
        char a6,
        unsigned __int8 (__fastcall *a7)(__int64, __int64, __int64 *),
        __int64 a8)
{
  unsigned __int8 *v8; // r15
  __int64 v10; // rbx
  __int64 v11; // rcx
  unsigned __int8 v12; // dl
  unsigned int v14; // eax
  int i; // eax
  int v16; // eax
  unsigned __int8 *v17; // rdx
  char v18; // al
  char *v19; // rcx
  char v20; // al
  char v21; // al
  unsigned int v22; // r14d
  __int64 v23; // rdx
  unsigned int v24; // r8d
  unsigned __int64 v25; // rax
  unsigned int v26; // eax
  bool v27; // zf
  unsigned int v28; // eax
  unsigned __int8 **v29; // rax
  unsigned __int8 **v30; // rdx
  __int16 v31; // ax
  unsigned __int8 *v32; // r15
  char v33; // al
  char v34; // dl
  int v35; // eax
  unsigned __int64 v36; // rax
  __int64 v37; // rdx
  char v38; // al
  _BYTE *v39; // rax
  __int64 v40; // rdx
  unsigned int v41; // [rsp+4h] [rbp-DCh]
  unsigned int v42; // [rsp+10h] [rbp-D0h]
  char *v46; // [rsp+18h] [rbp-C8h]
  __int64 v47; // [rsp+18h] [rbp-C8h]
  __int64 v48; // [rsp+18h] [rbp-C8h]
  char v49; // [rsp+2Fh] [rbp-B1h] BYREF
  unsigned __int64 v50; // [rsp+30h] [rbp-B0h] BYREF
  unsigned int v51; // [rsp+38h] [rbp-A8h]
  __int64 v52; // [rsp+40h] [rbp-A0h] BYREF
  unsigned int v53; // [rsp+48h] [rbp-98h]
  __int64 v54; // [rsp+50h] [rbp-90h] BYREF
  unsigned int v55; // [rsp+58h] [rbp-88h]
  __int64 v56; // [rsp+60h] [rbp-80h] BYREF
  unsigned int v57; // [rsp+68h] [rbp-78h]
  __int64 v58; // [rsp+70h] [rbp-70h] BYREF
  unsigned __int8 **v59; // [rsp+78h] [rbp-68h]
  __int64 v60; // [rsp+80h] [rbp-60h]
  int v61; // [rsp+88h] [rbp-58h]
  char v62; // [rsp+8Ch] [rbp-54h]
  unsigned __int8 *v63; // [rsp+90h] [rbp-50h] BYREF

  v8 = a1;
  v10 = a2;
  v11 = *((_QWORD *)a1 + 1);
  v12 = *(_BYTE *)(v11 + 8);
  if ( (unsigned int)v12 - 17 <= 1 )
    v12 = *(_BYTE *)(**(_QWORD **)(v11 + 16) + 8LL);
  if ( v12 != 14 )
    return v8;
  v14 = *(_DWORD *)(a3 + 8);
  v61 = 0;
  v62 = 1;
  v42 = v14;
  v59 = &v63;
  v60 = 0x100000004LL;
  v63 = a1;
  v58 = 1;
LABEL_6:
  for ( i = *v8; (unsigned __int8)i <= 0x1Cu; i = *v8 )
  {
    if ( (_BYTE)i == 5 )
    {
      v31 = *((_WORD *)v8 + 1);
      if ( v31 == 34 )
        goto LABEL_23;
      if ( (unsigned __int16)(v31 - 49) > 1u )
      {
        if ( !a4 || !a6 )
          goto LABEL_86;
        v16 = *((unsigned __int16 *)v8 + 1);
        goto LABEL_13;
      }
LABEL_59:
      if ( (v8[7] & 0x40) != 0 )
        v32 = (unsigned __int8 *)*((_QWORD *)v8 - 1);
      else
        v32 = &v8[-32 * (*((_DWORD *)v8 + 1) & 0x7FFFFFF)];
      v8 = *(unsigned __int8 **)v32;
      if ( !v62 )
        goto LABEL_62;
    }
    else
    {
      if ( (_BYTE)i == 1 && !(unsigned __int8)sub_B2F6B0((__int64)v8) )
        v8 = (unsigned __int8 *)*((_QWORD *)v8 - 4);
LABEL_50:
      if ( !v62 )
      {
LABEL_62:
        a2 = (__int64)v8;
        sub_C8CC70(&v58, v8);
        v33 = v62;
        if ( !v34 )
          goto LABEL_87;
        goto LABEL_6;
      }
    }
    v29 = v59;
    v30 = &v59[HIDWORD(v60)];
    if ( v59 != v30 )
    {
      while ( v8 != *v29 )
      {
        if ( v30 == ++v29 )
          goto LABEL_54;
      }
      return v8;
    }
LABEL_54:
    if ( HIDWORD(v60) >= (unsigned int)v60 )
      goto LABEL_62;
    ++HIDWORD(v60);
    *v30 = v8;
    ++v58;
  }
  if ( (_BYTE)i != 63 )
  {
    if ( (unsigned __int8)(i - 78) <= 1u )
      goto LABEL_59;
    if ( (unsigned __int8)(i - 34) <= 0x33u )
    {
      a2 = 0x8000000000041LL;
      if ( _bittest64(&a2, (unsigned int)(i - 34)) )
      {
        a2 = 52;
        v37 = sub_B494D0((__int64)v8, 52);
        if ( !v37 )
          v37 = (__int64)v8;
        if ( a5 && (v47 = v37, v38 = sub_B46A50((__int64)v8), v37 = v47, v38) )
          v8 = *(unsigned __int8 **)&v8[-32 * (*((_DWORD *)v8 + 1) & 0x7FFFFFF)];
        else
          v8 = (unsigned __int8 *)v37;
        goto LABEL_50;
      }
    }
    if ( !a4 || !a6 )
      goto LABEL_86;
    v16 = i - 29;
LABEL_13:
    if ( v16 != 48 )
      goto LABEL_86;
    v17 = (v8[7] & 0x40) != 0 ? (unsigned __int8 *)*((_QWORD *)v8 - 1) : &v8[-32 * (*((_DWORD *)v8 + 1) & 0x7FFFFFF)];
    v46 = *(char **)v17;
    if ( (unsigned int)sub_BCB060(*(_QWORD *)(*(_QWORD *)v17 + 8LL)) != v42 )
      goto LABEL_86;
    v18 = *v46;
    if ( (unsigned __int8)*v46 <= 0x1Cu )
    {
      if ( v18 != 5 || *((_WORD *)v46 + 1) != 13 )
        goto LABEL_86;
    }
    else if ( v18 != 42 )
    {
      goto LABEL_86;
    }
    v19 = (char *)*((_QWORD *)v46 - 8);
    v20 = *v19;
    if ( (unsigned __int8)*v19 <= 0x1Cu )
    {
      if ( v20 != 5 || *((_WORD *)v19 + 1) != 47 )
      {
LABEL_21:
        if ( !*((_QWORD *)v46 - 4) )
LABEL_22:
          BUG();
        goto LABEL_86;
      }
    }
    else if ( v20 != 76 )
    {
      goto LABEL_21;
    }
    v39 = (_BYTE *)*((_QWORD *)v46 - 4);
    if ( !v39 )
      goto LABEL_22;
    if ( *v39 != 17 )
      goto LABEL_86;
    a2 = (__int64)(v39 + 24);
    v48 = *((_QWORD *)v46 - 8);
    sub_C45EE0(a3, v39 + 24);
    v8 = *(unsigned __int8 **)(v48 - 32);
    goto LABEL_50;
  }
LABEL_23:
  if ( !a4 && (v8[1] & 2) == 0 )
    goto LABEL_86;
  v51 = sub_AE43F0(v10, *((_QWORD *)v8 + 1));
  if ( v51 > 0x40 )
    sub_C43690(&v50, 0, 0);
  else
    v50 = 0;
  v21 = sub_BB6360((__int64)v8, v10, (__int64)&v50, a7, a8);
  a2 = v51;
  if ( !v21 )
    goto LABEL_83;
  v22 = v51 + 1;
  v23 = 1LL << ((unsigned __int8)v51 - 1);
  if ( v51 > 0x40 )
  {
    v41 = v51;
    v35 = (*(_QWORD *)(v50 + 8LL * ((v51 - 1) >> 6)) & v23) != 0 ? sub_C44500(&v50) : sub_C444A0(&v50);
    a2 = v41;
    v24 = v22 - v35;
  }
  else if ( (v23 & v50) != 0 )
  {
    if ( v51 )
    {
      v24 = v51 - 63;
      if ( v50 << (64 - (unsigned __int8)v51) != -1 )
      {
        _BitScanReverse64(&v25, ~(v50 << (64 - (unsigned __int8)v51)));
        v24 = v22 - (v25 ^ 0x3F);
      }
    }
    else
    {
      v24 = 1;
    }
  }
  else
  {
    v24 = 1;
    if ( v50 )
    {
      _BitScanReverse64(&v36, v50);
      v24 = 65 - (v36 ^ 0x3F);
    }
  }
  if ( v24 > v42 )
    goto LABEL_83;
  sub_C44B10(&v52, &v50, v42);
  if ( !a7 )
  {
    a2 = (__int64)&v52;
    sub_C45EE0(a3, &v52);
LABEL_44:
    v8 = *(unsigned __int8 **)&v8[-32 * (*((_DWORD *)v8 + 1) & 0x7FFFFFF)];
    if ( v53 > 0x40 && v52 )
      j_j___libc_free_0_0(v52);
    if ( v51 > 0x40 && v50 )
      j_j___libc_free_0_0(v50);
    goto LABEL_50;
  }
  v26 = *(_DWORD *)(a3 + 8);
  v49 = 0;
  v55 = v26;
  if ( v26 > 0x40 )
    sub_C43780(&v54, a3);
  else
    v54 = *(_QWORD *)a3;
  a2 = a3;
  sub_C45F70(&v56, a3, &v52, &v49);
  if ( *(_DWORD *)(a3 + 8) > 0x40u && *(_QWORD *)a3 )
    j_j___libc_free_0_0(*(_QWORD *)a3);
  v27 = v49 == 0;
  *(_QWORD *)a3 = v56;
  v28 = v57;
  *(_DWORD *)(a3 + 8) = v57;
  if ( v27 )
  {
    if ( v55 > 0x40 && v54 )
      j_j___libc_free_0_0(v54);
    goto LABEL_44;
  }
  if ( v28 <= 0x40 && v55 <= 0x40 )
  {
    v40 = v54;
    *(_DWORD *)(a3 + 8) = v55;
    *(_QWORD *)a3 = v40;
  }
  else
  {
    sub_C43990(a3, &v54);
    if ( v55 > 0x40 && v54 )
      j_j___libc_free_0_0(v54);
  }
  if ( v53 > 0x40 && v52 )
    j_j___libc_free_0_0(v52);
  a2 = v51;
LABEL_83:
  if ( (unsigned int)a2 > 0x40 && v50 )
    j_j___libc_free_0_0(v50);
LABEL_86:
  v33 = v62;
LABEL_87:
  if ( !v33 )
    _libc_free(v59, a2);
  return v8;
}
