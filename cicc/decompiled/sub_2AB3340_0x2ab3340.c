// Function: sub_2AB3340
// Address: 0x2ab3340
//
__int64 __fastcall sub_2AB3340(__int64 a1, unsigned __int8 *a2, __int64 a3)
{
  unsigned __int8 *v3; // rbx
  __int64 v4; // r9
  int v5; // edx
  int v6; // edx
  __int64 v7; // rcx
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r12
  __int64 v11; // rax
  int v12; // r12d
  __int64 v13; // rax
  __int64 v14; // rdx
  int v15; // edx
  unsigned __int8 *v16; // r15
  char *v17; // rax
  __int64 v18; // rdx
  unsigned __int8 *v19; // r12
  __int64 v20; // rdx
  int v21; // ecx
  __int64 v22; // r8
  __int64 v23; // rdx
  __int64 v24; // r8
  __int64 v25; // rax
  __int64 v26; // r9
  __int64 *v27; // r12
  __int64 *v28; // rbx
  _QWORD *v29; // rax
  __int64 v30; // r8
  __int64 v31; // rdx
  unsigned __int64 v32; // r9
  __int64 v33; // rax
  char *v34; // r9
  __int64 v35; // r10
  __int64 v36; // r12
  bool v38; // zf
  __int128 v39; // [rsp-18h] [rbp-198h]
  _QWORD *v40; // [rsp+8h] [rbp-178h]
  _QWORD *v41; // [rsp+10h] [rbp-170h]
  __int64 v42; // [rsp+18h] [rbp-168h]
  __int64 v43; // [rsp+18h] [rbp-168h]
  __int64 v44; // [rsp+18h] [rbp-168h]
  int v45; // [rsp+20h] [rbp-160h]
  int v46; // [rsp+24h] [rbp-15Ch]
  char *v48; // [rsp+30h] [rbp-150h] BYREF
  __int64 v49; // [rsp+38h] [rbp-148h]
  _BYTE v50[48]; // [rsp+40h] [rbp-140h] BYREF
  char *v51; // [rsp+70h] [rbp-110h] BYREF
  __int64 v52; // [rsp+78h] [rbp-108h]
  _BYTE v53[48]; // [rsp+80h] [rbp-100h] BYREF
  _BYTE v54[24]; // [rsp+B0h] [rbp-D0h] BYREF
  char *v55; // [rsp+C8h] [rbp-B8h]
  char v56; // [rsp+D8h] [rbp-A8h] BYREF
  char *v57; // [rsp+F8h] [rbp-88h]
  char v58; // [rsp+108h] [rbp-78h] BYREF

  v3 = a2;
  v45 = sub_9B78C0((__int64)a2, *(__int64 **)(a1 + 456));
  v46 = 0;
  v41 = sub_2AAEE30(*((_QWORD *)a2 + 1), a3);
  if ( (unsigned __int8)sub_920620((__int64)a2) )
  {
    v5 = -1;
    if ( a2[1] >> 1 != 127 )
      v5 = a2[1] >> 1;
    v46 = v5;
  }
  v6 = *a2;
  if ( v6 == 40 )
  {
    v7 = -32 - 32LL * (unsigned int)sub_B491D0((__int64)a2);
  }
  else
  {
    v7 = -32;
    if ( v6 != 85 )
    {
      v7 = -96;
      if ( v6 != 34 )
        BUG();
    }
  }
  if ( (a2[7] & 0x80u) != 0 )
  {
    v42 = v7;
    v8 = sub_BD2BC0((__int64)a2);
    v7 = v42;
    v10 = v8 + v9;
    v11 = 0;
    if ( (a2[7] & 0x80u) != 0 )
    {
      v11 = sub_BD2BC0((__int64)a2);
      v7 = v42;
    }
    if ( (unsigned int)((v10 - v11) >> 4) )
    {
      v43 = v7;
      if ( (a2[7] & 0x80u) == 0 )
        BUG();
      v12 = *(_DWORD *)(sub_BD2BC0((__int64)a2) + 8);
      if ( (a2[7] & 0x80u) == 0 )
        BUG();
      v13 = sub_BD2BC0((__int64)a2);
      v7 = v43 - 32LL * (unsigned int)(*(_DWORD *)(v13 + v14 - 4) - v12);
    }
  }
  v15 = *((_DWORD *)a2 + 1);
  v49 = 0x600000000LL;
  v16 = &a2[v7];
  v17 = v50;
  v48 = v50;
  v18 = 32LL * (v15 & 0x7FFFFFF);
  v19 = &a2[-v18];
  v20 = v7 + v18;
  v21 = 0;
  v22 = v20 >> 5;
  if ( (unsigned __int64)v20 > 0xC0 )
  {
    v44 = v20 >> 5;
    sub_C8D5F0((__int64)&v48, v50, v20 >> 5, 8u, v22, v4);
    v21 = v49;
    LODWORD(v22) = v44;
    v17 = &v48[8 * (unsigned int)v49];
  }
  if ( v19 != v16 )
  {
    do
    {
      if ( v17 )
        *(_QWORD *)v17 = *(_QWORD *)v19;
      v19 += 32;
      v17 += 8;
    }
    while ( v16 != v19 );
    v21 = v49;
  }
  v23 = *((_QWORD *)a2 - 4);
  LODWORD(v49) = v22 + v21;
  v24 = (unsigned int)(v22 + v21);
  if ( !v23 || *(_BYTE *)v23 || (v25 = *(_QWORD *)(v23 + 24), v25 != *((_QWORD *)a2 + 10)) )
    BUG();
  v26 = *(_QWORD *)(v25 + 16);
  v51 = v53;
  v52 = 0x600000000LL;
  v27 = (__int64 *)(v26 + 8LL * *(unsigned int *)(v25 + 12));
  if ( (__int64 *)(v26 + 8) == v27 )
  {
    v35 = *(_QWORD *)(a1 + 456);
    v34 = v53;
    v33 = 0;
LABEL_30:
    if ( !*(_BYTE *)v23 && *(_QWORD *)(v23 + 24) == *((_QWORD *)v3 + 10) )
    {
      v38 = (*(_BYTE *)(v23 + 33) & 0x20) == 0;
      v23 = 0;
      if ( !v38 )
        v23 = (__int64)v3;
    }
    else
    {
      v23 = 0;
    }
    goto LABEL_33;
  }
  v28 = (__int64 *)(v26 + 8);
  do
  {
    v29 = sub_2AAEE30(*v28, a3);
    v31 = (unsigned int)v52;
    v32 = (unsigned int)v52 + 1LL;
    if ( v32 > HIDWORD(v52) )
    {
      v40 = v29;
      sub_C8D5F0((__int64)&v51, v53, (unsigned int)v52 + 1LL, 8u, v30, v32);
      v31 = (unsigned int)v52;
      v29 = v40;
    }
    ++v28;
    *(_QWORD *)&v51[8 * v31] = v29;
    v33 = (unsigned int)(v52 + 1);
    LODWORD(v52) = v52 + 1;
  }
  while ( v27 != v28 );
  v3 = a2;
  v24 = (unsigned int)v49;
  v34 = v51;
  v23 = *((_QWORD *)a2 - 4);
  v35 = *(_QWORD *)(a1 + 456);
  if ( v23 )
    goto LABEL_30;
LABEL_33:
  *((_QWORD *)&v39 + 1) = 1;
  *(_QWORD *)&v39 = 0;
  sub_DF8E30((__int64)v54, v45, (__int64)v41, v48, v24, v46, v34, v33, v23, v39, v35);
  v36 = sub_DFD690(*(_QWORD *)(a1 + 448), (__int64)v54);
  if ( v57 != &v58 )
    _libc_free((unsigned __int64)v57);
  if ( v55 != &v56 )
    _libc_free((unsigned __int64)v55);
  if ( v51 != v53 )
    _libc_free((unsigned __int64)v51);
  if ( v48 != v50 )
    _libc_free((unsigned __int64)v48);
  return v36;
}
