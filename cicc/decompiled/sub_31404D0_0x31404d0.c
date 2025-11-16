// Function: sub_31404D0
// Address: 0x31404d0
//
__int64 __fastcall sub_31404D0(__int64 a1, __int64 *a2)
{
  __m128i v2; // rax
  _BYTE *v3; // rdi
  __int64 v4; // r13
  _BYTE *v5; // rbx
  int v6; // r12d
  _QWORD *v7; // r13
  size_t v8; // r14
  int v9; // r15d
  __int64 v10; // r8
  __int64 v11; // r12
  int v12; // eax
  int v13; // ecx
  int v14; // r10d
  __int64 v15; // r9
  unsigned int i; // r15d
  const void *v17; // rsi
  bool v18; // al
  unsigned int v19; // r15d
  int v20; // r12d
  int v21; // eax
  int v22; // r10d
  unsigned int j; // r9d
  __int64 v24; // r15
  const void *v25; // rcx
  bool v26; // al
  unsigned int v27; // r9d
  int v28; // eax
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rdi
  __int64 v32; // rsi
  int v34; // eax
  int v35; // eax
  int v36; // r15d
  __int64 v37; // r12
  int v38; // eax
  int v39; // ecx
  int v40; // r10d
  unsigned int v41; // r15d
  const void *v42; // rsi
  bool v43; // al
  unsigned int v44; // r15d
  int v45; // eax
  const void *v46; // [rsp+8h] [rbp-138h]
  __int64 v47; // [rsp+18h] [rbp-128h]
  int v48; // [rsp+24h] [rbp-11Ch]
  int v49; // [rsp+24h] [rbp-11Ch]
  int v50; // [rsp+24h] [rbp-11Ch]
  unsigned int v51; // [rsp+28h] [rbp-118h]
  __int64 v52; // [rsp+28h] [rbp-118h]
  __int64 v53; // [rsp+28h] [rbp-118h]
  __int64 v55; // [rsp+40h] [rbp-100h]
  int v56; // [rsp+40h] [rbp-100h]
  int v57; // [rsp+40h] [rbp-100h]
  _BYTE *v58; // [rsp+48h] [rbp-F8h]
  __m128i v59; // [rsp+50h] [rbp-F0h] BYREF
  __int64 v60; // [rsp+60h] [rbp-E0h] BYREF
  __int64 v61; // [rsp+68h] [rbp-D8h]
  __int64 v62; // [rsp+70h] [rbp-D0h]
  __int64 v63; // [rsp+78h] [rbp-C8h]
  _BYTE *v64; // [rsp+80h] [rbp-C0h] BYREF
  __int64 v65; // [rsp+88h] [rbp-B8h]
  _BYTE v66[176]; // [rsp+90h] [rbp-B0h] BYREF

  if ( !*a2 )
  {
    *(_QWORD *)a1 = 0;
    *(_QWORD *)(a1 + 8) = 0;
    *(_QWORD *)(a1 + 16) = 0;
    *(_DWORD *)(a1 + 24) = 0;
    return a1;
  }
  v60 = 0;
  v64 = v66;
  v61 = 0;
  v62 = 0;
  v63 = 0;
  v65 = 0x800000000LL;
  v2.m128i_i64[0] = sub_A72240(a2);
  v59 = v2;
  sub_C937F0(&v59, (__int64)&v64, ",", 1u, 0xFFFFFFFFLL, 1);
  v3 = v64;
  v4 = 16LL * (unsigned int)v65;
  v58 = &v64[v4];
  if ( &v64[v4] == v64 )
    goto LABEL_25;
  v5 = v64;
  do
  {
    v6 = v63;
    v7 = *(_QWORD **)v5;
    v8 = *((_QWORD *)v5 + 1);
    if ( !(_DWORD)v63 )
    {
      ++v60;
LABEL_6:
      sub_BA8070((__int64)&v60, 2 * v6);
      v9 = v63;
      v10 = 0;
      if ( !(_DWORD)v63 )
        goto LABEL_42;
      v11 = v61;
      v12 = sub_C94890(v7, v8);
      v13 = v9 - 1;
      v14 = 1;
      v15 = 0;
      for ( i = (v9 - 1) & v12; ; i = v13 & v19 )
      {
        v10 = v11 + 16LL * i;
        v17 = *(const void **)v10;
        if ( *(_QWORD *)v10 == -1 )
          goto LABEL_58;
        v18 = (_QWORD *)((char *)v7 + 2) == 0;
        if ( v17 != (const void *)-2LL )
        {
          if ( v8 != *(_QWORD *)(v10 + 8) )
            goto LABEL_11;
          v49 = v14;
          v52 = v15;
          v56 = v13;
          if ( !v8 )
            goto LABEL_42;
          v35 = memcmp(v7, v17, v8);
          v10 = v11 + 16LL * i;
          v13 = v56;
          v15 = v52;
          v14 = v49;
          v18 = v35 == 0;
        }
        if ( v18 )
          goto LABEL_42;
        if ( !v15 && v17 == (const void *)-2LL )
          v15 = v10;
LABEL_11:
        v19 = v14 + i;
        ++v14;
      }
    }
    v20 = v63 - 1;
    v55 = v61;
    v21 = sub_C94890(*(_QWORD **)v5, *((_QWORD *)v5 + 1));
    v22 = 1;
    v10 = 0;
    for ( j = v20 & v21; ; j = v20 & v27 )
    {
      v24 = v55 + 16LL * j;
      v25 = *(const void **)v24;
      v26 = (_QWORD *)((char *)v7 + 1) == 0;
      if ( *(_QWORD *)v24 != -1 )
      {
        v26 = (_QWORD *)((char *)v7 + 2) == 0;
        if ( v25 != (const void *)-2LL )
        {
          if ( v8 != *(_QWORD *)(v24 + 8) )
            goto LABEL_16;
          v47 = v10;
          v48 = v22;
          v51 = j;
          if ( !v8 )
            goto LABEL_23;
          v46 = *(const void **)v24;
          v28 = memcmp(v7, v25, v8);
          v25 = v46;
          j = v51;
          v22 = v48;
          v10 = v47;
          v26 = v28 == 0;
        }
      }
      if ( v26 )
        goto LABEL_23;
      if ( v25 == (const void *)-1LL )
        break;
LABEL_16:
      if ( v25 == (const void *)-2LL && !v10 )
        v10 = v24;
      v27 = v22 + j;
      ++v22;
    }
    v6 = v63;
    if ( !v10 )
      v10 = v24;
    ++v60;
    v34 = v62 + 1;
    if ( 4 * ((int)v62 + 1) >= (unsigned int)(3 * v63) )
      goto LABEL_6;
    if ( (int)v63 - (v34 + HIDWORD(v62)) > (unsigned int)v63 >> 3 )
      goto LABEL_36;
    sub_BA8070((__int64)&v60, v63);
    v36 = v63;
    v10 = 0;
    if ( !(_DWORD)v63 )
      goto LABEL_42;
    v37 = v61;
    v38 = sub_C94890(v7, v8);
    v39 = v36 - 1;
    v40 = 1;
    v15 = 0;
    v41 = (v36 - 1) & v38;
    while ( 2 )
    {
      v10 = v37 + 16LL * v41;
      v42 = *(const void **)v10;
      if ( *(_QWORD *)v10 != -1 )
      {
        v43 = (_QWORD *)((char *)v7 + 2) == 0;
        if ( v42 != (const void *)-2LL )
        {
          if ( v8 != *(_QWORD *)(v10 + 8) )
          {
LABEL_48:
            if ( v42 != (const void *)-2LL || v15 )
              v10 = v15;
            v44 = v40 + v41;
            v15 = v10;
            ++v40;
            v41 = v39 & v44;
            continue;
          }
          v50 = v40;
          v53 = v15;
          v57 = v39;
          if ( !v8 )
            goto LABEL_42;
          v45 = memcmp(v7, v42, v8);
          v10 = v37 + 16LL * v41;
          v39 = v57;
          v15 = v53;
          v40 = v50;
          v43 = v45 == 0;
        }
        if ( v43 )
          goto LABEL_42;
        if ( v42 == (const void *)-1LL )
          goto LABEL_55;
        goto LABEL_48;
      }
      break;
    }
LABEL_58:
    if ( v7 == (_QWORD *)-1LL )
      goto LABEL_42;
LABEL_55:
    if ( v15 )
      v10 = v15;
LABEL_42:
    v34 = v62 + 1;
LABEL_36:
    LODWORD(v62) = v34;
    if ( *(_QWORD *)v10 != -1 )
      --HIDWORD(v62);
    *(_QWORD *)v10 = v7;
    *(_QWORD *)(v10 + 8) = v8;
LABEL_23:
    v5 += 16;
  }
  while ( v58 != v5 );
  v3 = v64;
LABEL_25:
  v29 = v61;
  v61 = 0;
  ++v60;
  *(_QWORD *)(a1 + 8) = v29;
  v30 = v62;
  *(_QWORD *)a1 = 1;
  *(_QWORD *)(a1 + 16) = v30;
  v62 = 0;
  *(_DWORD *)(a1 + 24) = v63;
  LODWORD(v63) = 0;
  if ( v3 == v66 )
  {
    v31 = 0;
    v32 = 0;
  }
  else
  {
    _libc_free((unsigned __int64)v3);
    v31 = v61;
    v32 = 16LL * (unsigned int)v63;
  }
  sub_C7D6A0(v31, v32, 8);
  return a1;
}
