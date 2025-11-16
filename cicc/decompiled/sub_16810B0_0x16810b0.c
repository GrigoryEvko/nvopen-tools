// Function: sub_16810B0
// Address: 0x16810b0
//
void __fastcall sub_16810B0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 *v3; // r12
  unsigned __int64 v4; // r15
  __int64 v5; // rdx
  unsigned __int64 v6; // rcx
  __int64 v7; // r13
  __int64 v8; // r13
  __int64 v9; // rbx
  __int64 *v10; // r14
  __int64 v11; // rdx
  _BYTE *v12; // rsi
  __int64 *v13; // rax
  _QWORD *v14; // r14
  _QWORD *v15; // r15
  __int64 v16; // rbx
  __int64 *v17; // rax
  unsigned __int64 v18; // rsi
  unsigned __int64 v19; // r14
  __int64 *v20; // r15
  __int64 v21; // rdx
  __int64 *v22; // rdi
  size_t v23; // rsi
  __int64 v24; // rdx
  __int64 v25; // r10
  _BYTE *v26; // rsi
  size_t v27; // rdx
  __int64 v28; // rax
  __int64 v29; // rsi
  _BYTE *v30; // r14
  __int64 v31; // rbx
  __int64 *v32; // r12
  __int64 v33; // rdx
  __int64 *v34; // rdi
  size_t v35; // rsi
  __int64 v36; // rdx
  __int64 v37; // r10
  _BYTE *v38; // rsi
  size_t v39; // rdx
  __int64 v40; // rax
  __int64 *v41; // r13
  __int64 *v42; // rbx
  __int64 *v43; // rbx
  __int64 v44; // rdx
  _BYTE *v45; // rsi
  __int64 *v46; // rax
  _BYTE *v49; // [rsp+8h] [rbp-C8h]
  __int64 v50; // [rsp+10h] [rbp-C0h]
  _BYTE *v51; // [rsp+20h] [rbp-B0h]
  __int64 *v52; // [rsp+28h] [rbp-A8h]
  _QWORD v53[2]; // [rsp+30h] [rbp-A0h] BYREF
  __int64 *v54; // [rsp+40h] [rbp-90h] BYREF
  size_t n; // [rsp+48h] [rbp-88h]
  _QWORD src[2]; // [rsp+50h] [rbp-80h] BYREF
  _BYTE *v57; // [rsp+60h] [rbp-70h] BYREF
  __int64 v58; // [rsp+68h] [rbp-68h]
  _BYTE v59[96]; // [rsp+70h] [rbp-60h] BYREF

  v3 = a1;
  *a1 = 0;
  a1[1] = 0;
  a1[2] = 0;
  v53[0] = a2;
  v53[1] = a3;
  v57 = v59;
  v58 = 0x300000000LL;
  sub_16D2880(v53, &v57, 44, 0xFFFFFFFFLL, 0);
  v4 = (unsigned __int64)v57;
  v5 = 16LL * (unsigned int)v58;
  v51 = &v57[v5];
  v6 = v5 >> 4;
  v50 = v5 >> 4;
  v7 = v5 >> 4;
  v52 = (__int64 *)*a1;
  if ( v5 >> 4 > (unsigned __int64)((v3[2] - *v3) >> 5) )
  {
    v8 = 32 * v7;
    v9 = 0;
    if ( v6 )
      v9 = sub_22077B0(v8);
    v10 = (__int64 *)v9;
    if ( (_BYTE *)v4 == v51 )
    {
LABEL_11:
      v14 = (_QWORD *)a1[1];
      v15 = (_QWORD *)*a1;
      if ( v14 != (_QWORD *)*a1 )
      {
        do
        {
          if ( (_QWORD *)*v15 != v15 + 2 )
            j_j___libc_free_0(*v15, v15[2] + 1LL);
          v15 += 4;
        }
        while ( v14 != v15 );
        v15 = (_QWORD *)*a1;
      }
      if ( v15 )
        j_j___libc_free_0(v15, a1[2] - (_QWORD)v15);
      *a1 = v9;
      v16 = v8 + v9;
      a1[1] = v16;
      a1[2] = v16;
      goto LABEL_19;
    }
    while ( 1 )
    {
      if ( !v10 )
        goto LABEL_7;
      v12 = *(_BYTE **)v4;
      v13 = v10 + 2;
      if ( *(_QWORD *)v4 )
      {
        v11 = *(_QWORD *)(v4 + 8);
        *v10 = (__int64)v13;
        sub_1680AA0(v10, v12, (__int64)&v12[v11]);
LABEL_7:
        v4 += 16LL;
        v10 += 4;
        if ( v51 == (_BYTE *)v4 )
          goto LABEL_11;
      }
      else
      {
        v4 += 16LL;
        *v10 = (__int64)v13;
        v10 += 4;
        *(v10 - 3) = 0;
        *((_BYTE *)v10 - 16) = 0;
        if ( v51 == (_BYTE *)v4 )
          goto LABEL_11;
      }
    }
  }
  v17 = (__int64 *)a1[1];
  v18 = ((char *)v17 - (char *)v52) >> 5;
  if ( v6 <= v18 )
  {
    if ( !v5 )
    {
LABEL_52:
      v41 = v52;
      if ( v52 != v17 )
      {
        v42 = v17;
        do
        {
          if ( (__int64 *)*v41 != v41 + 2 )
            j_j___libc_free_0(*v41, v41[2] + 1);
          v41 += 4;
        }
        while ( v41 != v42 );
        v3[1] = (__int64)v52;
      }
      goto LABEL_19;
    }
    v19 = (unsigned __int64)v57;
    v20 = v52 + 2;
    while ( 1 )
    {
      v26 = *(_BYTE **)v19;
      if ( !*(_QWORD *)v19 )
        break;
      v21 = *(_QWORD *)(v19 + 8);
      v54 = src;
      sub_1680AA0((__int64 *)&v54, v26, (__int64)&v26[v21]);
      v22 = (__int64 *)*(v20 - 2);
      if ( v54 == src )
      {
        v27 = n;
        if ( n )
        {
          if ( n == 1 )
            *(_BYTE *)v22 = src[0];
          else
            memcpy(v22, src, n);
          v27 = n;
        }
        goto LABEL_34;
      }
      v23 = n;
      v24 = src[0];
      if ( v22 == v20 )
      {
        *(v20 - 2) = (__int64)v54;
        *(v20 - 1) = v23;
        *v20 = v24;
LABEL_50:
        v54 = src;
        v22 = src;
        goto LABEL_29;
      }
      v25 = *v20;
      *(v20 - 2) = (__int64)v54;
      *(v20 - 1) = v23;
      *v20 = v24;
      if ( !v22 )
        goto LABEL_50;
      v54 = v22;
      src[0] = v25;
LABEL_29:
      n = 0;
      *(_BYTE *)v22 = 0;
      if ( v54 != src )
        j_j___libc_free_0(v54, src[0] + 1LL);
      v19 += 16LL;
      v20 += 4;
      if ( !--v7 )
      {
        v3 = a1;
        v52 += 4 * v50;
        v17 = (__int64 *)a1[1];
        goto LABEL_52;
      }
    }
    v54 = src;
    v27 = 0;
    LOBYTE(src[0]) = 0;
LABEL_34:
    v28 = *(v20 - 2);
    *(v20 - 1) = v27;
    *(_BYTE *)(v28 + v27) = 0;
    v22 = v54;
    goto LABEL_29;
  }
  v29 = 16 * v18;
  v30 = &v57[v29];
  if ( !v29 )
    goto LABEL_59;
  v49 = &v57[v29];
  v31 = v29 >> 4;
  v32 = v52 + 2;
  do
  {
    v38 = *(_BYTE **)v4;
    if ( !*(_QWORD *)v4 )
    {
      v54 = src;
      v39 = 0;
      LOBYTE(src[0]) = 0;
LABEL_46:
      v40 = *(v32 - 2);
      *(v32 - 1) = v39;
      *(_BYTE *)(v40 + v39) = 0;
      v34 = v54;
      goto LABEL_41;
    }
    v33 = *(_QWORD *)(v4 + 8);
    v54 = src;
    sub_1680AA0((__int64 *)&v54, v38, (__int64)&v38[v33]);
    v34 = (__int64 *)*(v32 - 2);
    if ( v54 == src )
    {
      v39 = n;
      if ( n )
      {
        if ( n == 1 )
          *(_BYTE *)v34 = src[0];
        else
          memcpy(v34, src, n);
        v39 = n;
      }
      goto LABEL_46;
    }
    v35 = n;
    v36 = src[0];
    if ( v32 == v34 )
    {
      *(v32 - 2) = (__int64)v54;
      *(v32 - 1) = v35;
      *v32 = v36;
    }
    else
    {
      v37 = *v32;
      *(v32 - 2) = (__int64)v54;
      *(v32 - 1) = v35;
      *v32 = v36;
      if ( v34 )
      {
        v54 = v34;
        src[0] = v37;
        goto LABEL_41;
      }
    }
    v54 = src;
    v34 = src;
LABEL_41:
    n = 0;
    *(_BYTE *)v34 = 0;
    if ( v54 != src )
      j_j___libc_free_0(v54, src[0] + 1LL);
    v4 += 16LL;
    v32 += 4;
    --v31;
  }
  while ( v31 );
  v3 = a1;
  v30 = v49;
  v17 = (__int64 *)a1[1];
LABEL_59:
  if ( v51 == v30 )
    goto LABEL_67;
  v43 = v17;
  while ( 2 )
  {
    while ( 2 )
    {
      if ( !v43 )
      {
LABEL_62:
        v30 += 16;
        v43 += 4;
        if ( v51 == v30 )
          goto LABEL_66;
        continue;
      }
      break;
    }
    v45 = *(_BYTE **)v30;
    v46 = v43 + 2;
    if ( *(_QWORD *)v30 )
    {
      v44 = *((_QWORD *)v30 + 1);
      *v43 = (__int64)v46;
      sub_1680AA0(v43, v45, (__int64)&v45[v44]);
      goto LABEL_62;
    }
    v30 += 16;
    *v43 = (__int64)v46;
    v43 += 4;
    *(v43 - 3) = 0;
    *((_BYTE *)v43 - 16) = 0;
    if ( v51 != v30 )
      continue;
    break;
  }
LABEL_66:
  v17 = v43;
LABEL_67:
  v3[1] = (__int64)v17;
LABEL_19:
  if ( v57 != v59 )
    _libc_free((unsigned __int64)v57);
}
