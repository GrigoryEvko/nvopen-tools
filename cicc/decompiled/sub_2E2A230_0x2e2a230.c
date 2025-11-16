// Function: sub_2E2A230
// Address: 0x2e2a230
//
void __fastcall sub_2E2A230(__m128i *a1, __int64 a2, __int64 a3, __int64 a4, int *a5, __int64 a6)
{
  __m128i *v6; // r14
  __int64 v7; // r13
  unsigned int v8; // r12d
  __int64 v9; // r10
  __int64 v10; // rbx
  unsigned int v11; // r15d
  __int64 v12; // rax
  unsigned int v13; // r15d
  char v14; // dl
  __int64 v15; // rax
  unsigned __int64 v16; // rdx
  __int64 v17; // rdi
  unsigned int *v18; // rbx
  unsigned int *v19; // r15
  __m128i *v20; // r13
  unsigned int v22; // esi
  __int64 v23; // rax
  unsigned int *v24; // r15
  unsigned int *v25; // rbx
  __int64 v26; // rax
  unsigned int *v27; // r12
  unsigned int *v28; // r15
  unsigned int v29; // esi
  __int64 v30; // rax
  unsigned __int64 v31; // rdx
  __int64 v32; // rax
  unsigned __int64 v33; // rdx
  __int64 v34; // [rsp+0h] [rbp-C0h]
  int *v35; // [rsp+0h] [rbp-C0h]
  int *v36; // [rsp+0h] [rbp-C0h]
  int *v37; // [rsp+8h] [rbp-B8h]
  __int64 v38; // [rsp+8h] [rbp-B8h]
  __int64 v39; // [rsp+8h] [rbp-B8h]
  unsigned int *v41; // [rsp+30h] [rbp-90h] BYREF
  __int64 v42; // [rsp+38h] [rbp-88h]
  _BYTE v43[16]; // [rsp+40h] [rbp-80h] BYREF
  unsigned int *v44; // [rsp+50h] [rbp-70h] BYREF
  __int64 v45; // [rsp+58h] [rbp-68h]
  _BYTE v46[16]; // [rsp+60h] [rbp-60h] BYREF
  unsigned int *v47; // [rsp+70h] [rbp-50h] BYREF
  __int64 v48; // [rsp+78h] [rbp-48h]
  _BYTE v49[64]; // [rsp+80h] [rbp-40h] BYREF

  v6 = a1;
  v7 = a2;
  v8 = a4;
  v9 = *(_DWORD *)(a2 + 40) & 0xFFFFFF;
  if ( *(_WORD *)(a2 + 68) != 68 && *(_WORD *)(a2 + 68) )
  {
    v44 = (unsigned int *)v46;
    v45 = 0x400000000LL;
    v48 = 0x400000000LL;
    v41 = (unsigned int *)v43;
    v47 = (unsigned int *)v49;
    v42 = 0x100000000LL;
    if ( !(_DWORD)v9 )
      goto LABEL_29;
  }
  else
  {
    a4 = (__int64)v49;
    v9 = 1;
    v44 = (unsigned int *)v46;
    v45 = 0x400000000LL;
    v48 = 0x400000000LL;
    v41 = (unsigned int *)v43;
    v47 = (unsigned int *)v49;
    v42 = 0x100000000LL;
  }
  v10 = 0;
  a5 = (int *)&v47;
  do
  {
    while ( 1 )
    {
      v11 = v10;
      v12 = *(_QWORD *)(a2 + 32) + 40 * v10;
      if ( *(_BYTE *)v12 == 12 )
        break;
      if ( *(_BYTE *)v12 || (v13 = *(_DWORD *)(v12 + 8)) == 0 )
      {
LABEL_18:
        if ( v9 == ++v10 )
          goto LABEL_19;
      }
      else
      {
        v14 = *(_BYTE *)(v12 + 3);
        a4 = v13 - 1;
        if ( (v14 & 0x10) == 0 )
        {
          if ( (unsigned int)a4 > 0x3FFFFFFE
            || (a4 = v13, (*(_QWORD *)(*(_QWORD *)(a1[5].m128i_i64[1] + 384) + 8LL * (v13 >> 6)) & (1LL << v13)) == 0) )
          {
            *(_BYTE *)(v12 + 3) &= ~0x40u;
          }
          if ( (*(_BYTE *)(v12 + 4) & 1) == 0
            && (*(_BYTE *)(v12 + 4) & 2) == 0
            && ((*(_BYTE *)(v12 + 3) & 0x10) == 0 || (*(_DWORD *)v12 & 0xFFF00) != 0) )
          {
            v15 = (unsigned int)v45;
            a4 = HIDWORD(v45);
            v16 = (unsigned int)v45 + 1LL;
            if ( v16 > HIDWORD(v45) )
            {
              v36 = a5;
              v39 = v9;
              sub_C8D5F0((__int64)&v44, v46, v16, 4u, (__int64)a5, a6);
              v15 = (unsigned int)v45;
              a5 = v36;
              v9 = v39;
            }
            v44[v15] = v13;
            LODWORD(v45) = v45 + 1;
          }
          goto LABEL_18;
        }
        if ( (unsigned int)a4 > 0x3FFFFFFE
          || (a6 = *(_QWORD *)(a1[5].m128i_i64[1] + 384), (*(_QWORD *)(a6 + 8LL * (v13 >> 6)) & (1LL << v13)) != 0) )
        {
          v30 = (unsigned int)v48;
          a4 = HIDWORD(v48);
          v31 = (unsigned int)v48 + 1LL;
          if ( v31 > HIDWORD(v48) )
            goto LABEL_54;
        }
        else
        {
          *(_BYTE *)(v12 + 3) = v14 & 0xBF;
          v30 = (unsigned int)v48;
          a4 = HIDWORD(v48);
          v31 = (unsigned int)v48 + 1LL;
          if ( v31 > HIDWORD(v48) )
          {
LABEL_54:
            v34 = v9;
            v37 = a5;
            sub_C8D5F0((__int64)a5, v49, v31, 4u, (__int64)a5, a6);
            v30 = (unsigned int)v48;
            v9 = v34;
            a5 = v37;
          }
        }
        ++v10;
        v47[v30] = v13;
        LODWORD(v48) = v48 + 1;
        if ( v9 == v10 )
          goto LABEL_19;
      }
    }
    v32 = (unsigned int)v42;
    a4 = HIDWORD(v42);
    v33 = (unsigned int)v42 + 1LL;
    if ( v33 > HIDWORD(v42) )
    {
      v35 = a5;
      v38 = v9;
      sub_C8D5F0((__int64)&v41, v43, v33, 4u, (__int64)a5, a6);
      v32 = (unsigned int)v42;
      a5 = v35;
      v9 = v38;
    }
    ++v10;
    v41[v32] = v11;
    LODWORD(v42) = v42 + 1;
  }
  while ( v9 != v10 );
LABEL_19:
  v17 = *(_QWORD *)(a2 + 24);
  v18 = &v44[(unsigned int)v45];
  if ( v18 != v44 )
  {
    v19 = v44;
    v20 = v6;
    do
    {
      while ( 1 )
      {
        v22 = *v19;
        if ( (*v19 & 0x80000000) == 0 )
          break;
        ++v19;
        sub_2E29FE0(v20, v22, v17, a2, (__int64)a5, a6);
        if ( v18 == v19 )
          goto LABEL_26;
      }
      a4 = v22;
      a5 = *(int **)(v20[5].m128i_i64[1] + 384);
      if ( (*(_QWORD *)&a5[2 * (v22 >> 6)] & (1LL << v22)) == 0 )
        sub_2E27C50(v20, v22, a2, v22, (__int64)a5);
      ++v19;
    }
    while ( v18 != v19 );
LABEL_26:
    v23 = a2;
    v6 = v20;
    v7 = v23;
  }
  v24 = v41;
  v25 = &v41[(unsigned int)v42];
  if ( v41 != v25 )
  {
    do
    {
      v26 = *v24++;
      sub_2E295D0(v6, *(_QWORD *)(v7 + 32) + 40 * v26, v8);
    }
    while ( v25 != v24 );
  }
LABEL_29:
  v27 = v47;
  v28 = &v47[(unsigned int)v48];
  if ( v28 != v47 )
  {
    do
    {
      while ( 1 )
      {
        v29 = *v27;
        if ( (*v27 & 0x80000000) == 0 )
          break;
        ++v27;
        sub_2E2A1D0(v6, v29, v7, a4, (__int64)a5, a6);
        if ( v28 == v27 )
          goto LABEL_36;
      }
      a4 = v29;
      if ( (*(_QWORD *)(*(_QWORD *)(v6[5].m128i_i64[1] + 384) + 8LL * (v29 >> 6)) & (1LL << v29)) == 0 )
        sub_2E29700(v6, v29, v7, a3, a5);
      ++v27;
    }
    while ( v28 != v27 );
  }
LABEL_36:
  sub_2E25630(v6, v7, a3);
  if ( v41 != (unsigned int *)v43 )
    _libc_free((unsigned __int64)v41);
  if ( v47 != (unsigned int *)v49 )
    _libc_free((unsigned __int64)v47);
  if ( v44 != (unsigned int *)v46 )
    _libc_free((unsigned __int64)v44);
}
