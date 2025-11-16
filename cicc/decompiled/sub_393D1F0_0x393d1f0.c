// Function: sub_393D1F0
// Address: 0x393d1f0
//
__int64 __fastcall sub_393D1F0(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  bool v3; // zf
  __int64 v4; // r15
  __int64 v5; // r12
  unsigned __int8 *v6; // rax
  __int64 v7; // rcx
  __int64 v8; // r13
  __int64 v9; // rax
  __int64 v10; // rdx
  int v11; // ebx
  __int64 v12; // rax
  unsigned int v13; // eax
  int v14; // r8d
  int v15; // r9d
  __int64 v16; // rax
  __m128i *v17; // rax
  __int64 v18; // rax
  int v19; // edx
  __int64 v20; // r13
  _BYTE *v21; // r12
  __int64 v22; // rsi
  _BYTE *v23; // rdx
  _BYTE *v24; // r8
  unsigned int v25; // eax
  __int64 v26; // rcx
  __int64 v28; // rdx
  _QWORD *v29; // r15
  __int64 v30; // rbx
  unsigned __int64 v31; // r13
  __int64 v32; // r14
  __int64 v33; // rbx
  unsigned __int64 v34; // r12
  size_t v35; // rdx
  int v36; // eax
  __int64 v37; // r12
  size_t v38; // r14
  size_t v39; // r8
  size_t v40; // rdx
  int v41; // eax
  __int64 v42; // rdi
  unsigned __int64 v43; // rbx
  int v44; // [rsp+14h] [rbp-17Ch]
  __int64 v45; // [rsp+18h] [rbp-178h]
  size_t v46; // [rsp+18h] [rbp-178h]
  __int64 v47; // [rsp+28h] [rbp-168h]
  __int64 v48; // [rsp+30h] [rbp-160h]
  __int64 v49; // [rsp+38h] [rbp-158h]
  __int64 v50; // [rsp+38h] [rbp-158h]
  __m128i s2; // [rsp+40h] [rbp-150h] BYREF
  _QWORD v52[2]; // [rsp+50h] [rbp-140h] BYREF
  _BYTE *v53; // [rsp+60h] [rbp-130h] BYREF
  __int64 v54; // [rsp+68h] [rbp-128h]
  _BYTE v55[288]; // [rsp+70h] [rbp-120h] BYREF

  v2 = a1;
  v3 = *(_DWORD *)(a2 + 8) == 2;
  v53 = v55;
  v54 = 0xA00000000LL;
  if ( !v3 )
    return v2;
  v4 = *(_QWORD *)(a2 - 8);
  v5 = a2;
  if ( !v4 )
    return v2;
  do
  {
    v6 = sub_15B1000(*(unsigned __int8 **)(v5 - 16));
    v7 = *((unsigned int *)v6 + 2);
    v8 = *(_QWORD *)&v6[8 * (3 - v7)];
    if ( v8 )
    {
      v9 = sub_161E970(*(_QWORD *)&v6[8 * (3 - v7)]);
      v49 = v10;
      v8 = v9;
    }
    else
    {
      v49 = 0;
    }
    v11 = 0;
    v5 = v4;
    v12 = *(_QWORD *)(v4 - 8LL * *(unsigned int *)(v4 + 8));
    if ( *(_BYTE *)v12 == 19 )
    {
      v13 = *(_DWORD *)(v12 + 24);
      if ( (v13 & 1) == 0 )
      {
        v11 = (v13 >> 1) & 0x1F;
        if ( ((v13 >> 1) & 0x20) != 0 )
          v11 |= (v13 >> 2) & 0xFE0;
      }
    }
    s2.m128i_i64[0] = __PAIR64__(v11, sub_393D1C0(v4));
    s2.m128i_i64[1] = v8;
    v52[0] = v49;
    v16 = (unsigned int)v54;
    if ( (unsigned int)v54 >= HIDWORD(v54) )
    {
      sub_16CD150((__int64)&v53, v55, 0, 24, v14, v15);
      v16 = (unsigned int)v54;
    }
    v17 = (__m128i *)&v53[24 * v16];
    *v17 = _mm_loadu_si128(&s2);
    v17[1].m128i_i64[0] = v52[0];
    v18 = (int)v54;
    v19 = v54 + 1;
    LODWORD(v54) = v54 + 1;
    if ( *(_DWORD *)(v4 + 8) != 2 )
      break;
    v4 = *(_QWORD *)(v4 - 8);
  }
  while ( v4 );
  v20 = a1;
  v21 = v53;
  if ( !v19 || (v44 = v18, (int)v18 < 0) )
  {
LABEL_64:
    v2 = v20;
    goto LABEL_24;
  }
  v47 = 24 * v18;
  while ( 2 )
  {
    v22 = v20 + 88;
    v20 = *(_QWORD *)(v20 + 96);
    v23 = &v21[v47];
    v24 = *(_BYTE **)&v21[v47 + 8];
    if ( !v20 )
      goto LABEL_23;
    v25 = *(_DWORD *)v23;
    v26 = v22;
    do
    {
      while ( 1 )
      {
        if ( *(_DWORD *)(v20 + 32) < v25 )
        {
          v20 = *(_QWORD *)(v20 + 24);
          goto LABEL_20;
        }
        if ( *(_DWORD *)(v20 + 32) == v25 && *(_DWORD *)(v20 + 36) < *((_DWORD *)v23 + 1) )
          break;
        v26 = v20;
        v20 = *(_QWORD *)(v20 + 16);
        if ( !v20 )
          goto LABEL_21;
      }
      v20 = *(_QWORD *)(v20 + 24);
LABEL_20:
      ;
    }
    while ( v20 );
LABEL_21:
    v48 = v26;
    if ( v26 == v22
      || *(_DWORD *)(v26 + 32) > v25
      || *(_DWORD *)(v26 + 32) == v25 && *((_DWORD *)v23 + 1) < *(_DWORD *)(v26 + 36) )
    {
      goto LABEL_23;
    }
    if ( v24 )
    {
      v28 = (__int64)&v24[*((_QWORD *)v23 + 2)];
      s2.m128i_i64[0] = (__int64)v52;
      sub_393CF10(s2.m128i_i64, v24, v28);
      v29 = (_QWORD *)s2.m128i_i64[0];
      v30 = *(_QWORD *)(v48 + 56);
      v50 = v48 + 48;
      if ( !v30 )
        goto LABEL_54;
    }
    else
    {
      v29 = v52;
      s2.m128i_i64[1] = 0;
      s2.m128i_i64[0] = (__int64)v52;
      LOBYTE(v52[0]) = 0;
      v30 = *(_QWORD *)(v26 + 56);
      v50 = v26 + 48;
      if ( !v30 )
        goto LABEL_57;
    }
    v45 = v20;
    v31 = s2.m128i_u64[1];
    v32 = v30;
    v33 = v50;
    while ( 2 )
    {
      while ( 2 )
      {
        v34 = *(_QWORD *)(v32 + 40);
        v35 = v31;
        if ( v34 <= v31 )
          v35 = *(_QWORD *)(v32 + 40);
        if ( !v35 || (v36 = memcmp(*(const void **)(v32 + 32), v29, v35)) == 0 )
        {
          v37 = v34 - v31;
          if ( v37 >= 0x80000000LL )
            goto LABEL_44;
          if ( v37 > (__int64)0xFFFFFFFF7FFFFFFFLL )
          {
            v36 = v37;
            break;
          }
LABEL_35:
          v32 = *(_QWORD *)(v32 + 24);
          if ( !v32 )
            goto LABEL_45;
          continue;
        }
        break;
      }
      if ( v36 < 0 )
        goto LABEL_35;
LABEL_44:
      v33 = v32;
      v32 = *(_QWORD *)(v32 + 16);
      if ( v32 )
        continue;
      break;
    }
LABEL_45:
    v38 = v31;
    v20 = v45;
    if ( v33 == v50 )
    {
      if ( v29 != v52 )
        goto LABEL_55;
      goto LABEL_56;
    }
    v39 = *(_QWORD *)(v33 + 40);
    v40 = v38;
    if ( v39 <= v38 )
      v40 = *(_QWORD *)(v33 + 40);
    if ( v40 )
    {
      v46 = *(_QWORD *)(v33 + 40);
      v41 = memcmp(v29, *(const void **)(v33 + 32), v40);
      v39 = v46;
      if ( v41 )
      {
LABEL_53:
        if ( v41 < 0 )
          goto LABEL_54;
LABEL_67:
        if ( v29 != v52 )
          j_j___libc_free_0((unsigned __int64)v29);
        v21 = v53;
        v20 = v33 + 64;
LABEL_62:
        v47 -= 24;
        if ( --v44 < 0 || !v20 )
          goto LABEL_64;
        continue;
      }
    }
    break;
  }
  if ( (__int64)(v38 - v39) >= 0x80000000LL )
    goto LABEL_67;
  if ( (__int64)(v38 - v39) > (__int64)0xFFFFFFFF7FFFFFFFLL )
  {
    v41 = v38 - v39;
    goto LABEL_53;
  }
LABEL_54:
  if ( v29 != v52 )
LABEL_55:
    j_j___libc_free_0((unsigned __int64)v29);
LABEL_56:
  v21 = v53;
LABEL_57:
  v42 = *(_QWORD *)(v48 + 64);
  if ( v42 != v50 )
  {
    v43 = 0;
    do
    {
      if ( v43 <= *(_QWORD *)(v42 + 80) )
      {
        v20 = v42 + 64;
        v43 = *(_QWORD *)(v42 + 80);
      }
      v42 = sub_220EF30(v42);
    }
    while ( v42 != v50 );
    goto LABEL_62;
  }
LABEL_23:
  v2 = 0;
LABEL_24:
  if ( v21 != v55 )
    _libc_free((unsigned __int64)v21);
  return v2;
}
