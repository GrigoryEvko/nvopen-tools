// Function: sub_840360
// Address: 0x840360
//
__int64 __fastcall sub_840360(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        _BOOL4 a4,
        int a5,
        int a6,
        __int64 a7,
        int a8,
        int a9,
        __int64 a10,
        _DWORD *a11,
        const __m128i **a12)
{
  __int64 v14; // r15
  _QWORD *v15; // rax
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 *v18; // r9
  __int64 v19; // rdx
  __int64 *v20; // rdi
  __int64 *v21; // rcx
  __int64 v22; // rdx
  bool v23; // zf
  __int64 i; // rax
  unsigned int v25; // r13d
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // rbx
  __int64 v29; // r8
  __int64 *v30; // r9
  __int64 v31; // r12
  __int64 v32; // rdx
  __int64 v33; // rcx
  __int64 v34; // r8
  __int64 *v35; // r9
  __int64 v37; // r8
  __int64 v38; // r9
  __int64 v39; // r14
  const __m128i *v40; // r12
  __m128i *v41; // rbx
  __int64 v42; // rax
  __int64 v43; // rax
  __int64 v44; // rax
  __int64 j; // rax
  __int64 k; // r13
  __int64 v47; // [rsp+0h] [rbp-60h]
  _QWORD *v48; // [rsp+8h] [rbp-58h]
  int v51; // [rsp+1Ch] [rbp-44h]
  unsigned int v52; // [rsp+24h] [rbp-3Ch] BYREF
  __int64 v53[7]; // [rsp+28h] [rbp-38h] BYREF

  v14 = a7;
  v51 = a3;
  v15 = (_QWORD *)sub_82BD70(a1, a2, a3);
  v19 = v15[128];
  if ( v19 == v15[127] )
  {
    v47 = v15[128];
    v48 = v15;
    sub_8332F0((__int64)v15, a2, v19, v16, v17, v18);
    v19 = v47;
    v15 = v48;
  }
  v20 = (__int64 *)v15[126];
  v21 = &v20[5 * v19];
  if ( v21 )
  {
    *(_BYTE *)v21 &= 0xFCu;
    v21[1] = 0;
    v21[2] = 0;
    v21[3] = 0;
    v21[4] = 0;
  }
  v22 = v19 + 1;
  v15[128] = v22;
  *(_OWORD *)a10 = 0;
  v23 = dword_4F04C44 == -1;
  *(_OWORD *)(a10 + 16) = 0;
  *(_OWORD *)(a10 + 32) = 0;
  if ( !v23
    || (v22 = (__int64)qword_4F04C68, v42 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v42 + 6) & 6) != 0)
    || *(_BYTE *)(v42 + 4) == 12 )
  {
    for ( i = *a1; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    if ( (*(_BYTE *)(i + 177) & 0x20) != 0 || a2 && (v20 = (__int64 *)a2, (unsigned int)sub_8DC060(a2)) )
    {
      *(_BYTE *)(a10 + 17) |= 1u;
      v25 = 1;
      goto LABEL_10;
    }
  }
  v53[0] = 0;
  sub_83EB80(a1, a2, (const __m128i *)a2, v51, a4, a5, a6, a7, a8, a9, v53);
  if ( a7 )
  {
    v39 = v53[0];
    if ( *(_BYTE *)(a7 + 140) == 12 )
    {
      do
        v14 = *(_QWORD *)(v14 + 160);
      while ( *(_BYTE *)(v14 + 140) == 12 );
    }
    if ( (unsigned int)sub_8D2FB0(v14) )
    {
      v43 = sub_8D46C0(v14);
      if ( (unsigned int)sub_8D2310(v43) )
      {
        while ( v39 && (*(_BYTE *)(v39 + 145) & 0x40) != 0 )
        {
          v44 = *(_QWORD *)(v39 + 64);
          if ( v44 )
          {
            for ( j = *(_QWORD *)(v44 + 152); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
              ;
            for ( k = *(_QWORD *)(j + 160); *(_BYTE *)(k + 140) == 12; k = *(_QWORD *)(k + 160) )
              ;
            if ( (unsigned int)sub_8D2FB0(k) && ((*(_BYTE *)(k + 168) ^ *(_BYTE *)(v14 + 168)) & 2) == 0 )
              *(_BYTE *)(v39 + 145) |= 0x80u;
          }
          v39 = *(_QWORD *)v39;
        }
      }
    }
  }
  v20 = v53;
  a2 = (__int64)a1 + 68;
  sub_82D8D0(v53, (__int64)a1 + 68, &v52, a11, v37, v38);
  v25 = v52;
  if ( v52 )
  {
    v40 = (const __m128i *)v53[0];
    if ( *a11 )
    {
      *(_BYTE *)(a10 + 16) |= 8u;
      if ( *a11 )
      {
        if ( a12 )
        {
          v25 = 0;
LABEL_34:
          *a12 = v40;
          goto LABEL_10;
        }
      }
    }
    v25 = 0;
    if ( !v40 )
      goto LABEL_10;
    goto LABEL_22;
  }
  v40 = (const __m128i *)v53[0];
  if ( v53[0] )
  {
    if ( *a11 )
    {
      *(_BYTE *)(a10 + 16) |= 8u;
      a2 = (unsigned int)*a11;
      if ( (_DWORD)a2 && a12 )
        goto LABEL_34;
    }
    else
    {
      *(__m128i *)a10 = _mm_loadu_si128((const __m128i *)(v53[0] + 64));
      *(__m128i *)(a10 + 16) = _mm_loadu_si128(v40 + 5);
      *(__m128i *)(a10 + 32) = _mm_loadu_si128(v40 + 6);
      if ( *a11 )
      {
        *(_BYTE *)(a10 + 16) |= 8u;
        if ( *a11 )
        {
          v25 = 1;
          if ( a12 )
            goto LABEL_34;
        }
      }
      v25 = 1;
    }
    do
    {
LABEL_22:
      v41 = (__m128i *)v40;
      v40 = (const __m128i *)v40->m128i_i64[0];
      sub_725130((__int64 *)v41[2].m128i_i64[1]);
      v20 = (__int64 *)v41[7].m128i_i64[1];
      sub_82D8A0(v20);
      v41->m128i_i64[0] = (__int64)qword_4D03C68;
      qword_4D03C68 = v41->m128i_i64;
    }
    while ( v40 );
    goto LABEL_10;
  }
  if ( *a11 )
  {
    *(_BYTE *)(a10 + 16) |= 8u;
    v22 = (unsigned int)*a11;
    if ( (_DWORD)v22 )
    {
      if ( a12 )
        goto LABEL_34;
    }
  }
LABEL_10:
  v28 = sub_82BD70(v20, a2, v22);
  v31 = *(_QWORD *)(*(_QWORD *)(v28 + 1008) + 8 * (5LL * *(_QWORD *)(v28 + 1024) - 5) + 32);
  if ( v31 )
  {
    sub_823A00(*(_QWORD *)v31, 16LL * (unsigned int)(*(_DWORD *)(v31 + 8) + 1), v26, v27, v29, v30);
    sub_823A00(v31, 16, v32, v33, v34, v35);
  }
  --*(_QWORD *)(v28 + 1024);
  return v25;
}
