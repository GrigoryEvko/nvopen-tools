// Function: sub_69DF50
// Address: 0x69df50
//
unsigned __int64 __fastcall sub_69DF50(__int64 *a1, int a2, const __m128i *a3, int a4)
{
  __int64 v4; // r12
  int v8; // r8d
  int *v9; // rdi
  unsigned int v10; // ecx
  __int64 v11; // rdx
  int *v12; // rsi
  unsigned int v13; // ebx
  int *v14; // rsi
  int *v15; // rax
  __m128i v17; // xmm0
  unsigned int v18; // edx
  int v19; // eax
  __m128i v20; // xmm0
  unsigned int v21; // edx
  int v22; // eax
  unsigned int v23; // r15d
  unsigned int v24; // r14d
  _DWORD *v25; // rax
  _DWORD *v26; // rsi
  _DWORD *v27; // rcx
  int *v28; // r8
  int *v29; // rcx
  int v30; // edi
  unsigned int j; // eax
  _DWORD *v32; // rdx
  _DWORD *v33; // rax
  _DWORD *v34; // rcx
  int *v35; // rcx
  int v36; // edi
  unsigned int i; // eax
  _DWORD *v38; // rdx
  unsigned int v39; // [rsp+8h] [rbp-58h]
  unsigned int v40; // [rsp+8h] [rbp-58h]
  unsigned int v41; // [rsp+Ch] [rbp-54h]
  unsigned int v42; // [rsp+Ch] [rbp-54h]

  v8 = *((_DWORD *)a1 + 2);
  v9 = (int *)*a1;
  v10 = v8 & a4;
  v11 = v10;
  v12 = &v9[6 * v10];
  v13 = *v12;
  if ( !*v12 )
  {
    v17 = _mm_loadu_si128(a3);
    *v12 = a2;
    if ( a2 )
      *(__m128i *)(v12 + 2) = v17;
    v18 = *((_DWORD *)a1 + 2);
    v19 = *((_DWORD *)a1 + 3) + 1;
    *((_DWORD *)a1 + 3) = v19;
    if ( 2 * v19 <= v18 )
      return v4 & 0xFFFFFFFF00000000LL | v13;
    v40 = v18;
    v23 = v18 + 1;
    v24 = 2 * v18 + 1;
    v42 = 2 * v18 + 2;
    v33 = (_DWORD *)sub_823970(24LL * v42);
    v26 = v33;
    if ( v42 )
    {
      v34 = &v33[6 * v24 + 6];
      do
      {
        if ( v33 )
          *v33 = 0;
        v33 += 6;
      }
      while ( v34 != v33 );
    }
    v28 = (int *)*a1;
    if ( v23 )
    {
      v35 = (int *)*a1;
      do
      {
        v36 = *v35;
        if ( *v35 )
        {
          for ( i = v36 & v24; ; i = v24 & (i + 1) )
          {
            v38 = &v26[6 * i];
            if ( !*v38 )
              break;
          }
          *v38 = v36;
          *(__m128i *)(v38 + 2) = _mm_loadu_si128((const __m128i *)(v35 + 2));
        }
        v35 += 6;
      }
      while ( &v28[6 * v40 + 6] != v35 );
    }
LABEL_40:
    *a1 = (__int64)v26;
    *((_DWORD *)a1 + 2) = v24;
    sub_823A00(v28, 24LL * v23);
    return v4 & 0xFFFFFFFF00000000LL | v13;
  }
  do
  {
    if ( a2 == v13 )
    {
      v15 = &v9[6 * v11];
      v4 = *((_QWORD *)v15 + 1);
      v13 = v15[2];
      *(__m128i *)(v15 + 2) = _mm_loadu_si128(a3);
      return v4 & 0xFFFFFFFF00000000LL | v13;
    }
    v10 = v8 & (v10 + 1);
    v11 = v10;
    v14 = &v9[6 * v10];
    v13 = *v14;
  }
  while ( *v14 );
  v20 = _mm_loadu_si128(a3);
  *v14 = a2;
  if ( a2 )
    *(__m128i *)(v14 + 2) = v20;
  v21 = *((_DWORD *)a1 + 2);
  v22 = *((_DWORD *)a1 + 3) + 1;
  *((_DWORD *)a1 + 3) = v22;
  if ( 2 * v22 > v21 )
  {
    v39 = v21;
    v23 = v21 + 1;
    v24 = 2 * v21 + 1;
    v41 = 2 * v21 + 2;
    v25 = (_DWORD *)sub_823970(24LL * v41);
    v26 = v25;
    if ( v41 )
    {
      v27 = &v25[6 * v24 + 6];
      do
      {
        if ( v25 )
          *v25 = 0;
        v25 += 6;
      }
      while ( v25 != v27 );
    }
    v28 = (int *)*a1;
    if ( v23 )
    {
      v29 = (int *)*a1;
      do
      {
        v30 = *v29;
        if ( *v29 )
        {
          for ( j = v30 & v24; ; j = v24 & (j + 1) )
          {
            v32 = &v26[6 * j];
            if ( !*v32 )
              break;
          }
          *v32 = v30;
          *(__m128i *)(v32 + 2) = _mm_loadu_si128((const __m128i *)(v29 + 2));
        }
        v29 += 6;
      }
      while ( &v28[6 * v39 + 6] != v29 );
    }
    goto LABEL_40;
  }
  return v4 & 0xFFFFFFFF00000000LL | v13;
}
