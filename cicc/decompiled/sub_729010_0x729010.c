// Function: sub_729010
// Address: 0x729010
//
__int64 __fastcall sub_729010(__int64 *a1, unsigned int a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r14
  __int64 v6; // r12
  __int64 v7; // rbx
  __m128i *v8; // rax
  __m128i *v9; // r15
  const __m128i *v10; // rdx
  __m128i *v11; // rsi
  _QWORD *v12; // r14
  __int64 v13; // rsi
  unsigned int v14; // ebx
  int v15; // eax
  __int64 result; // rax
  int v17; // eax
  unsigned int v18; // r14d
  _QWORD *v19; // rax
  _QWORD *v20; // rsi
  __int64 v21; // rcx
  __int64 v22; // r9
  unsigned __int64 *v23; // rcx
  __int64 v24; // r10
  unsigned __int64 v25; // rdi
  unsigned __int64 i; // rdx
  unsigned int v27; // edx
  unsigned __int64 *v28; // rax
  unsigned __int64 v29; // rdx
  unsigned __int64 v30; // rdx

  v4 = a2;
  v6 = *(_QWORD *)(a4 + 8);
  v7 = *(_QWORD *)(a4 + 16);
  v8 = (__m128i *)sub_823970(24 * v6);
  v9 = v8;
  v10 = *(const __m128i **)a4;
  if ( v7 > 0 )
  {
    v11 = (__m128i *)((char *)v8 + 24 * v7);
    do
    {
      if ( v8 )
      {
        *v8 = _mm_loadu_si128(v10);
        v8[1].m128i_i64[0] = v10[1].m128i_i64[0];
      }
      v8 = (__m128i *)((char *)v8 + 24);
      v10 = (const __m128i *)((char *)v10 + 24);
    }
    while ( v11 != v8 );
  }
  v12 = (_QWORD *)(*a1 + 32 * v4);
  if ( *v12 )
    sub_823A00(v12[1], 24LL * v12[2]);
  *v12 = a3;
  if ( a3 )
  {
    v12[1] = v9;
    v13 = 0;
    v9 = 0;
    v12[3] = v7;
    v12[2] = v6;
    v14 = *((_DWORD *)a1 + 2);
    v15 = *((_DWORD *)a1 + 3) + 1;
    *((_DWORD *)a1 + 3) = v15;
    if ( 2 * v15 <= v14 )
      return sub_823A00(v9, v13);
    v6 = 0;
  }
  else
  {
    v14 = *((_DWORD *)a1 + 2);
    v17 = *((_DWORD *)a1 + 3) + 1;
    *((_DWORD *)a1 + 3) = v17;
    result = (unsigned int)(2 * v17);
    if ( (unsigned int)result <= v14 )
      return result;
  }
  v18 = 2 * v14 + 1;
  v19 = (_QWORD *)sub_823970(32LL * (2 * v14 + 2));
  v20 = v19;
  if ( 2 * v14 != -2 )
  {
    v21 = (__int64)&v19[4 * v18 + 4];
    do
    {
      if ( v19 )
        *v19 = 0;
      v19 += 4;
    }
    while ( (_QWORD *)v21 != v19 );
  }
  v22 = *a1;
  if ( v14 != -1 )
  {
    v23 = (unsigned __int64 *)*a1;
    v24 = v22 + 32LL * v14 + 32;
    do
    {
      while ( 1 )
      {
        v25 = *v23;
        if ( *v23 )
          break;
        v23 += 4;
        if ( v23 == (unsigned __int64 *)v24 )
          goto LABEL_27;
      }
      for ( i = v25 >> 3; ; LODWORD(i) = v27 + 1 )
      {
        v27 = v18 & i;
        v28 = &v20[4 * v27];
        if ( !*v28 )
          break;
      }
      *v28 = v25;
      v29 = v23[3];
      v23 += 4;
      v28[3] = v29;
      v30 = *(v23 - 2);
      v28[1] = *(v23 - 3);
      v28[2] = v30;
      *(v23 - 3) = 0;
      *(v23 - 2) = 0;
      *(v23 - 1) = 0;
    }
    while ( v23 != (unsigned __int64 *)v24 );
  }
LABEL_27:
  *a1 = (__int64)v20;
  *((_DWORD *)a1 + 2) = v18;
  result = sub_823A00(v22, 32LL * (v14 + 1));
  if ( a3 )
  {
    v13 = 24 * v6;
    return sub_823A00(v9, v13);
  }
  return result;
}
