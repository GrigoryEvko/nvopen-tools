// Function: sub_2BB8DF0
// Address: 0x2bb8df0
//
void __fastcall sub_2BB8DF0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rdx
  __int64 v8; // rax
  int v10; // edx
  int v11; // eax
  int v12; // edx
  int v13; // eax
  unsigned __int64 v14; // rsi
  unsigned __int64 v15; // rsi
  int v16; // edi
  unsigned __int64 v17; // rcx
  unsigned __int64 v18; // r8
  __int64 v19; // rcx
  int *v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rsi
  int v23; // edi
  int v24; // edi
  const __m128i *v25; // rdx
  __m128i *v26; // rax
  __m128i *v27; // rdi
  const __m128i *v28; // rdx
  __m128i *v29; // rax
  __m128i *v30; // rdi
  int v31; // eax

  if ( a1 == a2 )
    return;
  v6 = a1 + 16;
  v8 = *(_QWORD *)a1;
  if ( *(_QWORD *)a1 != a1 + 16 )
  {
    v6 = *(_QWORD *)a2;
    a4 = a2 + 16;
    if ( *(_QWORD *)a2 != a2 + 16 )
    {
      *(_QWORD *)a1 = v6;
      v10 = *(_DWORD *)(a2 + 8);
      *(_QWORD *)a2 = v8;
      v11 = *(_DWORD *)(a1 + 8);
      *(_DWORD *)(a1 + 8) = v10;
      v12 = *(_DWORD *)(a2 + 12);
      *(_DWORD *)(a2 + 8) = v11;
      v13 = *(_DWORD *)(a1 + 12);
      *(_DWORD *)(a1 + 12) = v12;
      *(_DWORD *)(a2 + 12) = v13;
      return;
    }
  }
  v14 = *(unsigned int *)(a2 + 8);
  if ( v14 > *(unsigned int *)(a1 + 12) )
  {
    sub_2BB7B30(a1, v14, v6, a4, a5, a6);
    v15 = *(unsigned int *)(a1 + 8);
    v16 = v15;
    if ( *(_DWORD *)(a2 + 12) >= (unsigned int)v15 )
      goto LABEL_8;
    goto LABEL_30;
  }
  v15 = *(unsigned int *)(a1 + 8);
  v16 = v15;
  if ( *(_DWORD *)(a2 + 12) < (unsigned int)v15 )
  {
LABEL_30:
    sub_2BB7B30(a2, v15, v6, a4, a5, a6);
    v15 = *(unsigned int *)(a1 + 8);
    v16 = *(_DWORD *)(a1 + 8);
  }
LABEL_8:
  v17 = *(unsigned int *)(a2 + 8);
  v18 = v15;
  if ( v17 <= v15 )
    v18 = *(unsigned int *)(a2 + 8);
  if ( v18 )
  {
    v19 = 0;
    do
    {
      v20 = (int *)(v19 + *(_QWORD *)a2);
      v21 = v19 + *(_QWORD *)a1;
      v19 += 16;
      v22 = *(_QWORD *)(v21 + 8);
      *(_QWORD *)(v21 + 8) = *((_QWORD *)v20 + 1);
      v23 = v20[1];
      *((_QWORD *)v20 + 1) = v22;
      LODWORD(v22) = *(_DWORD *)(v21 + 4);
      *(_DWORD *)(v21 + 4) = v23;
      v24 = *v20;
      v20[1] = v22;
      LODWORD(v22) = *(_DWORD *)v21;
      *(_DWORD *)v21 = v24;
      *v20 = v22;
    }
    while ( 16 * v18 != v19 );
    v15 = *(unsigned int *)(a1 + 8);
    v17 = *(unsigned int *)(a2 + 8);
    v16 = *(_DWORD *)(a1 + 8);
  }
  if ( v17 >= v15 )
  {
    if ( v17 > v15 )
    {
      v28 = (const __m128i *)(*(_QWORD *)a2 + 16 * v18);
      v29 = (__m128i *)(*(_QWORD *)a1 + 16 * v15);
      if ( v28 == (const __m128i *)(*(_QWORD *)a2 + 16 * v17) )
      {
        v31 = v15;
      }
      else
      {
        v30 = &v29[v17 - v18];
        do
        {
          if ( v29 )
            *v29 = _mm_loadu_si128(v28);
          ++v29;
          ++v28;
        }
        while ( v29 != v30 );
        v31 = *(_DWORD *)(a1 + 8);
      }
      *(_DWORD *)(a1 + 8) = v31 + v17 - v15;
      *(_DWORD *)(a2 + 8) = v18;
    }
  }
  else
  {
    v25 = (const __m128i *)(*(_QWORD *)a1 + 16 * v18);
    v26 = (__m128i *)(*(_QWORD *)a2 + 16 * v17);
    if ( v25 != (const __m128i *)(*(_QWORD *)a1 + 16 * v15) )
    {
      v27 = &v26[v15 - v18];
      do
      {
        if ( v26 )
          *v26 = _mm_loadu_si128(v25);
        ++v26;
        ++v25;
      }
      while ( v26 != v27 );
      v16 = *(_DWORD *)(a2 + 8) + v15 - v17;
    }
    *(_DWORD *)(a2 + 8) = v16;
    *(_DWORD *)(a1 + 8) = v18;
  }
}
