// Function: sub_3376050
// Address: 0x3376050
//
void __fastcall sub_3376050(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rax
  int v9; // edx
  int v10; // eax
  int v11; // edx
  int v12; // eax
  unsigned __int64 v13; // rdx
  unsigned __int64 v14; // r13
  int v15; // eax
  unsigned __int64 v16; // r15
  unsigned __int64 v17; // r14
  __int64 v18; // rax
  __m128i *v19; // rdx
  __int64 v20; // rcx
  __int16 v21; // di
  __int64 v22; // rsi
  const void *v23; // rsi
  int v24; // ecx
  const void *v25; // rsi

  if ( a1 == a2 )
    return;
  v8 = *(_QWORD *)a1;
  if ( a1 + 16 != *(_QWORD *)a1 && *(_QWORD *)a2 != a2 + 16 )
  {
    *(_QWORD *)a1 = *(_QWORD *)a2;
    v9 = *(_DWORD *)(a2 + 8);
    *(_QWORD *)a2 = v8;
    v10 = *(_DWORD *)(a1 + 8);
    *(_DWORD *)(a1 + 8) = v9;
    v11 = *(_DWORD *)(a2 + 12);
    *(_DWORD *)(a2 + 8) = v10;
    v12 = *(_DWORD *)(a1 + 12);
    *(_DWORD *)(a1 + 12) = v11;
    *(_DWORD *)(a2 + 12) = v12;
    return;
  }
  v13 = *(unsigned int *)(a2 + 8);
  if ( v13 > *(unsigned int *)(a1 + 12) )
  {
    sub_C8D5F0(a1, (const void *)(a1 + 16), v13, 0x10u, a5, a6);
    v14 = *(unsigned int *)(a1 + 8);
    v15 = v14;
    if ( *(_DWORD *)(a2 + 12) >= (unsigned int)v14 )
      goto LABEL_8;
    goto LABEL_22;
  }
  v14 = *(unsigned int *)(a1 + 8);
  v15 = v14;
  if ( *(_DWORD *)(a2 + 12) < (unsigned int)v14 )
  {
LABEL_22:
    sub_C8D5F0(a2, (const void *)(a2 + 16), v14, 0x10u, a5, a6);
    v14 = *(unsigned int *)(a1 + 8);
    v15 = *(_DWORD *)(a1 + 8);
  }
LABEL_8:
  v16 = *(unsigned int *)(a2 + 8);
  v17 = v14;
  if ( v16 <= v14 )
    v17 = *(unsigned int *)(a2 + 8);
  if ( v17 )
  {
    v18 = 0;
    do
    {
      v19 = (__m128i *)(v18 + *(_QWORD *)a2);
      v20 = v18 + *(_QWORD *)a1;
      v18 += 16;
      v21 = *(_WORD *)v20;
      v22 = *(_QWORD *)(v20 + 8);
      *(__m128i *)v20 = _mm_loadu_si128(v19);
      v19->m128i_i16[0] = v21;
      v19->m128i_i64[1] = v22;
    }
    while ( 16 * v17 != v18 );
    v14 = *(unsigned int *)(a1 + 8);
    v16 = *(unsigned int *)(a2 + 8);
    v15 = *(_DWORD *)(a1 + 8);
  }
  if ( v16 >= v14 )
  {
    if ( v16 > v14 )
    {
      v24 = v14;
      v25 = (const void *)(*(_QWORD *)a2 + 16 * v17);
      if ( v25 != (const void *)(16 * v16 + *(_QWORD *)a2) )
      {
        memcpy((void *)(*(_QWORD *)a1 + 16 * v14), v25, 16 * v16 - 16 * v17);
        v24 = *(_DWORD *)(a1 + 8);
      }
      *(_DWORD *)(a1 + 8) = v24 + v16 - v14;
      *(_DWORD *)(a2 + 8) = v17;
    }
  }
  else
  {
    v23 = (const void *)(*(_QWORD *)a1 + 16 * v17);
    if ( v23 != (const void *)(16 * v14 + *(_QWORD *)a1) )
    {
      memcpy((void *)(*(_QWORD *)a2 + 16 * v16), v23, 16 * v14 - 16 * v17);
      v15 = v14 + *(_DWORD *)(a2 + 8) - v16;
    }
    *(_DWORD *)(a2 + 8) = v15;
    *(_DWORD *)(a1 + 8) = v17;
  }
}
