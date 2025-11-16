// Function: sub_1DBFD90
// Address: 0x1dbfd90
//
void __fastcall sub_1DBFD90(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 v8; // rax
  int v9; // edx
  int v10; // eax
  int v11; // edx
  int v12; // eax
  unsigned __int64 v13; // rdx
  unsigned __int64 v14; // r14
  int v15; // eax
  unsigned __int64 v16; // r15
  unsigned __int64 v17; // r13
  __int64 v18; // rcx
  __m128i *v19; // rax
  __int64 *v20; // rdx
  __int64 v21; // r8
  __int64 v22; // rdi
  __int64 v23; // rsi
  int v24; // esi

  if ( a1 == a2 )
    return;
  v8 = *(_QWORD *)a1;
  if ( *(_QWORD *)a1 != a1 + 16 && *(_QWORD *)a2 != a2 + 16 )
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
  if ( *(_DWORD *)(a1 + 12) < (unsigned int)v13 )
  {
    sub_16CD150(a1, (const void *)(a1 + 16), v13, 24, a5, a6);
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
    sub_16CD150(a2, (const void *)(a2 + 16), v14, 24, a5, a6);
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
      v20 = (__int64 *)(v18 + *(_QWORD *)a1);
      v18 += 24;
      v21 = *v20;
      v22 = v20[1];
      v23 = v20[2];
      *(__m128i *)v20 = _mm_loadu_si128(v19);
      v20[2] = v19[1].m128i_i64[0];
      v19->m128i_i64[0] = v21;
      v19->m128i_i64[1] = v22;
      v19[1].m128i_i64[0] = v23;
    }
    while ( 24 * v17 != v18 );
    v14 = *(unsigned int *)(a1 + 8);
    v16 = *(unsigned int *)(a2 + 8);
    v15 = *(_DWORD *)(a1 + 8);
  }
  if ( v16 >= v14 )
  {
    if ( v16 > v14 )
    {
      v24 = v14;
      if ( *(_QWORD *)a2 + 24 * v17 != 24 * v16 + *(_QWORD *)a2 )
      {
        memcpy((void *)(*(_QWORD *)a1 + 24 * v14), (const void *)(*(_QWORD *)a2 + 24 * v17), 24 * v16 - 24 * v17);
        v24 = *(_DWORD *)(a1 + 8);
      }
      *(_DWORD *)(a1 + 8) = v24 + v16 - v14;
      *(_DWORD *)(a2 + 8) = v17;
    }
  }
  else
  {
    if ( *(_QWORD *)a1 + 24 * v17 != 24 * v14 + *(_QWORD *)a1 )
    {
      memcpy((void *)(*(_QWORD *)a2 + 24 * v16), (const void *)(*(_QWORD *)a1 + 24 * v17), 24 * v14 - 24 * v17);
      v15 = v14 + *(_DWORD *)(a2 + 8) - v16;
    }
    *(_DWORD *)(a2 + 8) = v15;
    *(_DWORD *)(a1 + 8) = v17;
  }
}
