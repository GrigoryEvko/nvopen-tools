// Function: sub_3376260
// Address: 0x3376260
//
void __fastcall sub_3376260(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  int v8; // edx
  int v9; // eax
  int v10; // edx
  int v11; // eax
  unsigned __int64 v12; // rdx
  unsigned __int64 v13; // r14
  int v14; // eax
  unsigned __int64 v15; // r15
  unsigned __int64 v16; // r13
  __int64 v17; // rax
  __int64 v18; // rdx
  __m128i *v19; // rcx
  __m128i v20; // xmm0
  __int64 v21; // rsi
  const void *v22; // rsi
  int v23; // ecx
  const void *v24; // rsi

  if ( a1 == a2 )
    return;
  v7 = *(_QWORD *)a1;
  if ( a1 + 16 != *(_QWORD *)a1 && *(_QWORD *)a2 != a2 + 16 )
  {
    *(_QWORD *)a1 = *(_QWORD *)a2;
    v8 = *(_DWORD *)(a2 + 8);
    *(_QWORD *)a2 = v7;
    v9 = *(_DWORD *)(a1 + 8);
    *(_DWORD *)(a1 + 8) = v8;
    v10 = *(_DWORD *)(a2 + 12);
    *(_DWORD *)(a2 + 8) = v9;
    v11 = *(_DWORD *)(a1 + 12);
    *(_DWORD *)(a1 + 12) = v10;
    *(_DWORD *)(a2 + 12) = v11;
    return;
  }
  v12 = *(unsigned int *)(a2 + 8);
  if ( v12 > *(unsigned int *)(a1 + 12) )
  {
    sub_C8D5F0(a1, (const void *)(a1 + 16), v12, 0x10u, a5, a6);
    v13 = *(unsigned int *)(a1 + 8);
    v14 = v13;
    if ( *(_DWORD *)(a2 + 12) >= (unsigned int)v13 )
      goto LABEL_8;
    goto LABEL_22;
  }
  v13 = *(unsigned int *)(a1 + 8);
  v14 = v13;
  if ( *(_DWORD *)(a2 + 12) < (unsigned int)v13 )
  {
LABEL_22:
    sub_C8D5F0(a2, (const void *)(a2 + 16), v13, 0x10u, a5, a6);
    v13 = *(unsigned int *)(a1 + 8);
    v14 = *(_DWORD *)(a1 + 8);
  }
LABEL_8:
  v15 = *(unsigned int *)(a2 + 8);
  v16 = v13;
  if ( v15 <= v13 )
    v16 = *(unsigned int *)(a2 + 8);
  if ( v16 )
  {
    v17 = 0;
    do
    {
      v18 = v17 + *(_QWORD *)a2;
      v19 = (__m128i *)(v17 + *(_QWORD *)a1);
      v17 += 16;
      v20 = _mm_loadu_si128(v19);
      v21 = v19->m128i_i64[0];
      v19->m128i_i64[0] = *(_QWORD *)v18;
      v19->m128i_i8[8] = *(_BYTE *)(v18 + 8);
      *(_QWORD *)v18 = v21;
      *(_BYTE *)(v18 + 8) = v20.m128i_i8[8];
    }
    while ( 16 * v16 != v17 );
    v13 = *(unsigned int *)(a1 + 8);
    v15 = *(unsigned int *)(a2 + 8);
    v14 = *(_DWORD *)(a1 + 8);
  }
  if ( v15 >= v13 )
  {
    if ( v15 > v13 )
    {
      v23 = v13;
      v24 = (const void *)(*(_QWORD *)a2 + 16 * v16);
      if ( v24 != (const void *)(16 * v15 + *(_QWORD *)a2) )
      {
        memcpy((void *)(*(_QWORD *)a1 + 16 * v13), v24, 16 * v15 - 16 * v16);
        v23 = *(_DWORD *)(a1 + 8);
      }
      *(_DWORD *)(a1 + 8) = v23 + v15 - v13;
      *(_DWORD *)(a2 + 8) = v16;
    }
  }
  else
  {
    v22 = (const void *)(*(_QWORD *)a1 + 16 * v16);
    if ( v22 != (const void *)(16 * v13 + *(_QWORD *)a1) )
    {
      memcpy((void *)(*(_QWORD *)a2 + 16 * v15), v22, 16 * v13 - 16 * v16);
      v14 = v13 + *(_DWORD *)(a2 + 8) - v15;
    }
    *(_DWORD *)(a2 + 8) = v14;
    *(_DWORD *)(a1 + 8) = v16;
  }
}
