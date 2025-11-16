// Function: sub_2F68500
// Address: 0x2f68500
//
void __fastcall sub_2F68500(__int64 a1, __int64 *a2, __int64 *a3, __int64 a4, __int64 a5)
{
  __int64 *v5; // r9
  int v6; // r13d
  __int64 *v8; // r14
  __int64 v9; // r12
  __int64 v10; // rax
  __int64 v11; // r9
  __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // rbx
  __int64 v15; // r14
  __int64 v16; // rax
  unsigned __int64 v17; // r9
  const __m128i *v18; // r12
  __int64 v19; // rcx
  __int64 v20; // rdx
  __int64 v21; // rdx
  unsigned __int64 v22; // rcx
  unsigned __int64 v23; // rdx
  __m128i *v24; // rax
  const void *v25; // rsi
  char *v26; // r12
  __int64 v27; // [rsp+8h] [rbp-78h]
  __int64 *v28; // [rsp+28h] [rbp-58h]
  _QWORD v29[10]; // [rsp+30h] [rbp-50h] BYREF

  if ( (__int64 *)a1 != a2 )
  {
    v5 = (__int64 *)a2[8];
    v28 = &v5[*((unsigned int *)a2 + 18)];
    if ( v28 != v5 )
    {
      v6 = *(_DWORD *)(a1 + 72);
      v8 = (__int64 *)a2[8];
      do
      {
        v9 = *v8;
        v10 = sub_A777F0(0x10u, a3);
        if ( v10 )
        {
          v12 = *(_QWORD *)(v9 + 8);
          *(_DWORD *)v10 = v6;
          *(_QWORD *)(v10 + 8) = v12;
        }
        v13 = *(unsigned int *)(a1 + 72);
        if ( v13 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 76) )
        {
          v27 = v10;
          sub_C8D5F0(a1 + 64, (const void *)(a1 + 80), v13 + 1, 8u, a5, v11);
          v13 = *(unsigned int *)(a1 + 72);
          v10 = v27;
        }
        ++v8;
        *(_QWORD *)(*(_QWORD *)(a1 + 64) + 8 * v13) = v10;
        v6 = *(_DWORD *)(a1 + 72) + 1;
        *(_DWORD *)(a1 + 72) = v6;
      }
      while ( v28 != v8 );
    }
    v14 = *a2;
    v15 = *a2 + 24LL * *((unsigned int *)a2 + 2);
    if ( v15 != *a2 )
    {
      v16 = *(unsigned int *)(a1 + 8);
      do
      {
        v17 = v16 + 1;
        v18 = (const __m128i *)v29;
        v19 = **(unsigned int **)(v14 + 16);
        v20 = *(_QWORD *)(a1 + 64);
        v29[0] = *(_QWORD *)v14;
        v21 = *(_QWORD *)(v20 + 8 * v19);
        v29[1] = *(_QWORD *)(v14 + 8);
        v22 = *(unsigned int *)(a1 + 12);
        v29[2] = v21;
        v23 = *(_QWORD *)a1;
        if ( v16 + 1 > v22 )
        {
          v25 = (const void *)(a1 + 16);
          if ( v23 > (unsigned __int64)v29 || (unsigned __int64)v29 >= v23 + 24 * v16 )
          {
            sub_C8D5F0(a1, v25, v17, 0x18u, a5, v17);
            v23 = *(_QWORD *)a1;
            v16 = *(unsigned int *)(a1 + 8);
          }
          else
          {
            v26 = (char *)v29 - v23;
            sub_C8D5F0(a1, v25, v17, 0x18u, a5, v17);
            v23 = *(_QWORD *)a1;
            v16 = *(unsigned int *)(a1 + 8);
            v18 = (const __m128i *)&v26[*(_QWORD *)a1];
          }
        }
        v14 += 24;
        v24 = (__m128i *)(v23 + 24 * v16);
        *v24 = _mm_loadu_si128(v18);
        v24[1].m128i_i64[0] = v18[1].m128i_i64[0];
        v16 = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
        *(_DWORD *)(a1 + 8) = v16;
      }
      while ( v15 != v14 );
    }
  }
}
