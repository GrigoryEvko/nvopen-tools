// Function: sub_1EDCA90
// Address: 0x1edca90
//
void __fastcall sub_1EDCA90(__int64 a1, __int64 *a2, __int64 *a3, __int64 a4, int a5)
{
  __int64 *v5; // r9
  int v6; // r13d
  __int64 *v8; // r14
  __int64 v9; // r12
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rdx
  __int64 v13; // rbx
  __int64 v14; // r13
  __int64 v15; // rax
  __int64 v16; // rcx
  __int64 v17; // rdx
  __int64 v18; // rdx
  __int64 v19; // rcx
  __m128i *v20; // rax
  __int64 v21; // rdx
  __int64 v22; // [rsp+8h] [rbp-78h]
  __int64 *v23; // [rsp+28h] [rbp-58h]
  __m128i v24; // [rsp+30h] [rbp-50h] BYREF
  __int64 v25; // [rsp+40h] [rbp-40h]

  if ( (__int64 *)a1 != a2 )
  {
    v5 = (__int64 *)a2[8];
    v23 = &v5[*((unsigned int *)a2 + 18)];
    if ( v23 != v5 )
    {
      v6 = *(_DWORD *)(a1 + 72);
      v8 = (__int64 *)a2[8];
      do
      {
        v9 = *v8;
        v10 = sub_145CBF0(a3, 16, 16);
        v11 = *(_QWORD *)(v9 + 8);
        *(_DWORD *)v10 = v6;
        *(_QWORD *)(v10 + 8) = v11;
        v12 = *(unsigned int *)(a1 + 72);
        if ( (unsigned int)v12 >= *(_DWORD *)(a1 + 76) )
        {
          v22 = v10;
          sub_16CD150(a1 + 64, (const void *)(a1 + 80), 0, 8, a5, (int)v5);
          v12 = *(unsigned int *)(a1 + 72);
          v10 = v22;
        }
        ++v8;
        *(_QWORD *)(*(_QWORD *)(a1 + 64) + 8 * v12) = v10;
        v6 = *(_DWORD *)(a1 + 72) + 1;
        *(_DWORD *)(a1 + 72) = v6;
      }
      while ( v23 != v8 );
    }
    v13 = *a2;
    v14 = *a2 + 24LL * *((unsigned int *)a2 + 2);
    if ( v14 != *a2 )
    {
      v15 = *(unsigned int *)(a1 + 8);
      do
      {
        v16 = **(unsigned int **)(v13 + 16);
        v17 = *(_QWORD *)(a1 + 64);
        v24.m128i_i64[0] = *(_QWORD *)v13;
        v18 = *(_QWORD *)(v17 + 8 * v16);
        v19 = *(_QWORD *)(v13 + 8);
        v25 = v18;
        v24.m128i_i64[1] = v19;
        if ( *(_DWORD *)(a1 + 12) <= (unsigned int)v15 )
        {
          sub_16CD150(a1, (const void *)(a1 + 16), 0, 24, a5, (int)v5);
          v15 = *(unsigned int *)(a1 + 8);
        }
        v13 += 24;
        v20 = (__m128i *)(*(_QWORD *)a1 + 24 * v15);
        v21 = v25;
        *v20 = _mm_loadu_si128(&v24);
        v20[1].m128i_i64[0] = v21;
        v15 = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
        *(_DWORD *)(a1 + 8) = v15;
      }
      while ( v14 != v13 );
    }
  }
}
