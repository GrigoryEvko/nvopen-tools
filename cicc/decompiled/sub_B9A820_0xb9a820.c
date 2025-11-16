// Function: sub_B9A820
// Address: 0xb9a820
//
void __fastcall sub_B9A820(unsigned int **a1, __int64 a2)
{
  unsigned __int64 v2; // r14
  unsigned int *v3; // r12
  __int64 v4; // rdx
  __int64 v5; // rbx
  unsigned __int64 v6; // rcx
  __int64 v7; // rax
  __int64 v8; // rax
  unsigned __int64 v9; // r9
  unsigned __int64 v10; // r8
  __int64 v11; // rdx
  unsigned __int64 *v12; // rax
  __m128i *v13; // r14
  __int64 v14; // rdx
  unsigned int *v15; // rbx
  __int64 v16; // r12
  __int64 v17; // r15
  __int64 v18; // rax
  int *v19; // r13
  __int64 v20; // rdx
  __int64 v21; // rax
  __m128i v22; // xmm0
  __int64 v23; // rax
  unsigned __int64 v24; // [rsp+0h] [rbp-40h]
  unsigned __int64 v25; // [rsp+8h] [rbp-38h]

  v3 = *a1;
  v4 = *(unsigned int *)(a2 + 8);
  v5 = (__int64)&(*a1)[4 * *((unsigned int *)a1 + 2)];
  if ( (unsigned int *)v5 != *a1 )
  {
    do
    {
      v6 = *(unsigned int *)(a2 + 12);
      v7 = (unsigned int)v4;
      if ( (unsigned int)v4 >= v6 )
      {
        v9 = *((_QWORD *)v3 + 1);
        v10 = *v3 | v2 & 0xFFFFFFFF00000000LL;
        v11 = (unsigned int)v4 + 1LL;
        v2 = v10;
        if ( v6 < v7 + 1 )
        {
          v24 = v10;
          v25 = *((_QWORD *)v3 + 1);
          sub_C8D5F0(a2, a2 + 16, v11, 16);
          v7 = *(unsigned int *)(a2 + 8);
          v10 = v24;
          v9 = v25;
        }
        v12 = (unsigned __int64 *)(*(_QWORD *)a2 + 16 * v7);
        *v12 = v10;
        v12[1] = v9;
        v4 = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
        *(_DWORD *)(a2 + 8) = v4;
      }
      else
      {
        v8 = *(_QWORD *)a2 + 16LL * (unsigned int)v4;
        if ( v8 )
        {
          *(_DWORD *)v8 = *v3;
          *(_QWORD *)(v8 + 8) = *((_QWORD *)v3 + 1);
          LODWORD(v4) = *(_DWORD *)(a2 + 8);
        }
        v4 = (unsigned int)(v4 + 1);
        *(_DWORD *)(a2 + 8) = v4;
      }
      v3 += 4;
    }
    while ( (unsigned int *)v5 != v3 );
  }
  if ( (unsigned int)v4 > 1 )
  {
    v13 = *(__m128i **)a2;
    v14 = 16 * v4;
    v15 = (unsigned int *)(*(_QWORD *)a2 + v14);
    v16 = v14 >> 4;
    while ( 1 )
    {
      v17 = 4 * v16;
      v18 = sub_2207800(16 * v16, &unk_435FF63);
      v19 = (int *)v18;
      if ( v18 )
        break;
      v16 >>= 1;
      if ( !v16 )
      {
        v17 = 0;
        sub_B8FDC0(v13->m128i_i8, v15);
        goto LABEL_18;
      }
    }
    v20 = v18 + v17 * 4;
    v21 = v18 + 16;
    *(__m128i *)(v21 - 16) = _mm_loadu_si128(v13);
    if ( v20 == v21 )
    {
      v23 = (__int64)v19;
    }
    else
    {
      do
      {
        v22 = _mm_loadu_si128((const __m128i *)(v21 - 16));
        v21 += 16;
        *(__m128i *)(v21 - 16) = v22;
      }
      while ( v20 != v21 );
      v23 = (__int64)&v19[v17 - 4];
    }
    v13->m128i_i32[0] = *(_DWORD *)v23;
    v13->m128i_i64[1] = *(_QWORD *)(v23 + 8);
    sub_B9A750(v13->m128i_i32, v15, v19, v16);
LABEL_18:
    j_j___libc_free_0(v19, v17 * 4);
  }
}
