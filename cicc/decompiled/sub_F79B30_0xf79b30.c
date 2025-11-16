// Function: sub_F79B30
// Address: 0xf79b30
//
__int64 __fastcall sub_F79B30(unsigned __int64 **a1, unsigned int a2)
{
  __int64 v4; // r12
  unsigned __int64 v5; // rdx
  unsigned __int64 v6; // rsi
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // rcx
  int v9; // eax
  __int64 *v10; // r14
  __int64 v11; // r12
  __int64 v12; // rax
  __int64 v14; // r8
  const __m128i *v15; // r14
  __m128i *v16; // rax
  __int64 v17; // r9
  char *v18; // r14
  unsigned int v19; // [rsp+0h] [rbp-50h] BYREF
  __int64 v20; // [rsp+8h] [rbp-48h]
  __int64 v21; // [rsp+10h] [rbp-40h]

  v4 = (__int64)*a1;
  v5 = *((unsigned int *)*a1 + 2);
  v6 = **a1;
  v7 = *((unsigned int *)*a1 + 3);
  v8 = v6 + 24 * v5;
  if ( v5 >= v7 )
  {
    v14 = v5 + 1;
    v19 = a2;
    v15 = (const __m128i *)&v19;
    v20 = 0;
    v21 = 0;
    if ( v7 < v5 + 1 )
    {
      v17 = v4 + 16;
      if ( v6 > (unsigned __int64)&v19 || v8 <= (unsigned __int64)&v19 )
      {
        sub_C8D5F0(v4, (const void *)(v4 + 16), v5 + 1, 0x18u, v14, v17);
        v6 = *(_QWORD *)v4;
        v5 = *(unsigned int *)(v4 + 8);
      }
      else
      {
        v18 = (char *)&v19 - v6;
        sub_C8D5F0(v4, (const void *)(v4 + 16), v5 + 1, 0x18u, v14, v17);
        v6 = *(_QWORD *)v4;
        v5 = *(unsigned int *)(v4 + 8);
        v15 = (const __m128i *)&v18[*(_QWORD *)v4];
      }
    }
    v16 = (__m128i *)(v6 + 24 * v5);
    *v16 = _mm_loadu_si128(v15);
    v16[1].m128i_i64[0] = v15[1].m128i_i64[0];
    ++*(_DWORD *)(v4 + 8);
  }
  else
  {
    v9 = v5;
    if ( v8 )
    {
      *(_DWORD *)v8 = a2;
      *(_QWORD *)(v8 + 8) = 0;
      *(_QWORD *)(v8 + 16) = 0;
      v9 = *(_DWORD *)(v4 + 8);
    }
    *(_DWORD *)(v4 + 8) = v9 + 1;
  }
  v10 = (__int64 *)a1[1];
  v11 = sub_D95540(*(_QWORD *)(*a1[2] + 32));
  v12 = sub_D95540(*(_QWORD *)(*a1[2] + 40));
  return sub_DFD060(v10, a2, v12, v11);
}
