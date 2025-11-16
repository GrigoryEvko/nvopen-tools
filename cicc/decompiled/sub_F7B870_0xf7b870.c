// Function: sub_F7B870
// Address: 0xf7b870
//
unsigned __int64 __fastcall sub_F7B870(unsigned __int64 **a1, unsigned int a2, unsigned int a3)
{
  __int64 v6; // r13
  unsigned __int64 v7; // rdx
  unsigned __int64 v8; // rsi
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // rcx
  int v11; // eax
  __int64 v12; // r13
  unsigned int v13; // r15d
  __int64 v14; // rax
  signed __int64 v15; // rax
  __int64 v16; // r8
  __int64 v18; // r8
  const __m128i *v19; // r15
  __m128i *v20; // rax
  __int64 v21; // r9
  char *v22; // r15
  unsigned int v23; // [rsp+0h] [rbp-50h] BYREF
  __int64 v24; // [rsp+8h] [rbp-48h]
  __int64 v25; // [rsp+10h] [rbp-40h]

  v6 = (__int64)*a1;
  v7 = *((unsigned int *)*a1 + 2);
  v8 = **a1;
  v9 = *((unsigned int *)*a1 + 3);
  v10 = v8 + 24 * v7;
  if ( v7 >= v9 )
  {
    v18 = v7 + 1;
    v23 = a2;
    v19 = (const __m128i *)&v23;
    v24 = 0;
    v25 = 1;
    if ( v9 < v7 + 1 )
    {
      v21 = v6 + 16;
      if ( v8 > (unsigned __int64)&v23 || v10 <= (unsigned __int64)&v23 )
      {
        sub_C8D5F0(v6, (const void *)(v6 + 16), v7 + 1, 0x18u, v18, v21);
        v8 = *(_QWORD *)v6;
        v7 = *(unsigned int *)(v6 + 8);
      }
      else
      {
        v22 = (char *)&v23 - v8;
        sub_C8D5F0(v6, (const void *)(v6 + 16), v7 + 1, 0x18u, v18, v21);
        v8 = *(_QWORD *)v6;
        v7 = *(unsigned int *)(v6 + 8);
        v19 = (const __m128i *)&v22[*(_QWORD *)v6];
      }
    }
    v20 = (__m128i *)(v8 + 24 * v7);
    *v20 = _mm_loadu_si128(v19);
    v20[1].m128i_i64[0] = v19[1].m128i_i64[0];
    ++*(_DWORD *)(v6 + 8);
  }
  else
  {
    v11 = v7;
    if ( v10 )
    {
      *(_DWORD *)v10 = a2;
      *(_QWORD *)(v10 + 8) = 0;
      *(_QWORD *)(v10 + 16) = 1;
      v11 = *(_DWORD *)(v6 + 8);
    }
    *(_DWORD *)(v6 + 8) = v11 + 1;
  }
  v12 = (__int64)a1[1];
  v13 = *(_DWORD *)a1[3];
  v14 = sub_D95540(*a1[2]);
  v15 = sub_DFD800(v12, a2, v14, v13, 0, 0, 0, 0, 0, 0);
  v16 = v15 * a3;
  if ( !is_mul_ok(v15, a3) )
  {
    if ( !a3 )
      return 0x8000000000000000LL;
    v16 = 0x7FFFFFFFFFFFFFFFLL;
    if ( v15 <= 0 )
      return 0x8000000000000000LL;
  }
  return v16;
}
