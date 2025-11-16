// Function: sub_F7B720
// Address: 0xf7b720
//
__int64 __fastcall sub_F7B720(unsigned __int64 **a1, unsigned int a2, unsigned int a3)
{
  __int64 v6; // r13
  unsigned __int64 v7; // rdx
  unsigned __int64 v8; // rsi
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // rcx
  int v11; // eax
  __int64 v13; // r8
  const __m128i *v14; // r15
  __m128i *v15; // rax
  __int64 v16; // r9
  char *v17; // r15
  unsigned int v18; // [rsp+0h] [rbp-50h] BYREF
  __int64 v19; // [rsp+8h] [rbp-48h]
  __int64 v20; // [rsp+10h] [rbp-40h]

  v6 = (__int64)*a1;
  v7 = *((unsigned int *)*a1 + 2);
  v8 = **a1;
  v9 = *((unsigned int *)*a1 + 3);
  v10 = v8 + 24 * v7;
  if ( v7 >= v9 )
  {
    v13 = v7 + 1;
    v18 = a2;
    v14 = (const __m128i *)&v18;
    v19 = 0;
    v20 = 1;
    if ( v9 < v7 + 1 )
    {
      v16 = v6 + 16;
      if ( v8 > (unsigned __int64)&v18 || v10 <= (unsigned __int64)&v18 )
      {
        sub_C8D5F0(v6, (const void *)(v6 + 16), v7 + 1, 0x18u, v13, v16);
        v8 = *(_QWORD *)v6;
        v7 = *(unsigned int *)(v6 + 8);
      }
      else
      {
        v17 = (char *)&v18 - v8;
        sub_C8D5F0(v6, (const void *)(v6 + 16), v7 + 1, 0x18u, v13, v16);
        v8 = *(_QWORD *)v6;
        v7 = *(unsigned int *)(v6 + 8);
        v14 = (const __m128i *)&v17[*(_QWORD *)v6];
      }
    }
    v15 = (__m128i *)(v8 + 24 * v7);
    *v15 = _mm_loadu_si128(v14);
    v15[1].m128i_i64[0] = v14[1].m128i_i64[0];
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
  return sub_DFD800((__int64)a1[1], a2, *(_QWORD *)(*a1[2] + 40), *(_DWORD *)a1[3], 0, 0, 0, 0, 0, 0) * a3;
}
