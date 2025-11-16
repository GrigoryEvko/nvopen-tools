// Function: sub_F799E0
// Address: 0xf799e0
//
__int64 __fastcall sub_F799E0(unsigned __int64 **a1, unsigned int a2)
{
  __int64 v4; // r12
  unsigned __int64 v5; // rdx
  unsigned __int64 v6; // rsi
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // rcx
  int v9; // eax
  __int64 *v10; // r12
  __int64 v11; // rax
  __int64 v13; // r8
  const __m128i *v14; // r14
  __m128i *v15; // rax
  __int64 v16; // r9
  char *v17; // r14
  unsigned int v18; // [rsp+0h] [rbp-40h] BYREF
  __int64 v19; // [rsp+8h] [rbp-38h]
  __int64 v20; // [rsp+10h] [rbp-30h]

  v4 = (__int64)*a1;
  v5 = *((unsigned int *)*a1 + 2);
  v6 = **a1;
  v7 = *((unsigned int *)*a1 + 3);
  v8 = v6 + 24 * v5;
  if ( v5 >= v7 )
  {
    v13 = v5 + 1;
    v18 = a2;
    v14 = (const __m128i *)&v18;
    v19 = 0;
    v20 = 0;
    if ( v7 < v5 + 1 )
    {
      v16 = v4 + 16;
      if ( v6 > (unsigned __int64)&v18 || v8 <= (unsigned __int64)&v18 )
      {
        sub_C8D5F0(v4, (const void *)(v4 + 16), v5 + 1, 0x18u, v13, v16);
        v6 = *(_QWORD *)v4;
        v5 = *(unsigned int *)(v4 + 8);
      }
      else
      {
        v17 = (char *)&v18 - v6;
        sub_C8D5F0(v4, (const void *)(v4 + 16), v5 + 1, 0x18u, v13, v16);
        v6 = *(_QWORD *)v4;
        v5 = *(unsigned int *)(v4 + 8);
        v14 = (const __m128i *)&v17[*(_QWORD *)v4];
      }
    }
    v15 = (__m128i *)(v6 + 24 * v5);
    *v15 = _mm_loadu_si128(v14);
    v15[1].m128i_i64[0] = v14[1].m128i_i64[0];
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
  return sub_DFD060(v10, a2, *(_QWORD *)(*a1[2] + 40), v11);
}
