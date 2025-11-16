// Function: sub_F7BA00
// Address: 0xf7ba00
//
__int64 __fastcall sub_F7BA00(unsigned __int64 **a1, unsigned int a2, unsigned int a3)
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
  __int64 v16; // r8
  const __m128i *v17; // r15
  __m128i *v18; // rax
  __int64 v19; // r9
  char *v20; // r15
  unsigned int v21; // [rsp+0h] [rbp-50h] BYREF
  __int64 v22; // [rsp+8h] [rbp-48h]
  __int64 v23; // [rsp+10h] [rbp-40h]

  v6 = (__int64)*a1;
  v7 = *((unsigned int *)*a1 + 2);
  v8 = **a1;
  v9 = *((unsigned int *)*a1 + 3);
  v10 = v8 + 24 * v7;
  if ( v7 >= v9 )
  {
    v16 = v7 + 1;
    v21 = a2;
    v17 = (const __m128i *)&v21;
    v22 = 0;
    v23 = 1;
    if ( v9 < v7 + 1 )
    {
      v19 = v6 + 16;
      if ( v8 > (unsigned __int64)&v21 || v10 <= (unsigned __int64)&v21 )
      {
        sub_C8D5F0(v6, (const void *)(v6 + 16), v7 + 1, 0x18u, v16, v19);
        v8 = *(_QWORD *)v6;
        v7 = *(unsigned int *)(v6 + 8);
      }
      else
      {
        v20 = (char *)&v21 - v8;
        sub_C8D5F0(v6, (const void *)(v6 + 16), v7 + 1, 0x18u, v16, v19);
        v8 = *(_QWORD *)v6;
        v7 = *(unsigned int *)(v6 + 8);
        v17 = (const __m128i *)&v20[*(_QWORD *)v6];
      }
    }
    v18 = (__m128i *)(v8 + 24 * v7);
    *v18 = _mm_loadu_si128(v17);
    v18[1].m128i_i64[0] = v17[1].m128i_i64[0];
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
  v14 = sub_D95540(*(_QWORD *)(*a1[2] + 40));
  return sub_DFD800(v12, a2, v14, v13, 0, 0, 0, 0, 0, 0) * a3;
}
