// Function: sub_15A85E0
// Address: 0x15a85e0
//
__int64 __fastcall sub_15A85E0(__int64 a1, unsigned int a2, unsigned int a3, unsigned int a4, int a5, int a6)
{
  __int64 v8; // rax
  int v9; // ecx
  __m128i *v10; // r12
  __int64 result; // rax
  __m128i *v12; // r13
  unsigned __int64 v13; // rcx
  __int64 v14; // rdx
  unsigned int v15; // eax
  __int64 v16; // rdi
  __m128i *v17; // rsi
  const __m128i *v18; // rcx
  __int64 v19; // rax
  __int8 *v20; // r12
  __int64 v24; // [rsp+8h] [rbp-58h]
  __m128i v25; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v26; // [rsp+20h] [rbp-40h]
  char v27; // [rsp+24h] [rbp-3Ch] BYREF

  if ( a4 < a3 )
    sub_16BD130("Preferred alignment cannot be less than the ABI alignment", 1);
  v8 = sub_15A8580(a1, a2);
  v9 = a4;
  v10 = (__m128i *)v8;
  result = *(_QWORD *)(a1 + 224) + 20LL * *(unsigned int *)(a1 + 232);
  if ( v10 != (__m128i *)result && v10->m128i_i32[3] == a2 )
  {
    v10->m128i_i32[0] = a3;
    v10->m128i_i32[1] = a4;
    v10->m128i_i32[2] = a5;
    v10[1].m128i_i32[0] = a6;
    return result;
  }
  v12 = &v25;
  v24 = a1 + 224;
  sub_15A80E0(&v25, a2, a3, v9, a5, a6);
  v13 = *(unsigned int *)(a1 + 232);
  v14 = *(_QWORD *)(a1 + 224);
  v15 = *(_DWORD *)(a1 + 232);
  v16 = 20 * v13;
  v17 = (__m128i *)(v14 + 20 * v13);
  if ( v10 != v17 )
  {
    if ( v13 >= *(unsigned int *)(a1 + 236) )
    {
      v20 = &v10->m128i_i8[-v14];
      sub_16CD150(v24, a1 + 240, 0, 20);
      v14 = *(_QWORD *)(a1 + 224);
      v15 = *(_DWORD *)(a1 + 232);
      v10 = (__m128i *)&v20[v14];
      v16 = 20LL * v15;
      v17 = (__m128i *)(v14 + v16);
      v18 = (const __m128i *)(v14 + v16 - 20);
      if ( !(v14 + v16) )
        goto LABEL_8;
    }
    else
    {
      v18 = (const __m128i *)(v14 + v16 - 20);
      if ( !v17 )
      {
LABEL_8:
        if ( v10 != v18 )
        {
          memmove((void *)(v14 + v16 - ((char *)v18 - (char *)v10)), v10, (char *)v18 - (char *)v10);
          v15 = *(_DWORD *)(a1 + 232);
        }
        v19 = v15 + 1;
        *(_DWORD *)(a1 + 232) = v19;
        if ( v10 <= &v25 && (unsigned __int64)&v25 < *(_QWORD *)(a1 + 224) + 20 * v19 )
          v12 = (__m128i *)&v27;
        result = v12[1].m128i_u32[0];
        *v10 = _mm_loadu_si128(v12);
        v10[1].m128i_i32[0] = result;
        return result;
      }
    }
    *v17 = _mm_loadu_si128(v18);
    v17[1].m128i_i32[0] = v18[1].m128i_i32[0];
    v14 = *(_QWORD *)(a1 + 224);
    v15 = *(_DWORD *)(a1 + 232);
    v16 = 20LL * v15;
    v18 = (const __m128i *)(v14 + v16 - 20);
    goto LABEL_8;
  }
  if ( (unsigned int)v13 >= *(_DWORD *)(a1 + 236) )
  {
    sub_16CD150(v24, a1 + 240, 0, 20);
    v17 = (__m128i *)(*(_QWORD *)(a1 + 224) + 20LL * *(unsigned int *)(a1 + 232));
  }
  result = v26;
  *v17 = _mm_loadu_si128(&v25);
  v17[1].m128i_i32[0] = result;
  ++*(_DWORD *)(a1 + 232);
  return result;
}
