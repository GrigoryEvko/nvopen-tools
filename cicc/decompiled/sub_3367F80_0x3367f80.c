// Function: sub_3367F80
// Address: 0x3367f80
//
__int64 __fastcall sub_3367F80(__int64 a1, __int64 *a2)
{
  __int64 v3; // rdx
  __int64 result; // rax
  __int64 v5; // rax
  __int64 v6; // rbx
  __int64 v7; // rax
  unsigned __int16 v8; // dx
  __int64 v9; // rax
  __int64 v10; // r10
  __int64 v11; // rdx
  char v12; // r9
  unsigned __int64 v13; // rdx
  unsigned __int64 v14; // rsi
  unsigned __int64 v15; // rdi
  int v16; // eax
  unsigned __int64 v17; // rcx
  const __m128i *v18; // rbx
  const __m128i *i; // r13
  __m128i v20; // xmm0
  __int64 v21; // rax
  __int64 v22; // r8
  const __m128i *v23; // rbx
  __int64 v24; // r9
  __int8 *v25; // rbx
  unsigned __int16 v26; // [rsp+0h] [rbp-60h] BYREF
  __int64 v27; // [rsp+8h] [rbp-58h]
  __int64 v28; // [rsp+10h] [rbp-50h]
  __int64 v29; // [rsp+18h] [rbp-48h]
  __m128i v30; // [rsp+20h] [rbp-40h] BYREF
  char v31; // [rsp+30h] [rbp-30h]

  while ( 1 )
  {
    v3 = *a2;
    result = *(unsigned int *)(*a2 + 24);
    if ( (_DWORD)result == 54 )
      break;
    if ( (int)result > 54 )
    {
      if ( (_DWORD)result != 216 )
      {
        if ( (int)result <= 216 )
        {
          if ( (_DWORD)result != 156 && (_DWORD)result != 159 )
            return result;
          break;
        }
        if ( (_DWORD)result != 234 )
          return result;
      }
    }
    else
    {
      if ( (int)result > 4 )
      {
        if ( (_DWORD)result == 50 )
        {
          v5 = *(_QWORD *)(v3 + 40);
          v6 = *(_QWORD *)(v5 + 40);
          v7 = *(_QWORD *)(v6 + 48) + 16LL * *(unsigned int *)(v5 + 48);
          v8 = *(_WORD *)v7;
          v9 = *(_QWORD *)(v7 + 8);
          v26 = v8;
          v27 = v9;
          if ( v8 )
          {
            if ( v8 == 1 || (unsigned __int16)(v8 - 504) <= 7u )
              BUG();
            v21 = 16LL * (v8 - 1);
            v10 = *(_QWORD *)&byte_444C4A0[v21];
            v12 = byte_444C4A0[v21 + 8];
          }
          else
          {
            v28 = sub_3007260((__int64)&v26);
            v10 = v28;
            v29 = v11;
            v12 = v11;
          }
          v13 = *(unsigned int *)(a1 + 8);
          v14 = *(_QWORD *)a1;
          v15 = *(unsigned int *)(a1 + 12);
          v16 = *(_DWORD *)(a1 + 8);
          v17 = *(_QWORD *)a1 + 24 * v13;
          if ( v13 >= v15 )
          {
            v30.m128i_i32[0] = *(_DWORD *)(v6 + 96);
            v22 = v13 + 1;
            v23 = &v30;
            v30.m128i_i64[1] = v10;
            v31 = v12;
            if ( v15 < v13 + 1 )
            {
              v24 = a1 + 16;
              if ( v14 > (unsigned __int64)&v30 || v17 <= (unsigned __int64)&v30 )
              {
                sub_C8D5F0(a1, (const void *)(a1 + 16), v13 + 1, 0x18u, v22, v24);
                v14 = *(_QWORD *)a1;
                v13 = *(unsigned int *)(a1 + 8);
              }
              else
              {
                v25 = &v30.m128i_i8[-v14];
                sub_C8D5F0(a1, (const void *)(a1 + 16), v13 + 1, 0x18u, v22, v24);
                v14 = *(_QWORD *)a1;
                v13 = *(unsigned int *)(a1 + 8);
                v23 = (const __m128i *)&v25[*(_QWORD *)a1];
              }
            }
            result = v14 + 24 * v13;
            *(__m128i *)result = _mm_loadu_si128(v23);
            *(_QWORD *)(result + 16) = v23[1].m128i_i64[0];
            ++*(_DWORD *)(a1 + 8);
          }
          else
          {
            if ( v17 )
            {
              *(_DWORD *)v17 = *(_DWORD *)(v6 + 96);
              *(_QWORD *)(v17 + 8) = v10;
              *(_BYTE *)(v17 + 16) = v12;
              v16 = *(_DWORD *)(a1 + 8);
            }
            result = (unsigned int)(v16 + 1);
            *(_DWORD *)(a1 + 8) = result;
          }
        }
        return result;
      }
      if ( (int)result <= 2 )
        return result;
    }
    a2 = *(__int64 **)(v3 + 40);
  }
  v18 = *(const __m128i **)(v3 + 40);
  result = 5LL * *(unsigned int *)(v3 + 64);
  for ( i = (const __m128i *)((char *)v18 + 40 * *(unsigned int *)(v3 + 64)); i != v18; result = sub_3367F80(a1, &v30) )
  {
    v20 = _mm_loadu_si128(v18);
    v18 = (const __m128i *)((char *)v18 + 40);
    v30 = v20;
  }
  return result;
}
