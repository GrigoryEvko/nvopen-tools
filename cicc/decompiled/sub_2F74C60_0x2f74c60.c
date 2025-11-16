// Function: sub_2F74C60
// Address: 0x2f74c60
//
__int64 __fastcall sub_2F74C60(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        int a7,
        __int64 a8,
        __int64 a9)
{
  unsigned int v10; // ecx
  _BYTE *v11; // r10
  unsigned int v12; // esi
  unsigned int v13; // eax
  __int64 v14; // rdi
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v18; // rax
  unsigned __int64 v19; // rdx
  __int64 v20; // rcx
  const __m128i *v21; // rdx
  __m128i *v22; // rax
  __int64 v23; // r13
  const void *v24; // rsi
  unsigned int v25; // [rsp+0h] [rbp-40h] BYREF
  __int64 v26; // [rsp+8h] [rbp-38h]
  __int64 v27; // [rsp+10h] [rbp-30h]

  v10 = a7;
  if ( a7 < 0 )
    v10 = *(_DWORD *)(a1 + 224) + (a7 & 0x7FFFFFFF);
  v11 = (_BYTE *)(*(_QWORD *)(a1 + 208) + v10);
  v12 = *(_DWORD *)(a1 + 8);
  v25 = v10;
  v13 = (unsigned __int8)*v11;
  v26 = a8;
  v27 = a9;
  if ( v13 >= v12 )
    goto LABEL_9;
  v14 = *(_QWORD *)a1;
  while ( 1 )
  {
    v15 = v14 + 24LL * v13;
    if ( v10 == *(_DWORD *)v15 )
      break;
    v13 += 256;
    if ( v12 <= v13 )
      goto LABEL_9;
  }
  if ( v15 == v14 + 24LL * v12 )
  {
LABEL_9:
    *v11 = v12;
    v18 = *(unsigned int *)(a1 + 8);
    v19 = v18 + 1;
    if ( v18 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
    {
      v23 = *(_QWORD *)a1;
      v24 = (const void *)(a1 + 16);
      if ( *(_QWORD *)a1 <= (unsigned __int64)&v25 && (unsigned __int64)&v25 < v23 + 24 * v18 )
      {
        sub_C8D5F0(a1, v24, v19, 0x18u, a9, a8);
        v20 = *(_QWORD *)a1;
        v18 = *(unsigned int *)(a1 + 8);
        v21 = (const __m128i *)((char *)&v25 + *(_QWORD *)a1 - v23);
      }
      else
      {
        sub_C8D5F0(a1, v24, v19, 0x18u, a9, a8);
        v20 = *(_QWORD *)a1;
        v18 = *(unsigned int *)(a1 + 8);
        v21 = (const __m128i *)&v25;
      }
    }
    else
    {
      v20 = *(_QWORD *)a1;
      v21 = (const __m128i *)&v25;
    }
    v22 = (__m128i *)(v20 + 24 * v18);
    *v22 = _mm_loadu_si128(v21);
    v22[1].m128i_i64[0] = v21[1].m128i_i64[0];
    ++*(_DWORD *)(a1 + 8);
    return 0;
  }
  else
  {
    v16 = *(_QWORD *)(v15 + 8);
    *(_QWORD *)(v15 + 16) |= a9;
    *(_QWORD *)(v15 + 8) = v16 | a8;
    return v16;
  }
}
