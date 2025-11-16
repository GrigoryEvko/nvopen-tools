// Function: sub_1880940
// Address: 0x1880940
//
__int64 __fastcall sub_1880940(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rdx
  __int64 v4; // r14
  __int64 v5; // r8
  unsigned int v6; // r15d
  __m128i *v7; // rax
  const void *v8; // rdx
  __m128i *v9; // rbx
  __int64 v10; // rdx
  unsigned int v11; // eax
  __int64 v12; // [rsp+8h] [rbp-38h]

  result = sub_12539F0(a1, a2);
  if ( v3 )
  {
    v4 = v3;
    v5 = a1 + 8;
    v6 = 1;
    if ( !result && v3 != v5 )
    {
      v11 = sub_1872D20(*(const void **)a2, *(_QWORD *)(a2 + 8), *(const void **)(v3 + 32), *(_QWORD *)(v3 + 40));
      v5 = a1 + 8;
      v6 = v11 >> 31;
    }
    v12 = v5;
    v7 = (__m128i *)sub_22077B0(64);
    v8 = *(const void **)a2;
    v9 = v7;
    v7[2].m128i_i64[0] = (__int64)v7[3].m128i_i64;
    if ( v8 == (const void *)(a2 + 16) )
    {
      v7[3] = _mm_loadu_si128((const __m128i *)(a2 + 16));
    }
    else
    {
      v7[2].m128i_i64[0] = (__int64)v8;
      v7[3].m128i_i64[0] = *(_QWORD *)(a2 + 16);
    }
    v10 = *(_QWORD *)(a2 + 8);
    *(_QWORD *)a2 = a2 + 16;
    *(_QWORD *)(a2 + 8) = 0;
    *(_BYTE *)(a2 + 16) = 0;
    v7[2].m128i_i64[1] = v10;
    sub_220F040(v6, v7, v4, v12);
    ++*(_QWORD *)(a1 + 40);
    return (__int64)v9;
  }
  return result;
}
