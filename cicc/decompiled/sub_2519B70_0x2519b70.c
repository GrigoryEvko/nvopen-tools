// Function: sub_2519B70
// Address: 0x2519b70
//
__int64 *__fastcall sub_2519B70(__int64 a1, __int64 a2)
{
  bool v3; // zf
  __int64 *v4; // rax
  __int64 *result; // rax
  int v6; // ecx
  unsigned int v7; // esi
  int v8; // edx
  __m128i v9; // xmm0
  __int64 *v10; // [rsp+0h] [rbp-20h] BYREF
  __int64 *v11; // [rsp+8h] [rbp-18h] BYREF

  v3 = (unsigned __int8)sub_2512100(a1, (__int64 *)a2, &v10) == 0;
  v4 = v10;
  if ( !v3 )
    return v10 + 3;
  v6 = *(_DWORD *)(a1 + 16);
  v7 = *(_DWORD *)(a1 + 24);
  v11 = v10;
  ++*(_QWORD *)a1;
  v8 = v6 + 1;
  if ( 4 * (v6 + 1) >= 3 * v7 )
  {
    v7 *= 2;
  }
  else if ( v7 - *(_DWORD *)(a1 + 20) - v8 > v7 >> 3 )
  {
    goto LABEL_5;
  }
  sub_2519930(a1, v7);
  sub_2512100(a1, (__int64 *)a2, &v11);
  v8 = *(_DWORD *)(a1 + 16) + 1;
  v4 = v11;
LABEL_5:
  *(_DWORD *)(a1 + 16) = v8;
  if ( *v4 != -4096 || unk_4FEE4D0 != v4[1] || unk_4FEE4D8 != v4[2] )
    --*(_DWORD *)(a1 + 20);
  result = v4 + 3;
  *(result - 3) = *(_QWORD *)a2;
  v9 = _mm_loadu_si128((const __m128i *)(a2 + 8));
  *result = 0;
  *((__m128i *)result - 1) = v9;
  return result;
}
