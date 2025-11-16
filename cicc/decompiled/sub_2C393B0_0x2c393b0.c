// Function: sub_2C393B0
// Address: 0x2c393b0
//
_OWORD *__fastcall sub_2C393B0(__int64 a1, __m128i *a2)
{
  _OWORD *result; // rax
  __int64 v4; // r9
  unsigned int v5; // esi
  int v6; // eax
  __int64 *v7; // rdx
  int v8; // eax
  __int64 v9; // r8
  __int64 v10; // rax
  __m128i v11; // xmm0
  __m128i v12; // [rsp+0h] [rbp-30h] BYREF
  __int64 *v13; // [rsp+10h] [rbp-20h] BYREF
  __int64 *v14; // [rsp+18h] [rbp-18h] BYREF

  result = (_OWORD *)sub_2C2C000(a1, a2->m128i_i64, &v13);
  if ( (_BYTE)result )
    return result;
  v5 = *(_DWORD *)(a1 + 24);
  v6 = *(_DWORD *)(a1 + 16);
  v7 = v13;
  ++*(_QWORD *)a1;
  v8 = v6 + 1;
  v9 = 2 * v5;
  v14 = v7;
  if ( 4 * v8 >= 3 * v5 )
  {
    v5 *= 2;
  }
  else if ( v5 - *(_DWORD *)(a1 + 20) - v8 > v5 >> 3 )
  {
    goto LABEL_4;
  }
  sub_2C391F0(a1, v5);
  sub_2C2C000(a1, a2->m128i_i64, &v14);
  v7 = v14;
  v8 = *(_DWORD *)(a1 + 16) + 1;
LABEL_4:
  *(_DWORD *)(a1 + 16) = v8;
  if ( *v7 != -4096 || v7[1] != -4096 )
    --*(_DWORD *)(a1 + 20);
  *v7 = a2->m128i_i64[0];
  v7[1] = a2->m128i_i64[1];
  v10 = *(unsigned int *)(a1 + 40);
  v11 = _mm_loadu_si128(a2);
  if ( v10 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
  {
    v12 = v11;
    sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), v10 + 1, 0x10u, v9, v4);
    v10 = *(unsigned int *)(a1 + 40);
    v11 = _mm_load_si128(&v12);
  }
  result = (_OWORD *)(*(_QWORD *)(a1 + 32) + 16 * v10);
  *result = v11;
  ++*(_DWORD *)(a1 + 40);
  return result;
}
