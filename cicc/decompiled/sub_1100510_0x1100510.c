// Function: sub_1100510
// Address: 0x1100510
//
unsigned __int8 *__fastcall sub_1100510(__int64 a1, const __m128i *a2)
{
  __int64 v3; // rax
  unsigned int v4; // ebx
  __int64 **v5; // rdi
  __int64 v6; // rax
  __m128i v8[2]; // [rsp+0h] [rbp-70h] BYREF
  unsigned __int64 v9; // [rsp+20h] [rbp-50h]
  __int64 v10; // [rsp+28h] [rbp-48h]
  __m128i v11; // [rsp+30h] [rbp-40h]
  __int64 v12; // [rsp+40h] [rbp-30h]

  v3 = a2[10].m128i_i64[0];
  v4 = 8 * (*(_BYTE *)a1 != 70) + 256;
  v8[0] = _mm_loadu_si128(a2 + 6);
  v9 = _mm_loadu_si128(a2 + 8).m128i_u64[0];
  v12 = v3;
  v10 = a1;
  v8[1] = _mm_loadu_si128(a2 + 7);
  v11 = _mm_loadu_si128(a2 + 9);
  if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
    v5 = *(__int64 ***)(a1 - 8);
  else
    v5 = (__int64 **)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
  if ( ((unsigned int)sub_9B4030(*v5, v4, 0, v8) & v4) != 0 )
    return 0;
  v6 = sub_AD6530(*(_QWORD *)(a1 + 8), v4);
  return sub_F162A0((__int64)a2, a1, v6);
}
