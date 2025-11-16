// Function: sub_168FCF0
// Address: 0x168fcf0
//
__int64 __fastcall sub_168FCF0(__int64 a1, _BYTE *a2, __int64 a3)
{
  __int64 v3; // r14
  unsigned int v5; // r13d
  __m128i *v6; // rdi
  __int64 v7; // rax
  __m128i *v8; // rdi
  __int64 v9; // r12
  __int64 v10; // rax
  __m128i *v12; // [rsp+0h] [rbp-40h] BYREF
  __int64 v13; // [rsp+8h] [rbp-38h]
  __m128i v14[3]; // [rsp+10h] [rbp-30h] BYREF

  v3 = a1 + 328;
  v5 = *(_DWORD *)(a1 + 192);
  if ( a2 )
  {
    v12 = v14;
    sub_168F8E0((__int64 *)&v12, a2, (__int64)&a2[a3]);
  }
  else
  {
    v13 = 0;
    v12 = v14;
    v14[0].m128i_i8[0] = 0;
  }
  v6 = (__m128i *)sub_22077B0(48);
  v6[1].m128i_i64[0] = (__int64)v6[2].m128i_i64;
  if ( v12 == v14 )
  {
    v6[2] = _mm_load_si128(v14);
  }
  else
  {
    v6[1].m128i_i64[0] = (__int64)v12;
    v6[2].m128i_i64[0] = v14[0].m128i_i64[0];
  }
  v7 = v13;
  v12 = v14;
  v13 = 0;
  v6[1].m128i_i64[1] = v7;
  v14[0].m128i_i8[0] = 0;
  sub_2208C80(v6, v3);
  v8 = v12;
  ++*(_QWORD *)(a1 + 344);
  if ( v8 != v14 )
    j_j___libc_free_0(v8, v14[0].m128i_i64[0] + 1);
  v9 = *(_QWORD *)(*(_QWORD *)(a1 + 336) + 16LL);
  v10 = *(unsigned int *)(a1 + 192);
  if ( (unsigned int)v10 >= *(_DWORD *)(a1 + 196) )
  {
    sub_16CD150(a1 + 184, a1 + 200, 0, 8);
    v10 = *(unsigned int *)(a1 + 192);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 184) + 8 * v10) = v9;
  ++*(_DWORD *)(a1 + 192);
  return v5;
}
