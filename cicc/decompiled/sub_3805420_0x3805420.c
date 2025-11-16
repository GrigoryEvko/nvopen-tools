// Function: sub_3805420
// Address: 0x3805420
//
unsigned __int8 *__fastcall sub_3805420(__int64 a1, __int64 a2, __m128i a3)
{
  __int64 v4; // rsi
  __int64 v5; // r13
  int v6; // eax
  __int64 v7; // rbx
  __int64 *v8; // rsi
  unsigned __int8 *v9; // r12
  unsigned __int64 v11; // [rsp+0h] [rbp-40h] BYREF
  unsigned int v12; // [rsp+8h] [rbp-38h]
  __int64 v13; // [rsp+10h] [rbp-30h] BYREF
  int v14; // [rsp+18h] [rbp-28h]

  v4 = *(_QWORD *)(a2 + 80);
  v5 = *(_QWORD *)(a1 + 8);
  v13 = v4;
  if ( v4 )
    sub_B96E90((__int64)&v13, v4, 1);
  v6 = *(_DWORD *)(a2 + 72);
  v7 = *(_QWORD *)(a2 + 96);
  v14 = v6;
  v8 = (__int64 *)(v7 + 24);
  if ( *(void **)(v7 + 24) == sub_C33340() )
    sub_C3E660((__int64)&v11, (__int64)v8);
  else
    sub_C3A850((__int64)&v11, v8);
  v9 = sub_34007B0(v5, (__int64)&v11, (__int64)&v13, 6u, 0, 0, a3, 0);
  if ( v12 > 0x40 && v11 )
    j_j___libc_free_0_0(v11);
  if ( v13 )
    sub_B91220((__int64)&v13, v13);
  return v9;
}
