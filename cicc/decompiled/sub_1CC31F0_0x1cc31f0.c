// Function: sub_1CC31F0
// Address: 0x1cc31f0
//
__int64 __fastcall sub_1CC31F0(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        __m128i a4,
        __m128i a5,
        __m128i a6,
        double a7,
        double a8,
        __m128i a9,
        __m128 a10)
{
  __int64 *v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  double v14; // xmm4_8
  double v15; // xmm5_8
  unsigned int v16; // r12d
  __int64 v17; // rbx
  __int64 v18; // rdi
  _QWORD v20[42]; // [rsp+0h] [rbp-150h] BYREF

  v10 = *(__int64 **)(a1 + 8);
  memset(v20, 0, 0x128u);
  v20[7] = &v20[5];
  v20[8] = &v20[5];
  v20[13] = &v20[11];
  v20[14] = &v20[11];
  v20[19] = &v20[17];
  v20[20] = &v20[17];
  v20[25] = &v20[23];
  v20[26] = &v20[23];
  v20[31] = &v20[29];
  v20[32] = &v20[29];
  v11 = *v10;
  v12 = v10[1];
  if ( v11 == v12 )
LABEL_12:
    BUG();
  while ( *(_UNKNOWN **)v11 != &unk_4F9E06C )
  {
    v11 += 16;
    if ( v12 == v11 )
      goto LABEL_12;
  }
  v13 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v11 + 8) + 104LL))(*(_QWORD *)(v11 + 8), &unk_4F9E06C);
  v16 = sub_1CBFA40(v20, a2, v13 + 160, a3, a4, a5, a6, v14, v15, a9, a10);
  if ( v20[34] )
    j_j___libc_free_0(v20[34], v20[36] - v20[34]);
  sub_1CBB7A0(v20[30]);
  sub_1CBB5D0(v20[24]);
  sub_1CBB5D0(v20[18]);
  sub_1CBB5D0(v20[12]);
  v17 = v20[6];
  while ( v17 )
  {
    sub_1CBAD40(*(_QWORD *)(v17 + 24));
    v18 = v17;
    v17 = *(_QWORD *)(v17 + 16);
    j_j___libc_free_0(v18, 48);
  }
  if ( v20[0] )
    j_j___libc_free_0(v20[0], v20[2] - v20[0]);
  return v16;
}
