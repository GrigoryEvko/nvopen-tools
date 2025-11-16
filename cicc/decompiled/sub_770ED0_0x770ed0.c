// Function: sub_770ED0
// Address: 0x770ed0
//
void __fastcall sub_770ED0(_DWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __m128i a7, int a8)
{
  _DWORD *v8; // r13
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  __m128i v13; // [rsp-38h] [rbp-38h]
  int v14; // [rsp-28h] [rbp-28h]

  if ( (*(_BYTE *)(a2 + 132) & 0x20) == 0 )
  {
    v13 = _mm_loadu_si128(&a7);
    v14 = a8;
    v8 = sub_67D610(0xD36u, a1, 2u);
    sub_67F190((__int64)v8, (__int64)a1, v9, v10, v11, v12, v13.m128i_i8[0], v13.m128i_i64[1], v14);
    sub_67C730((_QWORD *)(a2 + 96), (__int64)v8);
    sub_770D30(a2);
  }
}
