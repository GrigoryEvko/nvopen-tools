// Function: sub_2266030
// Address: 0x2266030
//
void __fastcall sub_2266030(__int64 *a1, __m128i a2)
{
  __int64 v2; // r12
  __int64 v3[2]; // [rsp+0h] [rbp-40h] BYREF
  _QWORD v4[6]; // [rsp+10h] [rbp-30h] BYREF

  v2 = *a1;
  v3[0] = (__int64)v4;
  sub_2260190(v3, *(_BYTE **)(v2 + 96), *(_QWORD *)(v2 + 96) + *(_QWORD *)(v2 + 104));
  sub_2265950(v2, *(_DWORD *)(v2 + 1912), v2 + 128, v3, a2);
  if ( (_QWORD *)v3[0] != v4 )
    j_j___libc_free_0(v3[0]);
}
