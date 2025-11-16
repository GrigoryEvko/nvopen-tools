// Function: sub_356A490
// Address: 0x356a490
//
__int64 __fastcall sub_356A490(__int64 a1, _QWORD *a2)
{
  _QWORD *v2; // rax
  __m128i v4; // [rsp+0h] [rbp-30h] BYREF
  char v5; // [rsp+18h] [rbp-18h]

  v2 = sub_3569C80(a2, *a2 & 0xFFFFFFFFFFFFFFF8LL);
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)(a1 + 32) = v2;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 96) = 0;
  *(_QWORD *)(a1 + 104) = 0;
  *(_QWORD *)(a1 + 112) = 0;
  *(_QWORD *)(a1 + 16) = 0x100000008LL;
  *(_DWORD *)(a1 + 24) = 0;
  *(_QWORD *)a1 = 1;
  v4.m128i_i64[0] = (__int64)v2;
  v5 = 0;
  sub_3569710((unsigned __int64 *)(a1 + 96), &v4);
  return a1;
}
