// Function: sub_22DE7C0
// Address: 0x22de7c0
//
__int64 __fastcall sub_22DE7C0(__int64 a1, _QWORD *a2)
{
  _QWORD *v2; // rax
  __m128i v4[2]; // [rsp+0h] [rbp-40h] BYREF
  char v5; // [rsp+20h] [rbp-20h]

  v2 = sub_22DE030(a2, *a2 & 0xFFFFFFFFFFFFFFF8LL);
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)(a1 + 32) = v2;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 96) = 0;
  *(_QWORD *)(a1 + 104) = 0;
  *(_QWORD *)(a1 + 112) = 0;
  *(_QWORD *)(a1 + 16) = 0x100000008LL;
  *(_DWORD *)(a1 + 24) = 0;
  *(_QWORD *)a1 = 1;
  v4[0].m128i_i64[0] = (__int64)v2;
  v5 = 0;
  sub_22DD390((unsigned __int64 *)(a1 + 96), v4);
  return a1;
}
