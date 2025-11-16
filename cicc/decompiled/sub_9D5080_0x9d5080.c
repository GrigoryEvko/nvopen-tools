// Function: sub_9D5080
// Address: 0x9d5080
//
__int64 __fastcall sub_9D5080(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128i a7,
        const __m128i *a8,
        unsigned __int64 a9)
{
  __int64 v10; // rax
  _QWORD v11[7]; // [rsp+0h] [rbp-50h] BYREF
  char v12; // [rsp+38h] [rbp-18h]

  sub_9D45E0((__int64)v11, a2, a3, a4, a5, a6, a8, a9, a7);
  if ( (v12 & 1) != 0 )
  {
    v10 = v11[0];
    *(_BYTE *)(a1 + 24) |= 3u;
    *(_QWORD *)a1 = v10 & 0xFFFFFFFFFFFFFFFELL;
  }
  else
  {
    *(_BYTE *)(a1 + 24) = *(_BYTE *)(a1 + 24) & 0xFC | 2;
    *(_QWORD *)a1 = v11[0];
    *(_QWORD *)(a1 + 8) = v11[1];
    *(_QWORD *)(a1 + 16) = v11[2];
  }
  return a1;
}
