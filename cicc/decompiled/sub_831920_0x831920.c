// Function: sub_831920
// Address: 0x831920
//
__int64 __fastcall sub_831920(
        unsigned __int8 a1,
        int a2,
        __m128i *a3,
        __m128i *a4,
        __m128i *a5,
        __int64 *a6,
        unsigned int a7,
        _QWORD *a8)
{
  __int64 v11; // [rsp-10h] [rbp-A0h]
  unsigned __int8 v13; // [rsp+17h] [rbp-79h]
  _BYTE v15[112]; // [rsp+20h] [rbp-70h] BYREF

  v13 = sub_6E9B70(a1, a2);
  sub_87A720(a1, v15, a6);
  *(_BYTE *)(qword_4D04A60[a1] + 73LL) |= 8u;
  if ( (unsigned __int8)(a1 - 37) <= 1u || a2 )
    return sub_7032B0(v13, a3, a5, a6, a7);
  sub_7038B0(v13, a3, a4, a5, a6, a7, a8);
  return v11;
}
