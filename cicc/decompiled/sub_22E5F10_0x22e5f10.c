// Function: sub_22E5F10
// Address: 0x22e5f10
//
__int64 *__fastcall sub_22E5F10(__int64 *a1, _BYTE *a2, _QWORD *a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int8 *v7; // rsi

  if ( (*a3 & 4) != 0 )
  {
    *a1 = (__int64)(a1 + 2);
    sub_22E4AB0(a1, "Not implemented", (__int64)"");
    return a1;
  }
  else
  {
    v7 = (unsigned __int8 *)(*a3 & 0xFFFFFFFFFFFFFFF8LL);
    if ( *a2 )
      sub_11F3900((__int64)a1, v7);
    else
      sub_11F8430(
        a1,
        (__int64)v7,
        0,
        0,
        0,
        a6,
        (void (__fastcall *)(__int64, __int64 *, unsigned int *, __int64))sub_11F32A0,
        (__int64)sub_11F32F0);
    return a1;
  }
}
