// Function: sub_1100B60
// Address: 0x1100b60
//
unsigned __int8 *__fastcall sub_1100B60(
        __m128i *a1,
        unsigned __int8 *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 *a6)
{
  unsigned __int8 *v6; // r12

  v6 = sub_11005E0(a1, a2, a3, a4, a5, a6);
  if ( v6 || sub_B44910((__int64)a2) || !(unsigned __int8)sub_9AC470(*((_QWORD *)a2 - 4), a1 + 6, 0) )
    return v6;
  sub_B448D0((__int64)a2, 1);
  return a2;
}
