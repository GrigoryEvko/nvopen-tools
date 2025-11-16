// Function: sub_20CD3A0
// Address: 0x20cd3a0
//
__int64 __fastcall sub_20CD3A0(__int64 a1, __int64 *a2, __int64 a3, double a4, double a5, double a6)
{
  return sub_20CC690(
           (*(unsigned __int16 *)(**(_QWORD **)a1 + 18LL) >> 5) & 0x7FFFBFF,
           a2,
           a3,
           *(_QWORD *)(**(_QWORD **)a1 - 24LL),
           a4,
           a5,
           a6);
}
