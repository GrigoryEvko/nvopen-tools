// Function: sub_2D42E90
// Address: 0x2d42e90
//
__int64 __fastcall sub_2D42E90(__int64 a1, __int64 a2, __int64 a3)
{
  return sub_2A2C8D0(
           (*(_WORD *)(**(_QWORD **)a1 + 2LL) >> 4) & 0x1F,
           a2,
           a3,
           *(unsigned __int8 **)(**(_QWORD **)a1 - 32LL));
}
