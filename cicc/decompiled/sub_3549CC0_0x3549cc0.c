// Function: sub_3549CC0
// Address: 0x3549cc0
//
__int64 __fastcall sub_3549CC0(_QWORD *a1, _QWORD *a2)
{
  return ((((__int64)(a1[3] - a2[3]) >> 3) - 1) << 6) + ((__int64)(*a1 - a1[1]) >> 3) + ((__int64)(a2[2] - *a2) >> 3);
}
