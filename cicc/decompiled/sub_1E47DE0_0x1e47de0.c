// Function: sub_1E47DE0
// Address: 0x1e47de0
//
__int64 __fastcall sub_1E47DE0(_QWORD *a1, _QWORD *a2)
{
  return ((((__int64)(a1[3] - a2[3]) >> 3) - 1) << 6) + ((__int64)(*a1 - a1[1]) >> 3) + ((__int64)(a2[2] - *a2) >> 3);
}
