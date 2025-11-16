// Function: sub_12FB920
// Address: 0x12fb920
//
unsigned __int64 __fastcall sub_12FB920(__int64 a1, __int64 a2)
{
  if ( (a1 & 0x7FF8000000000000LL) == 0x7FF0000000000000LL && (a1 & 0x7FFFFFFFFFFFFLL) != 0 )
    sub_12F9B70(16);
  *(_QWORD *)(a2 + 8) = 0;
  *(_QWORD *)(a2 + 16) = a1 << 12;
  *(_BYTE *)a2 = a1 < 0;
  *(_BYTE *)a2 &= 1u;
  return (unsigned __int64)a1 >> 63;
}
