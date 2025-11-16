// Function: sub_12FB8A0
// Address: 0x12fb8a0
//
unsigned __int64 __fastcall sub_12FB8A0(unsigned __int64 a1, __int64 a2)
{
  if ( (a1 & 0x7FC00000) == 0x7F800000 && (a1 & 0x3FFFFF) != 0 )
    sub_12F9B70(16);
  *(_QWORD *)(a2 + 8) = 0;
  *(_BYTE *)a2 = a1 >> 31 != 0;
  *(_QWORD *)(a2 + 16) = a1 << 41;
  return a1 >> 31;
}
