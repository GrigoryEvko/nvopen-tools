// Function: sub_12FB820
// Address: 0x12fb820
//
unsigned __int64 __fastcall sub_12FB820(unsigned __int64 a1, __int64 a2)
{
  if ( (a1 & 0x7E00) == 0x7C00 && (a1 & 0x1FF) != 0 )
    sub_12F9B70(16);
  *(_QWORD *)(a2 + 8) = 0;
  *(_BYTE *)a2 = a1 >> 15 != 0;
  *(_QWORD *)(a2 + 16) = a1 << 54;
  return a1 >> 15;
}
