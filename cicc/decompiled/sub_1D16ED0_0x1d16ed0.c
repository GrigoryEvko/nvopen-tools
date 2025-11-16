// Function: sub_1D16ED0
// Address: 0x1d16ed0
//
__int64 __fastcall sub_1D16ED0(unsigned int a1)
{
  return a1 & 0xFFFFFFF9 | (2 * (_BYTE)a1) & 4 | (a1 >> 1) & 2;
}
