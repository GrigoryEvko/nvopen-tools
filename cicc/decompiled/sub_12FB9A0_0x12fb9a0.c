// Function: sub_12FB9A0
// Address: 0x12fb9a0
//
unsigned __int64 __fastcall sub_12FB9A0(unsigned __int8 *a1)
{
  return (*((_QWORD *)a1 + 2) >> 12) | ((unsigned __int64)*a1 << 63) | 0x7FF8000000000000LL;
}
