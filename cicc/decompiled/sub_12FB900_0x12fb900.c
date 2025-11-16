// Function: sub_12FB900
// Address: 0x12fb900
//
unsigned __int64 __fastcall sub_12FB900(unsigned __int8 *a1)
{
  return (*((_QWORD *)a1 + 2) >> 41) | ((unsigned __int64)*a1 << 31) | 0x7FC00000;
}
