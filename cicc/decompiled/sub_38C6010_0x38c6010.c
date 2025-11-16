// Function: sub_38C6010
// Address: 0x38c6010
//
__int64 __fastcall sub_38C6010(__int64 *a1, __int64 *a2, int a3, __int64 a4)
{
  return sub_38C5CD0(
           a1,
           a2,
           (unsigned __int8)a3 | a3 & 0xFF00 | (BYTE2(a3) << 16),
           byte_452E020,
           (unsigned __int8)a3 - 1,
           a4);
}
