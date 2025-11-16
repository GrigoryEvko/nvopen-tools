// Function: sub_E77580
// Address: 0xe77580
//
__int64 __fastcall sub_E77580(__int64 *a1, _QWORD *a2, int a3, __int64 a4)
{
  return sub_E770B0(
           a1,
           a2,
           (unsigned __int8)a3 | (unsigned __int64)((unsigned __int16)a3 & 0xFF00) | ((unsigned __int64)BYTE2(a3) << 16),
           (__int64)&unk_3F80270,
           (unsigned __int8)a3 - 1,
           a4);
}
