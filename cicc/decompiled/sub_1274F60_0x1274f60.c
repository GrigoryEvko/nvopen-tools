// Function: sub_1274F60
// Address: 0x1274f60
//
__int64 __fastcall sub_1274F60(__int64 *a1)
{
  __int64 result; // rax

  sub_1274D40(a1);
  sub_1273F90(a1);
  sub_1269EE0(a1);
  if ( a1[48] )
  {
    sub_1632AD0(*a1, 1, "Debug Info Version", 18, 3);
    sub_15A6130(a1[48] + 16);
  }
  result = dword_4D04654;
  if ( !dword_4D04654 )
    return sub_1269D00((__int64 **)a1);
  return result;
}
