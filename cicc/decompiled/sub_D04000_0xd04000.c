// Function: sub_D04000
// Address: 0xd04000
//
__int64 __fastcall sub_D04000(__int64 a1, __int64 a2, int a3)
{
  char v4; // r8
  __int64 result; // rax
  char v6; // r8

  v4 = sub_B49B80(a2, a3, 78);
  result = 2;
  if ( !v4 )
  {
    v6 = sub_B49B80(a2, a3, 51);
    result = 1;
    if ( !v6 )
      return (unsigned __int8)sub_B49B80(a2, a3, 50) == 0 ? 3 : 0;
  }
  return result;
}
