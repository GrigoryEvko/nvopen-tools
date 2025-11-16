// Function: sub_B506C0
// Address: 0xb506c0
//
__int64 __fastcall sub_B506C0(unsigned __int8 *a1)
{
  char v1; // r8
  __int64 result; // rax

  v1 = sub_B46D50(a1);
  result = 1;
  if ( v1 )
  {
    sub_BD28A0(a1 - 64, a1 - 32);
    return 0;
  }
  return result;
}
