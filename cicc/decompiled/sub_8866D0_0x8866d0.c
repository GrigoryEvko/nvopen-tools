// Function: sub_8866D0
// Address: 0x8866d0
//
__int64 __fastcall sub_8866D0(char *a1)
{
  __int64 result; // rax

  result = sub_8865A0(a1);
  if ( result )
  {
    if ( *(_BYTE *)(result + 80) != 19 )
      return 0;
  }
  return result;
}
