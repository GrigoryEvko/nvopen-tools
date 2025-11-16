// Function: sub_5CCAE0
// Address: 0x5ccae0
//
__int64 __fastcall sub_5CCAE0(unsigned __int8 a1, __int64 a2)
{
  char *v2; // rax
  __int64 result; // rax

  if ( *(_BYTE *)(a2 + 9) == 4 )
  {
    result = sub_684AA0(a1, 2470, a2 + 56);
  }
  else
  {
    v2 = sub_5C79F0(a2);
    result = sub_6849F0(a1, 1835, a2 + 56, v2);
  }
  *(_BYTE *)(a2 + 8) = 0;
  return result;
}
