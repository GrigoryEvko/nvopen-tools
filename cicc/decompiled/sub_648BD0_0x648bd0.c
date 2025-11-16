// Function: sub_648BD0
// Address: 0x648bd0
//
__int64 __fastcall sub_648BD0(__int64 **a1, __int64 a2)
{
  __int64 result; // rax

  while ( a1 )
  {
    while ( 1 )
    {
      result = *((unsigned __int8 *)a1 + 32);
      a1 = (__int64 **)*a1;
      if ( (result & 4) == 0 )
        break;
      if ( !a1 )
        return result;
      if ( ((_WORD)a1[4] & 0x104) == 0 )
        return sub_6851C0(306, a2);
    }
  }
  return result;
}
