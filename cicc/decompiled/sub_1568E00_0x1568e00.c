// Function: sub_1568E00
// Address: 0x1568e00
//
unsigned __int64 __fastcall sub_1568E00(__int64 a1)
{
  unsigned __int64 result; // rax

  result = sub_22416F0(a1, "mov\tfp", 0, 6);
  if ( !result )
  {
    result = sub_22416F0(a1, "objc_retainAutoreleaseReturnValue", 0, 33);
    if ( result != -1 )
    {
      result = sub_22416F0(a1, "# marker", 0, 8);
      if ( result != -1 )
      {
        if ( result > *(_QWORD *)(a1 + 8) )
          sub_222CF80("%s: __pos (which is %zu) > this->size() (which is %zu)", (char)"basic_string::replace");
        return sub_2241130(a1, result, *(_QWORD *)(a1 + 8) != result, ";", 1);
      }
    }
  }
  return result;
}
