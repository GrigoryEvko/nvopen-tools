// Function: sub_B2B600
// Address: 0xb2b600
//
__int64 __fastcall sub_B2B600(__int64 a1)
{
  __int64 result; // rax
  __int64 v2[2]; // [rsp+8h] [rbp-18h] BYREF

  v2[0] = a1;
  result = sub_A73700(v2);
  if ( !result )
  {
    result = sub_A736E0(v2);
    if ( !result )
    {
      result = sub_A73740(v2);
      if ( !result )
      {
        result = sub_A73760(v2);
        if ( !result )
          return sub_A73720(v2);
      }
    }
  }
  return result;
}
