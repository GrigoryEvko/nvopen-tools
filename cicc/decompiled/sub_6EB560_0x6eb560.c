// Function: sub_6EB560
// Address: 0x6eb560
//
__int64 __fastcall sub_6EB560(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  int v3; // r8d

  result = sub_8D5830(a1);
  if ( (_DWORD)result )
  {
    v3 = sub_6E5430();
    result = 1;
    if ( v3 )
    {
      sub_5EB950(8u, 322, a1, a2);
      return 1;
    }
  }
  return result;
}
