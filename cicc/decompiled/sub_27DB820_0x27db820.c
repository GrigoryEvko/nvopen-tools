// Function: sub_27DB820
// Address: 0x27db820
//
__int64 __fastcall sub_27DB820(_QWORD *a1, _QWORD *a2)
{
  __int64 result; // rax
  unsigned __int64 v3; // rcx
  unsigned __int64 v4; // rdx

  result = 0xFFFFFFFFLL;
  if ( *a2 <= *a1 )
  {
    if ( *a2 == *a1 )
    {
      v3 = a1[1];
      v4 = a2[1];
      if ( v3 < v4 )
        return result;
      return v3 > v4;
    }
    result = 1;
    if ( *a2 >= *a1 )
    {
      v3 = a1[1];
      v4 = a2[1];
      return v3 > v4;
    }
  }
  return result;
}
