// Function: sub_8CF730
// Address: 0x8cf730
//
__int64 __fastcall sub_8CF730(__int64 a1)
{
  __int64 result; // rax
  __int64 *v2; // rbx
  __int64 *v3; // rax
  __int64 v4; // rdi

  result = sub_8C6310(*(_QWORD *)(a1 + 104));
  if ( result )
  {
    v2 = (__int64 *)result;
    do
    {
      v3 = (__int64 *)v2[4];
      if ( v3 )
      {
        if ( *(__int64 **)(*v3 + 32) != v3 )
          sub_8CF610((__int64)v2, *v3);
      }
      else
      {
        v4 = *v2;
        if ( dword_4F077C4 == 2 && v4 )
        {
          if ( (unsigned int)sub_8C6B40(v4) )
            sub_8CA0A0((__int64)v2, 1u);
        }
      }
      result = sub_8C6310(v2[14]);
      v2 = (__int64 *)result;
    }
    while ( result );
  }
  return result;
}
