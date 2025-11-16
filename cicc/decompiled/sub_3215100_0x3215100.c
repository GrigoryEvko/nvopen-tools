// Function: sub_3215100
// Address: 0x3215100
//
unsigned __int64 __fastcall sub_3215100(unsigned __int64 a1)
{
  unsigned __int64 result; // rax
  __int64 v2; // rdx

  result = sub_32150B0(a1);
  if ( result )
  {
    v2 = *(_QWORD *)(result + 40);
    result = 0;
    if ( v2 )
    {
      if ( (v2 & 4) != 0 )
        return v2 & 0xFFFFFFFFFFFFFFF8LL;
    }
  }
  return result;
}
