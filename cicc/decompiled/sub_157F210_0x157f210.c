// Function: sub_157F210
// Address: 0x157f210
//
__int64 __fastcall sub_157F210(__int64 a1)
{
  unsigned __int64 v1; // rax
  unsigned __int64 v2; // r12
  int v3; // r14d
  __int64 v4; // r13
  unsigned int v5; // ebx

  v1 = sub_157EBA0(a1);
  if ( !v1 )
    return 0;
  v2 = v1;
  v3 = sub_15F4D60(v1);
  if ( !v3 )
    return 0;
  v4 = sub_15F4DF0(v2, 0);
  if ( v3 != 1 )
  {
    v5 = 1;
    while ( v4 == sub_15F4DF0(v2, v5) )
    {
      if ( ++v5 == v3 )
        return v4;
    }
    return 0;
  }
  return v4;
}
