// Function: sub_17F4470
// Address: 0x17f4470
//
__int64 __fastcall sub_17F4470(__int64 a1, __int64 a2)
{
  unsigned __int64 v3; // rax
  unsigned __int64 v4; // rdi
  int v5; // r13d
  unsigned __int64 v6; // r14
  unsigned int v7; // r15d
  __int64 v8; // rax

  v3 = sub_157EBA0(a1);
  if ( !v3 || !(unsigned int)sub_15F4D60(v3) )
    return 0;
  v4 = sub_157EBA0(a1);
  if ( v4 )
  {
    v5 = sub_15F4D60(v4);
    v6 = sub_157EBA0(a1);
    if ( v5 )
    {
      v7 = 0;
      while ( 1 )
      {
        v8 = sub_15F4DF0(v6, v7);
        if ( !sub_15CC8F0(a2, a1, v8) )
          break;
        if ( ++v7 == v5 )
          return 1;
      }
      return 0;
    }
  }
  return 1;
}
