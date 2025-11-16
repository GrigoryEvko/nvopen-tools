// Function: sub_721130
// Address: 0x721130
//
__int64 __fastcall sub_721130(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 i; // rax
  unsigned int v6; // edx

  if ( a3 )
  {
    for ( i = 0; i != a3; ++i )
    {
      v6 = *(unsigned __int8 *)(a1 + i) - *(unsigned __int8 *)(a2 + i);
      if ( v6 )
        break;
    }
  }
  else
  {
    return 0;
  }
  return v6;
}
