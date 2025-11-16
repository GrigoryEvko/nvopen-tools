// Function: sub_15A9690
// Address: 0x15a9690
//
__int64 __fastcall sub_15A9690(__int64 a1, __int64 a2, unsigned int a3)
{
  unsigned __int8 *v3; // rax
  unsigned __int8 *v6; // rdx
  __int64 v7; // rsi

  v3 = *(unsigned __int8 **)(a1 + 24);
  v6 = &v3[*(unsigned int *)(a1 + 32)];
  if ( v6 == v3 )
    return 0;
  while ( 1 )
  {
    v7 = *v3;
    if ( (unsigned int)v7 >= a3 )
      break;
    if ( v6 == ++v3 )
      return 0;
  }
  return sub_1644C60(a2, v7);
}
