// Function: sub_AE44B0
// Address: 0xae44b0
//
__int64 __fastcall sub_AE44B0(__int64 a1, __int64 a2, unsigned int a3)
{
  unsigned __int8 *v3; // rax
  unsigned __int8 *v5; // rcx
  __int64 v6; // rsi

  v3 = *(unsigned __int8 **)(a1 + 32);
  v5 = &v3[*(_QWORD *)(a1 + 40)];
  if ( v5 == v3 )
    return 0;
  while ( 1 )
  {
    v6 = *v3;
    if ( (unsigned int)v6 >= a3 )
      break;
    if ( v5 == ++v3 )
      return 0;
  }
  return sub_BCD140(a2, v6);
}
