// Function: sub_AD6D20
// Address: 0xad6d20
//
__int64 __fastcall sub_AD6D20(unsigned __int8 *a1, __int64 a2)
{
  unsigned int v2; // r13d
  __int64 v3; // rsi

  if ( *(_BYTE *)a2 != 17 )
    return 0;
  v2 = *(_DWORD *)(a2 + 32);
  if ( v2 <= 0x40 )
  {
    v3 = *(_QWORD *)(a2 + 24);
    return sub_AD69F0(a1, v3);
  }
  if ( v2 - (unsigned int)sub_C444A0(a2 + 24) <= 0x40 )
  {
    v3 = **(_QWORD **)(a2 + 24);
    return sub_AD69F0(a1, v3);
  }
  return 0;
}
