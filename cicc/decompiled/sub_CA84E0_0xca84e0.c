// Function: sub_CA84E0
// Address: 0xca84e0
//
__int64 __fastcall sub_CA84E0(__int64 a1)
{
  unsigned __int8 *v1; // rax
  unsigned int v2; // r12d

  v1 = *(unsigned __int8 **)(a1 + 40);
  if ( v1 == *(unsigned __int8 **)(a1 + 48) )
    return 32;
  v2 = *v1;
  if ( (((_BYTE)v2 - 43) & 0xFD) != 0 )
    return 32;
  sub_CA7F70(a1, 1u);
  return v2;
}
