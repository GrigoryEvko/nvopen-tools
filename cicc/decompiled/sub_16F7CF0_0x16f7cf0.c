// Function: sub_16F7CF0
// Address: 0x16f7cf0
//
__int64 __fastcall sub_16F7CF0(__int64 a1)
{
  unsigned __int8 *v1; // rax
  unsigned int v2; // r12d

  v1 = *(unsigned __int8 **)(a1 + 40);
  if ( v1 == *(unsigned __int8 **)(a1 + 48) )
    return 32;
  v2 = *v1;
  if ( (((_BYTE)v2 - 43) & 0xFD) != 0 )
    return 32;
  sub_16F7930(a1, 1u);
  return v2;
}
