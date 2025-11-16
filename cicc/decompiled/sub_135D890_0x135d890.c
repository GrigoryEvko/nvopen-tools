// Function: sub_135D890
// Address: 0x135d890
//
__int64 __fastcall sub_135D890(__int64 a1)
{
  unsigned int v1; // r12d

  if ( *(_BYTE *)(a1 + 16) == 53 )
    return (unsigned int)sub_139D0F0(a1, 0, 1) ^ 1;
  v1 = sub_134E780(a1);
  if ( (_BYTE)v1 || *(_BYTE *)(a1 + 16) == 17 && ((unsigned __int8)sub_15E0450(a1) || (unsigned __int8)sub_15E04B0(a1)) )
    return (unsigned int)sub_139D0F0(a1, 0, 1) ^ 1;
  return v1;
}
