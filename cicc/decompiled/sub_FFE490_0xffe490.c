// Function: sub_FFE490
// Address: 0xffe490
//
__int64 __fastcall sub_FFE490(__int64 a1, __int64 a2)
{
  _BYTE *v2; // r12
  _BYTE *v4; // rax

  v2 = *(_BYTE **)(a2 + 24);
  if ( *v2 == 82 )
  {
    v4 = *(_BYTE **)&v2[32 * (1 - (unsigned int)sub_BD2910(a2)) - 64];
    if ( *v4 == 61 && **((_BYTE **)v4 - 4) == 3 )
      return 1;
  }
  *(_BYTE *)(a1 + 8) = 1;
  return 0;
}
