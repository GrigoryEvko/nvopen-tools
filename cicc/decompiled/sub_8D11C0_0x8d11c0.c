// Function: sub_8D11C0
// Address: 0x8d11c0
//
__int64 __fastcall sub_8D11C0(__int64 a1, _DWORD *a2)
{
  char v2; // al

  v2 = *(_BYTE *)(a1 + 140);
  if ( (unsigned __int8)(v2 - 9) <= 2u || v2 == 2 && (*(_BYTE *)(a1 + 161) & 8) != 0 )
  {
    if ( (*(_BYTE *)(a1 + 88) & 0x70) == 0x10 )
    {
      while ( (*(_BYTE *)(a1 + 89) & 4) != 0 )
        a1 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL);
      *(_BYTE *)(*(_QWORD *)a1 + 82LL) |= 2u;
      *a2 = 0;
      return 0;
    }
    else
    {
      *a2 = 0;
      return 0;
    }
  }
  else
  {
    *a2 = 0;
    return 0;
  }
}
