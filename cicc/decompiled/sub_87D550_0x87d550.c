// Function: sub_87D550
// Address: 0x87d550
//
__int64 __fastcall sub_87D550(__int64 a1)
{
  char v1; // al
  __int64 v2; // rcx
  char v3; // dl
  unsigned int v4; // r8d

  v1 = *(_BYTE *)(a1 + 80);
  v2 = a1;
  v3 = v1;
  if ( v1 == 16 )
  {
    v2 = **(_QWORD **)(a1 + 88);
    v3 = *(_BYTE *)(v2 + 80);
  }
  if ( v3 == 24 )
    v3 = *(_BYTE *)(*(_QWORD *)(v2 + 88) + 80LL);
  v4 = 0;
  if ( v3 == 17 )
    return v4;
  switch ( v1 )
  {
    case 16:
    case 24:
      return *(_BYTE *)(a1 + 96) & 3;
    case 19:
      v4 = *(unsigned __int8 *)(*(_QWORD *)(a1 + 88) + 265LL);
      LOBYTE(v4) = (unsigned __int8)v4 >> 6;
      return v4;
    case 20:
      return *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(a1 + 88) + 176LL) + 88LL) & 3;
    case 3:
      if ( *(_BYTE *)(a1 + 104) )
        return v4;
      break;
    case 13:
      return v4;
  }
  return *(_BYTE *)(sub_87D520(a1) + 88) & 3;
}
