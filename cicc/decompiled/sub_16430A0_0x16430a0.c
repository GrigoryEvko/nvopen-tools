// Function: sub_16430A0
// Address: 0x16430a0
//
__int64 __fastcall sub_16430A0(__int64 a1, __int64 a2)
{
  unsigned int v2; // r12d
  int v3; // edx
  int v6; // esi
  int v7; // ebx

  if ( a1 == a2 )
    return 1;
  v3 = *(unsigned __int8 *)(a1 + 8);
  LOBYTE(v2) = v3 != 0 && v3 != 12;
  if ( (_BYTE)v2 )
  {
    v6 = *(unsigned __int8 *)(a2 + 8);
    LOBYTE(v2) = v6 != 0 && v6 != 12;
    if ( (_BYTE)v2 )
    {
      if ( (_BYTE)v3 == 16 )
      {
        if ( (_BYTE)v6 == 16 )
        {
          v7 = *(_DWORD *)(a1 + 32) * sub_1643030(*(_QWORD *)(a1 + 24));
          LOBYTE(v2) = *(_DWORD *)(a2 + 32) * (unsigned int)sub_1643030(*(_QWORD *)(a2 + 24)) == v7;
        }
        else if ( v6 == 9 )
        {
          if ( *(_DWORD *)(a1 + 32) * (unsigned int)sub_1643030(*(_QWORD *)(a1 + 24)) != 64 )
            return 0;
        }
        else
        {
          return 0;
        }
      }
      else if ( (_BYTE)v6 == 16 && v3 == 9 )
      {
        LOBYTE(v2) = *(_DWORD *)(a2 + 32) * (unsigned int)sub_1643030(*(_QWORD *)(a2 + 24)) == 64;
      }
      else
      {
        LOBYTE(v2) = (_BYTE)v6 == 15 && *(_BYTE *)(a1 + 8) == 15;
        if ( (_BYTE)v2 )
          LOBYTE(v2) = *(_DWORD *)(a2 + 8) >> 8 == *(_DWORD *)(a1 + 8) >> 8;
      }
    }
  }
  return v2;
}
