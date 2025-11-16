// Function: sub_16E2B00
// Address: 0x16e2b00
//
__int64 __fastcall sub_16E2B00(__int64 a1, __int64 a2)
{
  int v2; // eax
  unsigned int v3; // r8d
  int v5; // eax
  int v6; // edx
  int v7; // ecx

  v2 = *(_DWORD *)(a1 + 32);
  if ( v2 == 29 )
  {
    if ( *(_DWORD *)(a2 + 32) == 1 )
    {
      v5 = *(_DWORD *)(a1 + 40);
      v6 = *(_DWORD *)(a1 + 36);
      v7 = *(_DWORD *)(a2 + 36);
      if ( v5 != 1 )
        goto LABEL_10;
LABEL_24:
      v3 = 0;
      if ( v7 != v6 )
        return v3;
LABEL_20:
      v3 = 0;
      if ( *(_DWORD *)(a2 + 40) == 1 )
        LOBYTE(v3) = *(_DWORD *)(a1 + 44) == *(_DWORD *)(a2 + 44);
      return v3;
    }
    goto LABEL_5;
  }
  if ( v2 == 1 )
  {
    if ( *(_DWORD *)(a2 + 32) == 29 )
      goto LABEL_9;
    goto LABEL_5;
  }
  if ( v2 != 30 )
  {
    if ( v2 == 2 )
    {
      if ( *(_DWORD *)(a2 + 32) != 30 )
      {
        if ( *(_DWORD *)(a1 + 40) != 1 )
          goto LABEL_6;
LABEL_18:
        v3 = 0;
        if ( v2 != *(_DWORD *)(a2 + 32) || *(_DWORD *)(a1 + 36) != *(_DWORD *)(a2 + 36) )
          return v3;
        goto LABEL_20;
      }
      goto LABEL_9;
    }
LABEL_5:
    if ( *(_DWORD *)(a1 + 40) != 1 )
    {
LABEL_6:
      v3 = 0;
      if ( *(_QWORD *)(a1 + 32) == *(_QWORD *)(a2 + 32) && *(_QWORD *)(a1 + 40) == *(_QWORD *)(a2 + 40) )
        LOBYTE(v3) = *(_QWORD *)(a1 + 48) == *(_QWORD *)(a2 + 48);
      return v3;
    }
    goto LABEL_18;
  }
  if ( *(_DWORD *)(a2 + 32) != 2 )
    goto LABEL_5;
LABEL_9:
  v5 = *(_DWORD *)(a1 + 40);
  v6 = *(_DWORD *)(a1 + 36);
  v7 = *(_DWORD *)(a2 + 36);
  if ( v5 == 1 )
    goto LABEL_24;
LABEL_10:
  v3 = 0;
  if ( v7 == v6
    && v5 == *(_DWORD *)(a2 + 40)
    && *(_DWORD *)(a2 + 44) == *(_DWORD *)(a1 + 44)
    && *(_DWORD *)(a1 + 48) == *(_DWORD *)(a2 + 48) )
  {
    LOBYTE(v3) = *(_DWORD *)(a1 + 52) == *(_DWORD *)(a2 + 52);
  }
  return v3;
}
