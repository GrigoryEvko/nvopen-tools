// Function: sub_16341C0
// Address: 0x16341c0
//
__int64 __fastcall sub_16341C0(_QWORD *a1)
{
  unsigned int v1; // r8d
  unsigned __int64 v2; // rdx
  char *v3; // rsi
  char *v4; // rax
  __int64 v5; // rcx
  __int64 v6; // rdx
  char *v7; // rdx

  v1 = 0;
  v2 = *a1 & 0xFFFFFFFFFFFFFFF8LL;
  v3 = *(char **)(v2 + 32);
  v4 = *(char **)(v2 + 24);
  v5 = (v3 - v4) >> 3;
  if ( v4 != v3 )
  {
    v6 = (v3 - v4) >> 5;
    if ( v6 > 0 )
    {
      v7 = &v4[32 * v6];
      while ( (*(_BYTE *)(*(_QWORD *)v4 + 12LL) & 0x40) != 0 )
      {
        if ( (*(_BYTE *)(*((_QWORD *)v4 + 1) + 12LL) & 0x40) == 0 )
        {
          LOBYTE(v1) = v3 == v4 + 8;
          return v1;
        }
        if ( (*(_BYTE *)(*((_QWORD *)v4 + 2) + 12LL) & 0x40) == 0 )
        {
          LOBYTE(v1) = v3 == v4 + 16;
          return v1;
        }
        if ( (*(_BYTE *)(*((_QWORD *)v4 + 3) + 12LL) & 0x40) == 0 )
        {
          LOBYTE(v1) = v3 == v4 + 24;
          return v1;
        }
        v4 += 32;
        if ( v7 == v4 )
        {
          v5 = (v3 - v4) >> 3;
          goto LABEL_12;
        }
      }
      goto LABEL_9;
    }
LABEL_12:
    if ( v5 != 2 )
    {
      if ( v5 != 3 )
      {
        v1 = 1;
        if ( v5 != 1 )
          return v1;
        goto LABEL_15;
      }
      if ( (*(_BYTE *)(*(_QWORD *)v4 + 12LL) & 0x40) == 0 )
      {
LABEL_9:
        LOBYTE(v1) = v3 == v4;
        return v1;
      }
      v4 += 8;
    }
    if ( (*(_BYTE *)(*(_QWORD *)v4 + 12LL) & 0x40) != 0 )
    {
      v4 += 8;
LABEL_15:
      v1 = 1;
      if ( (*(_BYTE *)(*(_QWORD *)v4 + 12LL) & 0x40) != 0 )
        return v1;
      goto LABEL_9;
    }
    goto LABEL_9;
  }
  return v1;
}
