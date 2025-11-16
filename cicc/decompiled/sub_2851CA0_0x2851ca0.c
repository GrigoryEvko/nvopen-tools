// Function: sub_2851CA0
// Address: 0x2851ca0
//
__int64 __fastcall sub_2851CA0(__int64 a1, char *a2, __int64 a3, __int64 a4, unsigned int a5)
{
  char v6; // al
  __int64 v8; // rax
  unsigned int v9; // eax

  v6 = *a2;
  LOBYTE(a5) = *a2 == 61;
  if ( *a2 == 62 )
  {
    LOBYTE(a5) = *((_QWORD *)a2 - 4) == a3;
    return a5;
  }
  if ( v6 != 85 )
  {
    if ( v6 == 66 )
    {
      LOBYTE(a5) = *((_QWORD *)a2 - 8) == a3;
    }
    else if ( v6 == 65 )
    {
      LOBYTE(a5) = *((_QWORD *)a2 - 12) == a3;
    }
    return a5;
  }
  v8 = *((_QWORD *)a2 - 4);
  a5 = 0;
  if ( !v8 || *(_BYTE *)v8 || *(_QWORD *)(v8 + 24) != *((_QWORD *)a2 + 10) || (*(_BYTE *)(v8 + 33) & 0x20) == 0 )
    return a5;
  v9 = *(_DWORD *)(v8 + 36);
  if ( v9 > 0xF3 )
  {
    if ( v9 == 286 )
    {
LABEL_19:
      LOBYTE(a5) = *(_QWORD *)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)] == a3;
      return a5;
    }
    goto LABEL_15;
  }
  if ( v9 <= 0xE3 )
  {
LABEL_15:
    a5 = sub_DFDD50(a1);
    if ( (_BYTE)a5 )
      LOBYTE(a5) = a3 == 0;
    return a5;
  }
  switch ( v9 )
  {
    case 0xE4u:
    case 0xF3u:
      goto LABEL_19;
    case 0xE6u:
      LOBYTE(a5) = *(_QWORD *)&a2[32 * (1LL - (*((_DWORD *)a2 + 1) & 0x7FFFFFF))] == a3;
      break;
    case 0xEEu:
    case 0xF1u:
      a5 = 1;
      if ( a3 != *(_QWORD *)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)] )
        LOBYTE(a5) = *(_QWORD *)&a2[32 * (1LL - (*((_DWORD *)a2 + 1) & 0x7FFFFFF))] == a3;
      break;
    default:
      goto LABEL_15;
  }
  return a5;
}
