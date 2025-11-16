// Function: sub_BDB940
// Address: 0xbdb940
//
char __fastcall sub_BDB940(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v2; // rax
  unsigned __int64 v3; // rdx
  __int64 v4; // rdi
  unsigned __int64 v5; // rsi
  __int64 v6; // rdx
  char v7; // cl
  __int64 v9; // [rsp+0h] [rbp-30h] BYREF
  unsigned __int64 v10; // [rsp+8h] [rbp-28h]
  char v11[9]; // [rsp+1Fh] [rbp-11h] BYREF

  v9 = a1;
  v10 = a2;
  v11[0] = 44;
  v2 = sub_C931B0(&v9, v11, 1, 0);
  if ( v2 == -1 )
  {
    v4 = v9;
    v2 = v10;
    v5 = 0;
    v6 = 0;
  }
  else
  {
    v3 = v2 + 1;
    v4 = v9;
    if ( v2 + 1 > v10 )
    {
      v3 = v10;
      v5 = 0;
    }
    else
    {
      v5 = v10 - v3;
    }
    v6 = v9 + v3;
    if ( v2 > v10 )
      v2 = v10;
  }
  v7 = 0;
  if ( v2 )
  {
    if ( v2 == 4 )
    {
      if ( *(_DWORD *)v4 == 1701143913 )
        goto LABEL_10;
      goto LABEL_9;
    }
    if ( v2 == 13 )
    {
      if ( *(_QWORD *)v4 == 0x6576726573657270LL && *(_DWORD *)(v4 + 8) == 1734964013 )
      {
        v7 = 1;
        if ( *(_BYTE *)(v4 + 12) == 110 )
          goto LABEL_10;
      }
      if ( *(_QWORD *)v4 == 0x6576697469736F70LL && *(_DWORD *)(v4 + 8) == 1919253037 )
      {
        v7 = 2;
        if ( *(_BYTE *)(v4 + 12) == 111 )
          goto LABEL_10;
      }
    }
    else if ( v2 == 7 && *(_DWORD *)v4 == 1634629988 && *(_WORD *)(v4 + 4) == 26989 && *(_BYTE *)(v4 + 6) == 99 )
    {
      v7 = 3;
      goto LABEL_10;
    }
LABEL_9:
    v7 = -1;
  }
LABEL_10:
  switch ( v5 )
  {
    case 0uLL:
      return v7;
    case 4uLL:
      if ( *(_DWORD *)v6 == 1701143913 )
        return v7;
      break;
    case 0xDuLL:
      if ( *(_QWORD *)v6 == 0x6576726573657270LL && *(_DWORD *)(v6 + 8) == 1734964013 && *(_BYTE *)(v6 + 12) == 110
        || *(_QWORD *)v6 == 0x6576697469736F70LL && *(_DWORD *)(v6 + 8) == 1919253037 && *(_BYTE *)(v6 + 12) == 111 )
      {
        return v7;
      }
      break;
    default:
      if ( v5 == 7 && *(_DWORD *)v6 == 1634629988 && *(_WORD *)(v6 + 4) == 26989 && *(_BYTE *)(v6 + 6) == 99 )
        return v7;
      break;
  }
  return v7;
}
