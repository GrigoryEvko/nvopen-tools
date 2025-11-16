// Function: sub_6413B0
// Address: 0x6413b0
//
__int64 __fastcall sub_6413B0(__int64 a1, int a2)
{
  __int64 v2; // rdx
  __int64 v3; // rcx
  __int64 v4; // r9
  __int64 result; // rax
  char v6; // al
  __int64 v7; // rdx
  int v8; // eax
  char v9; // dl
  bool v10; // al
  char v11; // dl

  v2 = 776LL * a2;
  v3 = qword_4F04C68[0];
  v4 = qword_4F04C68[0] + v2;
  if ( *(_BYTE *)(qword_4F04C68[0] + v2 + 4) == 6 )
    *(_BYTE *)(a1 + 64) |= 1u;
  if ( dword_4F04C58 != -1 && (*(_QWORD *)(a1 + 48) || *(_BYTE *)(a1 + 44) == 1) )
    *(_BYTE *)(a1 + 64) |= 0x10u;
  result = *(unsigned __int8 *)(a1 + 64);
  if ( dword_4F077C4 == 2 )
  {
    if ( (result & 1) != 0 )
    {
      v6 = *(_BYTE *)(v4 + 4);
      v7 = v3 + v2 - 776;
      while ( 1 )
      {
        if ( (unsigned __int8)(v6 - 6) > 1u )
        {
          v8 = dword_4F04C34;
          goto LABEL_29;
        }
        v6 = *(_BYTE *)(v7 + 4);
        --a2;
        if ( v6 == 9 )
          break;
        if ( v6 != 10 )
        {
          v7 -= 776;
          if ( v6 != 8 )
            continue;
        }
        a2 = dword_4F04C34;
LABEL_37:
        *(_DWORD *)(a1 + 40) = a2;
        goto LABEL_38;
      }
      v8 = dword_4F04C34;
      if ( (*(_BYTE *)(v7 + 6) & 6) == 0 )
      {
        a2 = dword_4F04C34;
        goto LABEL_37;
      }
LABEL_29:
      *(_DWORD *)(a1 + 40) = a2;
      if ( v8 != a2 )
      {
        v9 = *(_BYTE *)(v3 + 776LL * a2 + 4);
        v10 = v9 == 2 || v9 == 17;
        goto LABEL_39;
      }
LABEL_38:
      v10 = 0;
LABEL_39:
      v11 = 32 * v10;
      LOBYTE(result) = (32 * v10) | *(_BYTE *)(a1 + 64) & 0xDF;
      *(_BYTE *)(a1 + 64) = v11 | *(_BYTE *)(a1 + 64) & 0xDF;
    }
    else
    {
      *(_DWORD *)(a1 + 40) = a2;
    }
    if ( (result & 0x30) != 0 )
      a2 = dword_4F04C34;
    if ( (*(_BYTE *)(v3 + 776LL * a2 + 8) & 2) != 0 )
      *(_BYTE *)(a1 + 64) |= 0x40u;
    result = *(_BYTE *)(v3 + 776LL * (int)dword_4F04C5C + 9) & 0xE;
    if ( (_BYTE)result == 6 )
      *(_BYTE *)(a1 + 64) |= 0x80u;
  }
  else
  {
    if ( (result & 0x10) != 0 )
    {
      if ( dword_4F077C4 == 1 )
      {
        if ( *(_BYTE *)(a1 + 44) == 1 )
        {
          a2 = 0;
        }
        else
        {
          result = 0;
          if ( *(_QWORD *)(a1 + 48) )
            a2 = 0;
        }
      }
      else if ( *(_QWORD *)(a1 + 48) )
      {
        result = 0;
        if ( *(_BYTE *)(a1 + 44) == 2 )
          a2 = 0;
      }
    }
    *(_DWORD *)(a1 + 40) = a2;
  }
  return result;
}
