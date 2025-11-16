// Function: sub_81A4B0
// Address: 0x81a4b0
//
_BOOL8 __fastcall sub_81A4B0(__int64 a1, unsigned __int64 a2, _BYTE *a3)
{
  _BYTE *v4; // r8
  _BOOL8 result; // rax
  __int64 v6; // rdx
  unsigned __int64 v7; // rcx
  char v8; // al
  unsigned __int8 v9; // si
  bool v10; // r10
  _BOOL4 v11; // esi
  bool v12; // al
  unsigned __int64 v13; // rsi
  unsigned __int8 v14; // al
  bool v15; // r10
  unsigned __int64 v16; // rax
  __int64 v17; // r10
  unsigned __int64 v18; // r11

  v4 = a3;
  result = 0;
  if ( !a3 )
    return result;
  if ( !a2 )
    return *v4 == 0;
  v6 = 0;
  v7 = 0;
  do
  {
    v8 = *(_BYTE *)(a1 + v7);
    v9 = v4[v6];
    if ( v8 == v9 )
    {
      if ( v8 == 1 )
        goto LABEL_21;
      if ( (unsigned __int8)(v8 - 3) <= 3u || v8 == 8 )
      {
        v16 = v7 + 3;
        v17 = v6 + 3;
        v18 = v7 + 4;
        if ( *(_BYTE *)(a1 + v7 + 1) != v4[v6 + 1] || *(_BYTE *)(a1 + v7 + 2) != v4[v6 + 2] )
          return 0;
        v7 += 4LL;
        v11 = *(_BYTE *)(a1 + v16) != v4[v17];
        v12 = *(_BYTE *)(a1 + v16) != v4[v17] || a2 <= v18;
        v6 += 3;
      }
      else
      {
        if ( v8 == 2 || v8 == 7 )
        {
LABEL_21:
          v7 += 4LL;
          v6 += 3;
          v12 = a2 <= v7;
          v11 = 0;
          goto LABEL_10;
        }
        v13 = v7 + 1;
        if ( v8 )
        {
          ++v7;
          v12 = a2 <= v13;
          v11 = 0;
        }
        else
        {
          v14 = v4[++v6];
          v15 = *(_BYTE *)(a1 + v7 + 1) != v14;
          v7 += 2LL;
          v11 = v15;
          v12 = v15 || a2 <= v7;
        }
      }
    }
    else
    {
      if ( v8 || *(_BYTE *)(a1 + v7 + 1) != 4 )
        return 0;
      v10 = *(_BYTE *)(a1 + v7 + 2) != v9;
      v7 += 3LL;
      v11 = v10;
      v12 = v10 || a2 <= v7;
    }
LABEL_10:
    ++v6;
  }
  while ( !v12 );
  result = 0;
  if ( !v11 )
  {
    v4 += v6;
    return *v4 == 0;
  }
  return result;
}
