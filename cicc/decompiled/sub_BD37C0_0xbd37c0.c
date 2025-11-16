// Function: sub_BD37C0
// Address: 0xbd37c0
//
__int64 __fastcall sub_BD37C0(__int64 a1, __int64 a2)
{
  unsigned int v2; // r10d
  __int64 v3; // rcx
  __int64 v4; // r9
  __int64 v6; // rsi
  __int64 v7; // rdx
  char *v8; // rax
  __int64 v9; // rbx
  char *v10; // r11
  __int64 v11; // rdx
  char *v12; // rdx
  __int64 v13; // rax

  v3 = *(_QWORD *)(a2 + 56);
  v4 = a2 + 48;
  v6 = *(_QWORD *)(a1 + 16);
  LOBYTE(v2) = v6 != 0 && v3 != v4;
  if ( !(_BYTE)v2 )
    return 0;
  do
  {
    if ( !v3 )
      BUG();
    v7 = 32LL * (*(_DWORD *)(v3 - 20) & 0x7FFFFFF);
    if ( (*(_BYTE *)(v3 - 17) & 0x40) != 0 )
    {
      v8 = *(char **)(v3 - 32);
      v9 = v7 >> 5;
      v10 = &v8[v7];
      v11 = v7 >> 7;
      if ( v11 )
        goto LABEL_5;
    }
    else
    {
      v10 = (char *)(v3 - 24);
      v9 = v7 >> 5;
      v8 = (char *)(v3 - 24 - v7);
      v11 = v7 >> 7;
      if ( v11 )
      {
LABEL_5:
        v12 = &v8[128 * v11];
        while ( a1 != *(_QWORD *)v8 )
        {
          if ( a1 == *((_QWORD *)v8 + 4) )
          {
            v8 += 32;
            goto LABEL_11;
          }
          if ( a1 == *((_QWORD *)v8 + 8) )
          {
            v8 += 64;
            goto LABEL_11;
          }
          if ( a1 == *((_QWORD *)v8 + 12) )
          {
            v8 += 96;
            goto LABEL_11;
          }
          v8 += 128;
          if ( v12 == v8 )
          {
            v9 = (v10 - v8) >> 5;
            if ( v9 != 2 )
              goto LABEL_20;
            goto LABEL_28;
          }
        }
        goto LABEL_11;
      }
    }
    if ( v9 == 2 )
      goto LABEL_28;
LABEL_20:
    if ( v9 != 3 )
    {
      if ( v9 != 1 || a1 != *(_QWORD *)v8 )
        goto LABEL_12;
      goto LABEL_11;
    }
    if ( a1 != *(_QWORD *)v8 )
    {
      v8 += 32;
LABEL_28:
      if ( a1 != *(_QWORD *)v8 )
      {
        v8 += 32;
        if ( a1 != *(_QWORD *)v8 )
          goto LABEL_12;
      }
    }
LABEL_11:
    if ( v10 != v8 )
      return v2;
LABEL_12:
    v13 = *(_QWORD *)(v6 + 24);
    if ( *(_BYTE *)v13 > 0x1Cu && a2 == *(_QWORD *)(v13 + 40) )
      return v2;
    v6 = *(_QWORD *)(v6 + 8);
    v3 = *(_QWORD *)(v3 + 8);
  }
  while ( v6 && v4 != v3 );
  return 0;
}
