// Function: sub_2A9FE50
// Address: 0x2a9fe50
//
__int64 __fastcall sub_2A9FE50(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  _QWORD *v4; // rsi
  __int64 v5; // r12
  unsigned int v6; // r13d
  bool v7; // al
  __int64 v8; // r13
  __int64 v9; // rdx
  _BYTE *v10; // rax
  unsigned int v11; // r13d
  int v12; // r13d
  char v13; // r15
  unsigned int v14; // r14d
  __int64 v15; // rax
  unsigned int v16; // r15d

  if ( *(_BYTE *)a2 != 63 || (*(_DWORD *)(a2 + 4) & 0x7FFFFFF) != 2 )
    return 0;
  v3 = *(_QWORD *)(a1 + 8);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
  {
    v4 = *(_QWORD **)(a2 - 8);
    if ( v3 != *v4 )
      return 0;
  }
  else
  {
    if ( *(_QWORD *)(a2 - 64) != v3 )
      return 0;
    v4 = (_QWORD *)(a2 - 64);
  }
  v5 = v4[4];
  if ( *(_BYTE *)v5 != 17 )
  {
    v8 = *(_QWORD *)(v5 + 8);
    v9 = (unsigned int)*(unsigned __int8 *)(v8 + 8) - 17;
    if ( (unsigned int)v9 > 1 || *(_BYTE *)v5 > 0x15u )
      return 0;
    v10 = sub_AD7630(v5, 0, v9);
    if ( !v10 || *v10 != 17 )
    {
      if ( *(_BYTE *)(v8 + 8) == 17 )
      {
        v12 = *(_DWORD *)(v8 + 32);
        if ( v12 )
        {
          v13 = 0;
          v14 = 0;
          while ( 1 )
          {
            v15 = sub_AD69F0((unsigned __int8 *)v5, v14);
            if ( !v15 )
              break;
            if ( *(_BYTE *)v15 != 13 )
            {
              if ( *(_BYTE *)v15 != 17 )
                return 0;
              v16 = *(_DWORD *)(v15 + 32);
              if ( v16 <= 0x40 )
              {
                if ( *(_QWORD *)(v15 + 24) != 1 )
                  return 0;
              }
              else if ( (unsigned int)sub_C444A0(v15 + 24) != v16 - 1 )
              {
                return 0;
              }
              v13 = 1;
            }
            if ( v12 == ++v14 )
            {
              if ( v13 )
                goto LABEL_10;
              return 0;
            }
          }
        }
      }
      return 0;
    }
    v11 = *((_DWORD *)v10 + 8);
    if ( v11 > 0x40 )
    {
      v7 = v11 - 1 == (unsigned int)sub_C444A0((__int64)(v10 + 24));
      goto LABEL_9;
    }
    if ( *((_QWORD *)v10 + 3) == 1 )
      goto LABEL_10;
    return 0;
  }
  v6 = *(_DWORD *)(v5 + 32);
  if ( v6 <= 0x40 )
    v7 = *(_QWORD *)(v5 + 24) == 1;
  else
    v7 = v6 - 1 == (unsigned int)sub_C444A0(v5 + 24);
LABEL_9:
  if ( !v7 )
    return 0;
LABEL_10:
  if ( *(_QWORD *)a1 )
    **(_QWORD **)a1 = v5;
  return 1;
}
