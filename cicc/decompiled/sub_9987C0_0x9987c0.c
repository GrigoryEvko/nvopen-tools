// Function: sub_9987C0
// Address: 0x9987c0
//
bool __fastcall sub_9987C0(__int64 a1, int a2, unsigned __int8 *a3)
{
  bool result; // al
  __int64 v5; // r13
  unsigned int v6; // r14d
  bool v7; // al
  __int64 v8; // r14
  __int64 v9; // rax
  unsigned int v10; // r14d
  char v11; // r15
  unsigned int v12; // r14d
  __int64 v13; // rax
  unsigned int v14; // r15d
  __int64 v15; // rsi
  int v16; // [rsp-3Ch] [rbp-3Ch]

  if ( a2 + 29 != *a3 )
    return 0;
  v5 = *((_QWORD *)a3 - 8);
  if ( *(_BYTE *)v5 == 17 )
  {
    v6 = *(_DWORD *)(v5 + 32);
    if ( !v6 )
      goto LABEL_30;
    if ( v6 <= 0x40 )
      v7 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v6) == *(_QWORD *)(v5 + 24);
    else
      v7 = v6 == (unsigned int)sub_C445E0(v5 + 24);
    goto LABEL_7;
  }
  v8 = *(_QWORD *)(v5 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v8 + 8) - 17 > 1 || *(_BYTE *)v5 > 0x15u )
  {
LABEL_8:
    if ( (unsigned __int8)sub_995B10((_QWORD **)a1, *((_QWORD *)a3 - 4)) )
      return *((_QWORD *)a3 - 8) == *(_QWORD *)(a1 + 8);
    return 0;
  }
  v9 = sub_AD7630(v5, 0);
  if ( !v9 || *(_BYTE *)v9 != 17 )
  {
    if ( *(_BYTE *)(v8 + 8) == 17 )
    {
      v16 = *(_DWORD *)(v8 + 32);
      if ( v16 )
      {
        v11 = 0;
        v12 = 0;
        while ( 1 )
        {
          v13 = sub_AD69F0(v5, v12);
          if ( !v13 )
            break;
          if ( *(_BYTE *)v13 != 13 )
          {
            if ( *(_BYTE *)v13 != 17 )
              goto LABEL_8;
            v14 = *(_DWORD *)(v13 + 32);
            if ( v14 )
            {
              if ( v14 <= 0x40 )
              {
                if ( *(_QWORD *)(v13 + 24) != 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v14) )
                  goto LABEL_8;
              }
              else if ( v14 != (unsigned int)sub_C445E0(v13 + 24) )
              {
                goto LABEL_8;
              }
            }
            v11 = 1;
          }
          if ( v16 == ++v12 )
          {
            if ( !v11 )
              goto LABEL_8;
            goto LABEL_30;
          }
        }
      }
    }
    goto LABEL_8;
  }
  v10 = *(_DWORD *)(v9 + 32);
  if ( !v10 )
    goto LABEL_30;
  if ( v10 <= 0x40 )
    v7 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v10) == *(_QWORD *)(v9 + 24);
  else
    v7 = v10 == (unsigned int)sub_C445E0(v9 + 24);
LABEL_7:
  if ( !v7 )
    goto LABEL_8;
LABEL_30:
  if ( *(_QWORD *)a1 )
    **(_QWORD **)a1 = v5;
  v15 = *((_QWORD *)a3 - 4);
  result = 1;
  if ( *(_QWORD *)(a1 + 8) != v15 )
  {
    if ( (unsigned __int8)sub_995B10((_QWORD **)a1, v15) )
      return *((_QWORD *)a3 - 8) == *(_QWORD *)(a1 + 8);
    return 0;
  }
  return result;
}
