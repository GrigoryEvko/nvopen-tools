// Function: sub_996420
// Address: 0x996420
//
__int64 __fastcall sub_996420(_QWORD **a1, int a2, unsigned __int8 *a3)
{
  __int64 result; // rax
  __int64 v5; // r13
  unsigned int v6; // r14d
  bool v7; // al
  __int64 v8; // rsi
  __int64 v9; // rdx
  __int64 v10; // r14
  __int64 v11; // rax
  unsigned int v12; // r14d
  char v13; // r15
  unsigned int v14; // r14d
  __int64 v15; // rax
  unsigned int v16; // r15d
  int v17; // [rsp-3Ch] [rbp-3Ch]

  if ( a2 + 29 != *a3 )
    return 0;
  v5 = *((_QWORD *)a3 - 8);
  if ( *(_BYTE *)v5 == 17 )
  {
    v6 = *(_DWORD *)(v5 + 32);
    if ( !v6 )
      goto LABEL_31;
    if ( v6 <= 0x40 )
      v7 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v6) == *(_QWORD *)(v5 + 24);
    else
      v7 = v6 == (unsigned int)sub_C445E0(v5 + 24);
    goto LABEL_7;
  }
  v10 = *(_QWORD *)(v5 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v10 + 8) - 17 > 1 || *(_BYTE *)v5 > 0x15u )
    goto LABEL_8;
  v11 = sub_AD7630(v5, 0);
  if ( !v11 || *(_BYTE *)v11 != 17 )
  {
    if ( *(_BYTE *)(v10 + 8) == 17 )
    {
      v17 = *(_DWORD *)(v10 + 32);
      if ( v17 )
      {
        v13 = 0;
        v14 = 0;
        while ( 1 )
        {
          v15 = sub_AD69F0(v5, v14);
          if ( !v15 )
            break;
          if ( *(_BYTE *)v15 != 13 )
          {
            if ( *(_BYTE *)v15 != 17 )
              goto LABEL_8;
            v16 = *(_DWORD *)(v15 + 32);
            if ( v16 )
            {
              if ( v16 <= 0x40 )
              {
                if ( *(_QWORD *)(v15 + 24) != 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v16) )
                  goto LABEL_8;
              }
              else if ( v16 != (unsigned int)sub_C445E0(v15 + 24) )
              {
                goto LABEL_8;
              }
            }
            v13 = 1;
          }
          if ( v17 == ++v14 )
          {
            if ( !v13 )
              goto LABEL_8;
            goto LABEL_31;
          }
        }
      }
    }
    goto LABEL_8;
  }
  v12 = *(_DWORD *)(v11 + 32);
  if ( v12 )
  {
    if ( v12 <= 0x40 )
      v7 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v12) == *(_QWORD *)(v11 + 24);
    else
      v7 = v12 == (unsigned int)sub_C445E0(v11 + 24);
LABEL_7:
    if ( !v7 )
    {
LABEL_8:
      v8 = *((_QWORD *)a3 - 4);
      goto LABEL_9;
    }
  }
LABEL_31:
  if ( *a1 )
    **a1 = v5;
  v8 = *((_QWORD *)a3 - 4);
  if ( v8 )
  {
    *a1[1] = v8;
    return 1;
  }
LABEL_9:
  result = sub_995B10(a1, v8);
  if ( !(_BYTE)result )
    return 0;
  v9 = *((_QWORD *)a3 - 8);
  if ( !v9 )
    return 0;
  *a1[1] = v9;
  return result;
}
