// Function: sub_6E4BC0
// Address: 0x6e4bc0
//
__int64 __fastcall sub_6E4BC0(__int64 a1, __int64 a2)
{
  char v4; // al
  _DWORD *v5; // r13
  _DWORD *v6; // rax
  int v7; // edx
  _QWORD *v8; // rsi
  _QWORD *v9; // rcx
  int v10; // edi
  _QWORD *v11; // rax
  char v12; // al
  char v13; // al
  char v14; // al
  char v15; // al
  bool v16; // dl
  char v17; // cl
  char v18; // al
  char v19; // al
  char v20; // dl
  __int64 result; // rax
  __int64 v22; // rax
  char v23; // dl

  *(_QWORD *)(a1 + 68) = *(_QWORD *)(a2 + 68);
  *(_QWORD *)(a1 + 76) = *(_QWORD *)(a2 + 76);
  v4 = *(_BYTE *)(a2 + 16);
  if ( v4 != 1 )
  {
    if ( v4 == 2 && *(_BYTE *)(a1 + 16) == 2 )
    {
      v22 = *(_QWORD *)(a2 + 288);
      if ( v22 )
      {
        if ( v22 == *(_QWORD *)(a1 + 288) )
          goto LABEL_14;
      }
    }
LABEL_4:
    v5 = (_DWORD *)sub_6DEE30(a1);
    if ( !v5 )
      goto LABEL_14;
    v6 = (_DWORD *)sub_6DEE30(a2);
    v7 = v5[7];
    v8 = v5 + 7;
    if ( !v7 )
    {
      if ( v6 )
      {
        v7 = v6[7];
        if ( v7 )
        {
          v8 = v6 + 7;
          if ( !v5[9] )
            goto LABEL_9;
          goto LABEL_30;
        }
      }
      v7 = *(_DWORD *)(a1 + 68);
      v8 = (_QWORD *)(a1 + 68);
    }
    if ( !v5[9] )
    {
      if ( !v6 )
      {
LABEL_23:
        v9 = (_QWORD *)(a1 + 68);
        if ( v5[11] )
          goto LABEL_24;
LABEL_31:
        if ( !v6 )
        {
LABEL_12:
          v11 = (_QWORD *)(a1 + 76);
          if ( !v7 )
          {
LABEL_13:
            *(_QWORD *)(v5 + 7) = *v9;
            *(_QWORD *)(v5 + 9) = *v9;
            *(_QWORD *)(v5 + 11) = *v11;
            goto LABEL_14;
          }
LABEL_26:
          *(_QWORD *)(v5 + 7) = *v8;
          *(_QWORD *)(v5 + 9) = *v9;
          *(_QWORD *)(v5 + 11) = *v11;
          goto LABEL_14;
        }
        goto LABEL_11;
      }
LABEL_9:
      if ( v6[9] )
      {
        v9 = v6 + 9;
        if ( !v5[11] )
        {
LABEL_11:
          v10 = v6[11];
          v11 = v6 + 11;
          if ( !v10 )
            goto LABEL_12;
LABEL_25:
          if ( !v7 )
            goto LABEL_13;
          goto LABEL_26;
        }
LABEL_24:
        v11 = v5 + 11;
        goto LABEL_25;
      }
      goto LABEL_23;
    }
LABEL_30:
    v9 = v5 + 9;
    if ( v5[11] )
      goto LABEL_24;
    goto LABEL_31;
  }
  if ( *(_BYTE *)(a1 + 16) != 1 || *(_QWORD *)(a2 + 144) != *(_QWORD *)(a1 + 144) )
    goto LABEL_4;
LABEL_14:
  *(_QWORD *)(a1 + 8) = *(_QWORD *)(a2 + 8);
  v12 = *(_BYTE *)(a2 + 18) & 1 | *(_BYTE *)(a1 + 18) & 0xFE;
  *(_BYTE *)(a1 + 18) = v12;
  v13 = *(_BYTE *)(a2 + 18) & 2 | v12 & 0xFD;
  *(_BYTE *)(a1 + 18) = v13;
  v14 = *(_BYTE *)(a2 + 18) & 4 | v13 & 0xFB;
  *(_BYTE *)(a1 + 18) = v14;
  *(_BYTE *)(a1 + 18) = *(_BYTE *)(a2 + 18) & 0x40 | v14 & 0xBF;
  v15 = *(_BYTE *)(a2 + 19) & 1 | *(_BYTE *)(a1 + 19) & 0xFE;
  v16 = 0;
  *(_BYTE *)(a1 + 19) = v15;
  if ( (*(_BYTE *)(a2 + 19) & 2) != 0 )
  {
    v17 = *(_BYTE *)(a1 + 16);
    v16 = 1;
    if ( v17 != 3 )
    {
      v16 = 0;
      if ( v17 == 2 && *(_BYTE *)(a1 + 317) == 12 )
      {
        v23 = *(_BYTE *)(a1 + 320);
        if ( v23 == 11 )
          v23 = *(_BYTE *)(*(_QWORD *)(a1 + 328) + 176LL);
        v16 = v23 == 3;
      }
    }
  }
  v18 = (2 * v16) | v15 & 0xFD;
  *(_BYTE *)(a1 + 19) = v18;
  v19 = *(_BYTE *)(a2 + 19) & 4 | v18 & 0xFB;
  *(_BYTE *)(a1 + 19) = v19;
  *(_BYTE *)(a1 + 64) |= *(_BYTE *)(a2 + 64);
  v20 = *(_BYTE *)(a2 + 20) & 2 | *(_BYTE *)(a1 + 20) & 0xFD;
  *(_BYTE *)(a1 + 20) = v20;
  if ( (v19 & 2) != 0 )
    *(_QWORD *)(a1 + 120) = *(_QWORD *)(a2 + 120);
  *(_BYTE *)(a1 + 20) = *(_BYTE *)(a2 + 20) & 8 | v20 & 0xF7;
  result = *(_QWORD *)(a2 + 128);
  *(_QWORD *)(a1 + 128) = result;
  return result;
}
