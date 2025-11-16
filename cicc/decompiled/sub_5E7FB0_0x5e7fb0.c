// Function: sub_5E7FB0
// Address: 0x5e7fb0
//
__int64 *__fastcall sub_5E7FB0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // r14
  __int64 *result; // rax
  __int64 j; // r12
  _BYTE *v7; // r13
  _BYTE *v8; // r12
  int v9; // edi
  int v10; // eax
  int v11; // r9d
  unsigned int v12; // [rsp+4h] [rbp-4Ch]
  __int64 v13; // [rsp+8h] [rbp-48h]
  char v14; // [rsp+12h] [rbp-3Eh]
  char v15; // [rsp+13h] [rbp-3Dh]
  int i; // [rsp+14h] [rbp-3Ch]
  int v17; // [rsp+18h] [rbp-38h]
  int v18; // [rsp+1Ch] [rbp-34h]

  v15 = *(_BYTE *)(a1 + 140);
  v3 = a1;
  if ( v15 == 12 )
  {
    do
      v3 = *(_QWORD *)(v3 + 160);
    while ( *(_BYTE *)(v3 + 140) == 12 );
  }
  v18 = 1;
  v17 = 0;
  v4 = **(_QWORD **)(*(_QWORD *)v3 + 96LL);
  if ( v4 )
  {
    while ( *(_BYTE *)(v4 + 80) != 8 )
    {
LABEL_5:
      v4 = *(_QWORD *)(v4 + 16);
      if ( !v4 )
      {
        if ( (v18 & v17) != 0 )
        {
          if ( dword_4F077BC )
          {
            if ( dword_4F077B4 )
              *(_DWORD *)(a2 + 8) = 1;
          }
          else
          {
            *(_DWORD *)(a2 + 8) = 1;
          }
        }
        goto LABEL_9;
      }
    }
    v7 = *(_BYTE **)(*(_QWORD *)(v4 + 88) + 120LL);
    v13 = *(_QWORD *)(v4 + 88);
    if ( (unsigned int)sub_8D3410(v7) )
    {
      v7 = (_BYTE *)sub_8D40F0(v7);
      if ( (v7[140] & 0xFB) != 8 )
        goto LABEL_17;
    }
    else if ( (v7[140] & 0xFB) != 8 )
    {
LABEL_17:
      v8 = v7;
      if ( v15 == 11 )
      {
        v12 = 1;
        i = 0;
        v14 = 1;
        goto LABEL_26;
      }
      i = 0;
      goto LABEL_22;
    }
    v8 = v7;
    for ( i = sub_8D4C10(v7, unk_4F077C4 != 2) & 1; v8[140] == 12; v8 = (_BYTE *)*((_QWORD *)v8 + 20) )
      ;
    if ( v15 == 11 )
    {
      v12 = 1;
      v14 = 1;
      goto LABEL_23;
    }
LABEL_22:
    v12 = (*(_BYTE *)(*(_QWORD *)(v4 + 104) + 28LL) & 2) != 0;
    v14 = (*(_BYTE *)(*(_QWORD *)(v4 + 104) + 28LL) & 2) != 0;
LABEL_23:
    if ( i && (!*(_DWORD *)(a2 + 20) || !*(_DWORD *)(a2 + 24)) )
    {
      v11 = *(_DWORD *)(a2 + 36);
      *(_QWORD *)(a2 + 20) = 0x100000001LL;
      if ( v11 )
        sub_686750(4, 1635, a1 + 64, v4, a1);
      if ( (*(_BYTE *)(v13 + 145) & 0x20) == 0 )
      {
        if ( ((unsigned __int8)v14 & (dword_4F077BC == 0)) != 0 )
        {
          v14 &= dword_4F077BC == 0;
        }
        else if ( !(unsigned int)sub_8D5A50(v8) )
        {
          *(_DWORD *)(a2 + 8) = 1;
        }
      }
    }
LABEL_26:
    if ( (unsigned int)sub_8D32E0(v8) )
    {
      v9 = *(_DWORD *)(a2 + 36);
      *(_QWORD *)(a2 + 20) = 0x100000001LL;
      if ( v9 )
        sub_686750(4, 1636, a1 + 64, v4, a1);
      if ( (unsigned int)sub_8D3110(v8) )
        goto LABEL_47;
    }
    if ( !*(_DWORD *)(a2 + 12) && (unsigned int)sub_8D3110(v8) )
    {
LABEL_47:
      *(_DWORD *)(a2 + 12) = 1;
      if ( (unsigned __int8)(v8[140] - 9) > 2u )
        goto LABEL_32;
    }
    else if ( (unsigned __int8)(v8[140] - 9) > 2u )
    {
      goto LABEL_32;
    }
    sub_5E76F0((_BYTE *)a1, (_DWORD *)a2, v7, 0, (*(_BYTE *)(v13 + 144) & 0x20) != 0, v12);
LABEL_32:
    if ( v14 )
    {
      v10 = 0;
      v17 = 1;
      if ( i )
        v10 = v18;
      v18 = v10;
    }
    goto LABEL_5;
  }
LABEL_9:
  result = *(__int64 **)(a1 + 168);
  for ( j = *result; j; j = *(_QWORD *)j )
  {
    while ( (*(_BYTE *)(j + 96) & 3) == 0 )
    {
      j = *(_QWORD *)j;
      if ( !j )
        return result;
    }
    result = (__int64 *)sub_5E76F0((_BYTE *)a1, (_DWORD *)a2, *(_BYTE **)(j + 40), j, 0, 0);
  }
  return result;
}
