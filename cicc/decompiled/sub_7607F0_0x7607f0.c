// Function: sub_7607F0
// Address: 0x7607f0
//
__int64 __fastcall sub_7607F0(__int64 a1, char a2)
{
  __int64 v2; // rbx
  __int64 v3; // rax
  char v4; // dl
  unsigned int v5; // r13d
  _QWORD *v7; // r13
  __int64 v8; // rdi
  __int64 *v9; // rax
  __int64 v10; // r15
  __int64 k; // r12
  __int64 *v12; // rax
  __int64 v13; // r13
  char v14; // al
  _QWORD *i; // r15
  __int64 v16; // r14
  __int64 j; // r12

  v2 = a1;
  v3 = sub_72A270(a1, a2);
  if ( !v3 )
  {
    if ( a2 != 23 || *(_BYTE *)(a1 + 28) != 2 )
      return 0;
    v14 = *(_BYTE *)(a1 + 29);
    if ( (v14 & 0x40) == 0 )
    {
      if ( (*(_BYTE *)(a1 - 8) & 2) == 0 )
        *(_BYTE *)(a1 + 29) = v14 | 0x40;
      return 0;
    }
    return 1;
  }
  if ( dword_4F08010 && (*(_BYTE *)(a1 - 8) & 2) == 0 )
  {
    if ( !dword_4F07588 )
      goto LABEL_5;
    v12 = *(__int64 **)(a1 + 32);
    if ( !v12 )
      goto LABEL_5;
    v13 = *v12;
    if ( a1 == *v12 )
      goto LABEL_5;
    if ( (*(_BYTE *)(v13 - 8) & 2) == 0 )
      goto LABEL_5;
    sub_7604D0(*v12, a2);
    if ( a2 != 11 )
      goto LABEL_5;
    if ( dword_4F077C4 == 2 && *(char *)(v13 + 192) < 0 && !*(_BYTE *)(v13 + 172) )
      sub_7605A0(v13);
    return 1;
  }
  v4 = *(_BYTE *)(v3 + 88);
  if ( (v4 & 8) != 0 )
  {
LABEL_5:
    v5 = 1;
    goto LABEL_6;
  }
  *(_BYTE *)(v3 + 88) = v4 | 8;
  v5 = unk_4D03B60;
  if ( unk_4D03B60 )
  {
    v5 = dword_4F07588;
    if ( !dword_4F07588 )
      goto LABEL_24;
    goto LABEL_35;
  }
  if ( a2 != 6 )
  {
    if ( ((a2 - 7) & 0xFB) == 0 )
      goto LABEL_14;
    if ( !dword_4F07588 )
      return 0;
LABEL_35:
    v5 = 0;
    if ( !a1 )
      goto LABEL_24;
LABEL_21:
    v9 = *(__int64 **)(v2 + 32);
    if ( !v9 )
      goto LABEL_24;
    goto LABEL_22;
  }
  if ( (unsigned __int8)(*(_BYTE *)(a1 + 140) - 9) <= 2u && *(char *)(a1 + 141) >= 0 )
  {
LABEL_14:
    if ( (*(_BYTE *)(a1 + 90) & 2) == 0 )
    {
      v5 = 1;
      if ( (*(_BYTE *)(v3 + 89) & 4) != 0 && (unk_4D048F8 || (*(_BYTE *)(a1 - 8) & 2) != 0 || a2 == 8) )
      {
        v7 = *(_QWORD **)(*(_QWORD *)(v3 + 40) + 32LL);
        sub_75C0C0(v7, 6);
        v8 = (__int64)v7;
        v5 = 1;
        sub_75C030(v8);
      }
    }
    if ( !dword_4F07588 )
      goto LABEL_24;
    goto LABEL_21;
  }
  if ( !dword_4F07588 )
    return 0;
  v9 = *(__int64 **)(a1 + 32);
  if ( !v9 )
    return v5;
LABEL_22:
  v10 = *v9;
  if ( v2 != *v9 && (*(_BYTE *)(v10 - 8) & 2) != 0 )
  {
    sub_7604D0(*v9, a2);
    if ( a2 == 11 )
    {
      if ( dword_4F077C4 == 2 && *(char *)(v10 + 192) < 0 && !*(_BYTE *)(v10 + 172) )
        sub_7605A0(v10);
      goto LABEL_25;
    }
LABEL_6:
    if ( a2 == 7 )
    {
      while ( 1 )
      {
        v2 = *(_QWORD *)(v2 + 232);
        if ( !v2 )
          break;
        if ( (*(_BYTE *)(v2 + 88) & 8) == 0 )
          sub_7604D0(v2, 7u);
      }
    }
    return v5;
  }
LABEL_24:
  if ( a2 != 11 )
    goto LABEL_6;
LABEL_25:
  if ( !*(_BYTE *)(v2 + 172) )
  {
    if ( (unsigned __int8)(*(_BYTE *)(v2 + 174) - 1) <= 1u )
    {
      for ( i = *(_QWORD **)(v2 + 176); i; i = (_QWORD *)*i )
      {
        v16 = i[1];
        if ( (*(_BYTE *)(v16 + 205) & 0x1C) != 0x10 )
        {
          sub_7604D0(i[1], 0xBu);
          for ( j = *(_QWORD *)(v16 + 112); j; j = *(_QWORD *)(j + 112) )
          {
            if ( v16 != *(_QWORD *)(j + 272) )
              break;
            sub_7604D0(j, 0xBu);
          }
        }
      }
    }
    for ( k = *(_QWORD *)(v2 + 112); k; k = *(_QWORD *)(k + 112) )
    {
      if ( v2 != *(_QWORD *)(k + 272) )
        break;
      sub_7604D0(k, 0xBu);
    }
  }
  return v5;
}
