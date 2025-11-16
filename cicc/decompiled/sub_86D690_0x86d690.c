// Function: sub_86D690
// Address: 0x86d690
//
unsigned int *__fastcall sub_86D690(__int64 a1, __int64 a2)
{
  _QWORD *v4; // r14
  __int64 v5; // r15
  __int64 v6; // r13
  char v7; // cl
  bool v8; // al
  __int64 v9; // rax
  unsigned int *result; // rax
  __int64 v11; // r13
  _QWORD *v12; // rax
  _QWORD *v13; // r13
  __int64 v14; // rsi
  __int64 v15; // rax
  int v16; // r14d
  __int64 v17; // rdi
  __int64 v18; // r10
  __int64 v19; // rsi
  __int64 v20; // r9
  __int64 v21; // [rsp+8h] [rbp-48h]
  __int64 v22; // [rsp+10h] [rbp-40h]
  _DWORD *v23; // [rsp+18h] [rbp-38h]

  v4 = *(_QWORD **)(*(_QWORD *)(a2 + 8) + 80LL);
  if ( *v4 )
    *(_QWORD *)(*(_QWORD *)(a2 + 32) + 24LL) = a1;
  else
    *v4 = a1;
  *(_QWORD *)(a2 + 32) = a1;
  v5 = *(_QWORD *)(a1 + 8);
  if ( !v5 )
  {
    if ( v4[1] )
      sub_6851C0(0x7Cu, (_DWORD *)(a1 + 40));
    else
      v4[1] = a1;
    goto LABEL_7;
  }
  v6 = *(_QWORD *)(a1 + 16);
  v7 = *(_BYTE *)(v5 + 173);
  if ( !v6 )
  {
    v8 = v7 != 1;
LABEL_6:
    *(_BYTE *)(a2 + 5) = (16 * v8) | *(_BYTE *)(a2 + 5) & 0xEF;
    if ( v7 != 1 )
      goto LABEL_7;
    v6 = v5;
    if ( !v4[2] )
    {
LABEL_21:
      v4[2] = a1;
      *(_QWORD *)(a2 + 24) = v6;
      *(_QWORD *)(a2 + 40) = a1;
      goto LABEL_7;
    }
    goto LABEL_18;
  }
  v8 = 1;
  if ( *(_BYTE *)(v6 + 173) != 1 )
    goto LABEL_6;
  if ( v7 == 1 )
  {
    *(_BYTE *)(a2 + 5) &= ~0x10u;
  }
  else
  {
    *(_BYTE *)(a2 + 5) |= 0x10u;
    v5 = v6;
  }
  if ( !v4[2] )
    goto LABEL_21;
LABEL_18:
  if ( (int)sub_621060(v5, *(_QWORD *)(a2 + 24)) <= 0 )
  {
    v12 = (_QWORD *)v4[2];
    v13 = v4 + 2;
    if ( v12 )
    {
      v23 = (_DWORD *)(a1 + 40);
      do
      {
        v14 = v12[1];
        v15 = v12[2];
        if ( v15 && *(_BYTE *)(v14 + 173) != 1 )
          v14 = v15;
        v16 = sub_621060(v5, v14);
        if ( !v16 )
          goto LABEL_36;
        if ( HIDWORD(qword_4F077B4) )
        {
          v17 = *(_QWORD *)(a1 + 16);
          v18 = *(_QWORD *)(a1 + 8);
          v19 = *(_QWORD *)(*v13 + 8LL);
          v20 = *(_QWORD *)(*v13 + 16LL);
          if ( v17 && *(_BYTE *)(v17 + 173) == 1 )
          {
            if ( *(_BYTE *)(v18 + 173) != 1 )
              v18 = *(_QWORD *)(a1 + 16);
          }
          else
          {
            v17 = *(_QWORD *)(a1 + 8);
          }
          if ( v20 && *(_BYTE *)(v20 + 173) == 1 )
          {
            if ( *(_BYTE *)(v19 + 173) != 1 )
              v19 = *(_QWORD *)(*v13 + 16LL);
          }
          else
          {
            v20 = *(_QWORD *)(*v13 + 8LL);
          }
          v21 = v20;
          v22 = v18;
          if ( (int)sub_621060(v17, v19) >= 0 && (int)sub_621060(v22, v21) <= 0 )
LABEL_36:
            sub_6854F0(8u, 0x62Au, v23, (_QWORD *)(*v13 + 40LL));
        }
        v12 = (_QWORD *)*v13;
        if ( v16 < 0 )
          break;
        v13 = v12 + 4;
        v12 = (_QWORD *)v12[4];
      }
      while ( v12 );
    }
    *(_QWORD *)(a1 + 32) = v12;
    *v13 = a1;
    if ( !*(_QWORD *)(a1 + 32) )
      *(_QWORD *)(a2 + 40) = a1;
  }
  else
  {
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 32LL) = a1;
    *(_QWORD *)(a2 + 40) = a1;
    *(_QWORD *)(a2 + 24) = v6;
  }
LABEL_7:
  v9 = sub_86B2C0(4);
  sub_86CBE0(v9);
  result = &dword_4D044B4;
  if ( dword_4D044B4 )
  {
    result = &dword_4F077C4;
    if ( dword_4F077C4 == 2 )
    {
      v11 = qword_4D03B98 + 176LL * unk_4D03B90;
      sub_733780(0x15u, *(_QWORD *)a1, 0, 2, 0);
      if ( !*(_DWORD *)v11 )
        *(_QWORD *)(v11 + 128) = qword_4F06BC0;
      if ( a2 == v11 || (result = *(unsigned int **)(a2 + 8), *((_QWORD *)result + 9) == *(_QWORD *)(v11 + 8)) )
      {
        result = (unsigned int *)qword_4F06BC0;
        *(_QWORD *)(a2 + 128) = qword_4F06BC0;
      }
    }
  }
  return result;
}
