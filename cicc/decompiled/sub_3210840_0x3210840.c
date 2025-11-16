// Function: sub_3210840
// Address: 0x3210840
//
void __fastcall sub_3210840(_QWORD *a1, __int64 a2)
{
  __int64 v3; // rsi
  __int64 v4; // rsi
  __int64 v5; // rsi
  __int64 v6; // rax
  __int64 v7; // rsi
  __int64 v8; // rdx
  __int64 v9; // rbx
  int v10; // eax
  int v11; // [rsp+4h] [rbp-4Ch]
  __int64 v12; // [rsp+8h] [rbp-48h]
  __int64 v13; // [rsp+10h] [rbp-40h] BYREF
  __int64 v14[7]; // [rsp+18h] [rbp-38h] BYREF

  sub_3211700();
  if ( !a1[1] || !a1[99] || (unsigned __int16)(*(_WORD *)(a2 + 68) - 14) <= 4u || (*(_BYTE *)(a2 + 44) & 1) != 0 )
    return;
  v3 = *(_QWORD *)(a2 + 56);
  v13 = v3;
  if ( v3 )
  {
    sub_B96E90((__int64)&v13, v3, 1);
    v14[0] = v13;
    if ( v13 )
    {
      sub_B96E90((__int64)v14, v13, 1);
      if ( v14[0] )
      {
        if ( (unsigned int)sub_B10CE0((__int64)v14) )
        {
          v4 = v14[0];
LABEL_11:
          if ( v4 )
          {
            sub_B91220((__int64)v14, v4);
            v5 = v13;
            v6 = *(_QWORD *)(a2 + 24);
            goto LABEL_13;
          }
          v8 = *(_QWORD *)(a2 + 24);
          goto LABEL_22;
        }
        v6 = *(_QWORD *)(a2 + 24);
        v4 = v14[0];
        if ( a1[5] == v6 )
          goto LABEL_11;
        if ( v14[0] )
        {
          sub_B91220((__int64)v14, v14[0]);
          v6 = *(_QWORD *)(a2 + 24);
        }
        goto LABEL_26;
      }
    }
  }
  else
  {
    v14[0] = 0;
  }
  v8 = *(_QWORD *)(a2 + 24);
  v6 = v8;
  if ( a1[5] == v8 )
  {
LABEL_22:
    v5 = v13;
    v6 = v8;
    goto LABEL_13;
  }
LABEL_26:
  v9 = *(_QWORD *)(v6 + 56);
  v5 = v13;
  v12 = v6 + 48;
  if ( v6 + 48 == v9 )
    goto LABEL_13;
  do
  {
    if ( (unsigned __int16)(*(_WORD *)(v9 + 68) - 14) <= 4u )
      goto LABEL_30;
    if ( &v13 != (__int64 *)(v9 + 56) )
    {
      if ( v5 )
        sub_B91220((__int64)&v13, v5);
      v5 = *(_QWORD *)(v9 + 56);
      v13 = v5;
      if ( !v5 )
        goto LABEL_30;
      sub_B96E90((__int64)&v13, v5, 1);
      v5 = v13;
    }
    v14[0] = v5;
    if ( v5 )
    {
      sub_B96E90((__int64)v14, v5, 1);
      if ( v14[0] )
      {
        v10 = sub_B10CE0((__int64)v14);
        if ( v14[0] )
        {
          v11 = v10;
          sub_B91220((__int64)v14, v14[0]);
          v5 = v13;
          v10 = v11;
        }
        else
        {
          v5 = v13;
        }
        if ( v10 )
          break;
      }
      else
      {
        v5 = v13;
      }
    }
LABEL_30:
    if ( (*(_BYTE *)v9 & 4) == 0 && (*(_BYTE *)(v9 + 44) & 8) != 0 )
    {
      do
        v9 = *(_QWORD *)(v9 + 8);
      while ( (*(_BYTE *)(v9 + 44) & 8) != 0 );
    }
    v9 = *(_QWORD *)(v9 + 8);
  }
  while ( v12 != v9 );
  v6 = *(_QWORD *)(a2 + 24);
LABEL_13:
  a1[5] = v6;
  v14[0] = v5;
  if ( !v5 )
    return;
  sub_B96E90((__int64)v14, v5, 1);
  if ( !v14[0] )
    goto LABEL_47;
  if ( !(unsigned int)sub_B10CE0((__int64)v14) )
  {
    if ( v14[0] )
      sub_B91220((__int64)v14, v14[0]);
LABEL_47:
    v7 = v13;
    if ( v13 )
      goto LABEL_19;
    return;
  }
  if ( v14[0] )
    sub_B91220((__int64)v14, v14[0]);
  sub_320DBD0(a1, &v13);
  v7 = v13;
  if ( v13 )
LABEL_19:
    sub_B91220((__int64)&v13, v7);
}
