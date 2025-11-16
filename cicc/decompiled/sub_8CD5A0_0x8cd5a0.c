// Function: sub_8CD5A0
// Address: 0x8cd5a0
//
__int64 __fastcall sub_8CD5A0(__int64 *a1)
{
  __int64 v1; // r13
  __int64 v2; // r12
  __int64 *v3; // rbx
  __int64 v4; // r14
  bool v5; // r15
  __int64 v6; // rcx
  __int64 v7; // r8
  unsigned int v8; // r14d
  char v9; // al
  char v11; // al
  char v12; // dl
  __int64 *v13; // rax
  __int64 v14; // rsi
  char v15; // al
  __int64 v16; // rbx
  __int64 v17; // r15
  __int64 v18; // r14
  char v19; // al
  __int64 v20; // rax
  __int64 *v21; // rax
  __int64 v22; // rcx
  bool v23; // [rsp+8h] [rbp-38h]
  __int64 *v24; // [rsp+8h] [rbp-38h]

  v1 = (__int64)a1;
  v2 = (__int64)a1;
  v3 = (__int64 *)a1[4];
  v4 = *a1;
  if ( v3 && (v1 = *v3, a1 == (__int64 *)*v3) )
  {
    v2 = v3[1];
    if ( !v2 )
      v2 = *v3;
    v5 = 0;
    if ( !(unsigned int)sub_8C6470(v2) )
    {
LABEL_4:
      if ( (*(_BYTE *)(*(_QWORD *)v2 + 83LL) & 0x40) != 0 )
        goto LABEL_5;
      goto LABEL_22;
    }
  }
  else
  {
    v5 = 0;
    if ( !(unsigned int)sub_8C6470((__int64)a1) )
      goto LABEL_4;
  }
  v5 = (unsigned int)sub_8C6470(v1) != 0;
  if ( (*(_BYTE *)(*(_QWORD *)v2 + 83LL) & 0x40) != 0 )
  {
LABEL_5:
    if ( v2 != v1 )
    {
      if ( v4 )
      {
        v23 = 0;
        goto LABEL_8;
      }
LABEL_15:
      v8 = 1;
LABEL_16:
      sub_8C6CA0(v2, v1, 6u, (_QWORD *)(v1 + 64));
      sub_8C6CA0(v1, v2, 6u, (_QWORD *)(v2 + 64));
      return v8;
    }
    goto LABEL_32;
  }
LABEL_22:
  v23 = (*(_BYTE *)(*(_QWORD *)v1 + 83LL) & 0x40) == 0;
  if ( v2 == v1 )
  {
LABEL_32:
    if ( !v3 )
      goto LABEL_15;
    if ( *(_BYTE *)(v2 + 140) != 2 )
      goto LABEL_15;
    v15 = *(_BYTE *)(v2 + 161);
    if ( (v15 & 8) == 0 || (*(_BYTE *)(v2 + 89) & 4) != 0 || (**(_BYTE **)(v2 + 176) & 1) == 0 )
      goto LABEL_15;
    v24 = *(__int64 **)(v2 + 168);
    if ( (v15 & 0x10) != 0 )
      v24 = *(__int64 **)(*(_QWORD *)(v2 + 168) + 96LL);
    if ( !v24 )
      goto LABEL_15;
    while ( 1 )
    {
      v16 = *v24;
      v17 = *(_QWORD *)(*(_QWORD *)*v24 + 32LL);
      v18 = sub_880F80(*v24);
      if ( v17 )
        break;
LABEL_45:
      v24 = (__int64 *)v24[15];
      if ( !v24 )
        goto LABEL_15;
    }
    while ( 1 )
    {
      if ( *(_DWORD *)(v17 + 40) == -1 || v18 == sub_880F80(v17) || !(unsigned int)sub_8C7F70(v17, v16) )
        goto LABEL_44;
      if ( !(unsigned int)sub_8C6B40(v17) )
        goto LABEL_64;
      v19 = *(_BYTE *)(v17 + 80);
      if ( v19 == 2 )
      {
        if ( sub_8C7520((__int64 **)v2, *(__int64 ***)(*(_QWORD *)(v17 + 88) + 128LL)) )
        {
          if ( (unsigned int)sub_8CD200((__int64 *)v2, *(_QWORD *)(*(_QWORD *)(v17 + 88) + 128LL)) )
          {
            v21 = *(__int64 **)(v17 + 88);
            if ( v21 == v24 )
              goto LABEL_44;
            if ( v21 )
            {
              if ( dword_4F07588 )
              {
                v22 = v24[4];
                if ( v21[4] == v22 )
                {
                  if ( v22 )
                    goto LABEL_44;
                }
              }
            }
          }
        }
        v19 = *(_BYTE *)(v17 + 80);
      }
      if ( (unsigned __int8)(v19 - 4) > 2u && (v19 != 3 || !*(_BYTE *)(v17 + 104)) )
      {
LABEL_64:
        v20 = sub_87D520(v17);
        if ( v20 && (*(_BYTE *)(v20 - 8) & 2) == 0 )
          *(_BYTE *)(v20 + 90) |= 8u;
      }
LABEL_44:
      v17 = *(_QWORD *)(v17 + 8);
      if ( !v17 )
        goto LABEL_45;
    }
  }
  if ( !v4 )
  {
    if ( (*(_BYTE *)(*(_QWORD *)v1 + 83LL) & 0x40) == 0 )
    {
LABEL_25:
      v8 = 1;
      goto LABEL_26;
    }
    goto LABEL_15;
  }
LABEL_8:
  v8 = sub_8C7610(v2);
  if ( !v8 )
  {
    sub_8C7090(6, v2);
    return v8;
  }
  v9 = *(_BYTE *)(v2 + 140);
  if ( (unsigned __int8)(v9 - 9) <= 2u )
  {
    v8 = sub_8CE860(v2);
  }
  else
  {
    if ( v9 != 2 || (*(_BYTE *)(v2 + 161) & 8) == 0 )
    {
      if ( !(unsigned int)sub_8D97D0(v2, v1, 0, v6, v7) || !(unsigned int)sub_8DBAE0(v2, v1) )
      {
        v8 = 0;
        sub_8C6700((__int64 *)v2, (unsigned int *)(v1 + 64), 0x42Au, 0x425u);
        sub_8C7090(6, v2);
        return v8;
      }
      if ( !v23 )
        goto LABEL_15;
      goto LABEL_25;
    }
    v8 = sub_8C7CC0(v2);
  }
  if ( !v8 )
    return v8;
  if ( !v23 )
    goto LABEL_16;
LABEL_26:
  v11 = *(_BYTE *)(v2 + 140);
  v12 = *(_BYTE *)(v1 + 140);
  if ( v11 == v12 )
  {
    if ( v5 && (*(_QWORD *)(v2 + 128) != *(_QWORD *)(v1 + 128) || *(_DWORD *)(v2 + 136) != *(_DWORD *)(v1 + 136)) )
      goto LABEL_29;
  }
  else if ( v5 || (unsigned __int8)(v11 - 9) > 1u || (unsigned __int8)(v12 - 9) > 1u )
  {
    goto LABEL_29;
  }
  if ( ((*(_BYTE *)(v1 + 143) ^ *(_BYTE *)(v2 + 143)) & 0x10) == 0
    && ((*(_BYTE *)(v1 + 88) ^ *(_BYTE *)(v2 + 88)) & 0x73) == 0 )
  {
    goto LABEL_16;
  }
LABEL_29:
  v13 = *(__int64 **)(v2 + 32);
  v14 = v2;
  if ( v13 )
    v14 = *v13;
  v8 = 0;
  sub_8C6700((__int64 *)v2, (unsigned int *)(v14 + 64), 0x42Au, 0x425u);
  return v8;
}
