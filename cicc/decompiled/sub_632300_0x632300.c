// Function: sub_632300
// Address: 0x632300
//
__int64 __fastcall sub_632300(__int64 *a1, int a2, unsigned int a3, __int64 a4)
{
  __int64 *v4; // r15
  __int64 v6; // rax
  char v7; // dl
  _BOOL4 v8; // r14d
  bool v9; // bl
  __int64 v10; // r12
  __int64 v11; // rsi
  __int64 result; // rax
  __int64 v13; // rdx
  char v14; // al
  __int64 v15; // rbx
  __int64 v16; // rax
  __int64 v17; // r8
  char v18; // dl
  __int64 v19; // rsi
  char v20; // dl
  __int64 v21; // rax
  __int64 v22; // rdi
  int v23; // eax
  __int64 v24; // rax
  bool v25; // [rsp+7h] [rbp-49h]
  _BOOL4 v29; // [rsp+18h] [rbp-38h]
  __int64 v30; // [rsp+18h] [rbp-38h]

  v4 = a1 + 17;
  v6 = *a1;
  if ( *a1 )
  {
    v7 = *(_BYTE *)(v6 + 80);
    if ( v7 == 9 || v7 == 7 )
    {
      v16 = *(_QWORD *)(v6 + 88);
    }
    else
    {
      v8 = 0;
      v9 = 0;
      if ( v7 != 21 )
        goto LABEL_5;
      v16 = *(_QWORD *)(*(_QWORD *)(v6 + 88) + 192LL);
    }
    v8 = v16 != 0;
    v9 = v16 != 0;
  }
  else
  {
    v8 = 0;
    v9 = 0;
  }
LABEL_5:
  *((_BYTE *)a1 + 176) |= 1u;
  if ( word_4F06418[0] == 28 && !a1[41] )
    goto LABEL_35;
  v10 = sub_6BB940(a1, 1, 1);
  if ( a1[41] )
  {
    if ( !(unsigned int)sub_8D97B0(a1[36]) )
    {
      v19 = sub_6E1A20(a1[41]);
      sub_6851C0(146, v19);
    }
    sub_6E1BF0(a1 + 41);
  }
  else
  {
    sub_690C20();
  }
  if ( v10 )
  {
    v25 = (a1[22] & 8) != 0;
    v29 = *(_QWORD *)(v10 + 16) != 0;
    if ( (unsigned int)sub_6E1A80(v10) )
    {
      v21 = sub_72C9A0();
      *((_BYTE *)a1 + 177) |= 2u;
      v22 = a1[36];
      a1[17] = v21;
      if ( (unsigned int)sub_8D23E0(v22) )
        a1[36] = sub_72C930();
    }
    else if ( v9 && (*((_BYTE *)a1 + 131) & 0x10) != 0 && (unsigned int)sub_8D3410(a1[36]) )
    {
      sub_692C90(a1, v10);
    }
    else if ( !(unsigned int)sub_8D3880(a1[36])
           || !(unsigned int)sub_8D3880(a1[36])
           || !(unsigned int)sub_6320D0(v10, a1 + 36, (__int64)v4, (__int64)v4) )
    {
      v11 = a1[36];
      *((_BYTE *)a1 + 178) |= 2u;
      sub_694AA0(v10, v11, v8, a3, v4);
    }
    sub_6E1990(v10);
    *((_BYTE *)a1 + 176) = a1[22] & 0xF7 | (8 * v25);
  }
  else
  {
LABEL_35:
    sub_6966B0(a1[36], v4, a4);
    v29 = 1;
  }
  if ( (unsigned int)sub_8D3BB0(a1[36]) && (*((_BYTE *)a1 + 177) & 2) == 0 )
  {
    if ( a1[18] )
      goto LABEL_21;
    goto LABEL_34;
  }
  if ( (a1[22] & 8) != 0 && !a1[18] )
LABEL_34:
    sub_630880(v4, 0);
LABEL_21:
  result = v29;
  if ( !v29 )
    goto LABEL_24;
  result = a1[17];
  if ( result )
    goto LABEL_23;
  result = a1[18];
  if ( result )
  {
    v20 = *(_BYTE *)(result + 48);
    if ( (unsigned __int8)(v20 - 3) <= 1u )
    {
      result = *(_QWORD *)(result + 56);
      *(_BYTE *)(result + 26) |= 4u;
    }
    else
    {
      if ( (v20 & 0xFB) == 2 )
      {
        result = *(_QWORD *)(result + 56);
LABEL_23:
        *(_BYTE *)(result + 170) |= 4u;
        goto LABEL_24;
      }
      if ( v20 == 5 )
      {
        result = *(_QWORD *)(result + 64);
        if ( result )
          *(_BYTE *)(result + 26) |= 4u;
      }
    }
  }
LABEL_24:
  if ( !v9 )
    return result;
  v13 = *a1;
  v14 = *(_BYTE *)(*a1 + 80);
  if ( v14 == 9 || v14 == 7 )
  {
    v15 = *(_QWORD *)(v13 + 88);
    result = sub_8D23E0(*(_QWORD *)(v15 + 120));
    if ( !(_DWORD)result )
      return result;
    goto LABEL_38;
  }
  if ( v14 != 21 )
    BUG();
  v15 = *(_QWORD *)(*(_QWORD *)(v13 + 88) + 192LL);
  result = sub_8D23E0(*(_QWORD *)(v15 + 120));
  if ( (_DWORD)result )
  {
LABEL_38:
    result = sub_8D3410(a1[36]);
    v17 = a1[36];
    if ( (_DWORD)result )
      goto LABEL_68;
    v18 = *(_BYTE *)(v17 + 140);
    if ( v18 == 12 )
    {
      result = a1[36];
      do
      {
        result = *(_QWORD *)(result + 160);
        v18 = *(_BYTE *)(result + 140);
      }
      while ( v18 == 12 );
    }
    if ( !v18 )
    {
LABEL_68:
      if ( (*((_BYTE *)a1 + 179) & 0x20) != 0 )
      {
        v30 = a1[36];
        v23 = sub_8D23E0(v30);
        v17 = v30;
        if ( v23 )
        {
          v24 = a1[17];
          if ( v24 )
            v17 = *(_QWORD *)(v24 + 128);
        }
      }
      sub_6301D0(v15, *a1, a4, a2, v17);
      result = *(_QWORD *)(v15 + 120);
      a1[36] = result;
    }
  }
  return result;
}
