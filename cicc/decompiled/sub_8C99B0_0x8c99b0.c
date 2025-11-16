// Function: sub_8C99B0
// Address: 0x8c99b0
//
__int64 __fastcall sub_8C99B0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  _QWORD *v3; // r13
  __int64 v4; // r15
  _QWORD *i; // r12
  __int64 v6; // r14
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 *v9; // r15
  _QWORD *v10; // r12
  __int64 v11; // r14
  __int64 v12; // r14
  __int64 v13; // r13
  __int64 v14; // rdx
  __int64 v15; // r12
  _QWORD *j; // r12
  __int64 v17; // r14
  __int64 v18; // r15
  __int64 v19; // r14
  __int64 v20; // [rsp-40h] [rbp-40h]
  __int64 v21; // [rsp-40h] [rbp-40h]
  __int64 v22; // [rsp-40h] [rbp-40h]
  __int64 v23; // [rsp-40h] [rbp-40h]
  __int64 *v24; // [rsp-40h] [rbp-40h]

  result = *(unsigned __int8 *)(a1 + 80);
  if ( (unsigned __int8)result <= 0xAu )
    return result;
  if ( (unsigned __int8)(result - 19) > 3u )
  {
    v3 = 0;
    goto LABEL_17;
  }
  v3 = *(_QWORD **)(a1 + 88);
  if ( (((_BYTE)result - 19) & 0xFD) != 0 )
  {
LABEL_17:
    if ( (result & 0xFD) != 0x14 || (*(_BYTE *)(a1 + 81) & 0x40) != 0 )
      return result;
    goto LABEL_19;
  }
  if ( (*(_BYTE *)(a1 + 81) & 0x40) != 0 )
    return result;
  if ( (_BYTE)result == 19 )
  {
    v4 = v3[22];
    if ( v4 )
    {
      result = sub_8CA0A0(*(_QWORD *)(v4 + 88), a2);
      if ( (_DWORD)a2 )
        result = sub_8C9930((__int64)v3, v4);
    }
    for ( i = (_QWORD *)v3[21]; i; i = (_QWORD *)*i )
    {
      v6 = i[1];
      if ( v4 != v6 )
      {
        result = sub_8CA0A0(*(_QWORD *)(v6 + 88), (unsigned int)a2);
        if ( (_DWORD)a2 )
        {
          result = (__int64)sub_8C6880((__int64)v3, v6, v7, v8);
          if ( !result )
            result = sub_8C9930((__int64)v3, v6);
        }
      }
    }
    return result;
  }
LABEL_19:
  if ( (_BYTE)result != 21 )
  {
    v9 = (__int64 *)v3[22];
    v10 = (_QWORD *)v3[21];
    if ( !v9 )
    {
      if ( v10 )
        goto LABEL_26;
LABEL_31:
      if ( (_DWORD)a2 )
        return (__int64)sub_8C7090(11, (__int64)v10);
      goto LABEL_37;
    }
    if ( (_DWORD)a2 )
    {
      sub_8C7090(11, v3[22]);
      sub_8C9930((__int64)v3, *v9);
      if ( v10 )
        goto LABEL_26;
      v10 = (_QWORD *)v3[22];
      return (__int64)sub_8C7090(11, (__int64)v10);
    }
    v12 = v9[4];
    if ( v12 )
    {
      sub_8C6400(11, v3[22]);
      result = sub_8D0810(v12);
      v9[4] = 0;
    }
    if ( !v10 )
    {
      v10 = (_QWORD *)v3[22];
LABEL_37:
      v13 = v10[4];
      if ( v13 )
      {
        sub_8C6400(11, (__int64)v10);
        result = sub_8D0810(v13);
        v10[4] = 0;
      }
      return result;
    }
    while ( 1 )
    {
LABEL_26:
      result = v10[3];
      v11 = *(_QWORD *)(result + 88);
      if ( v9 == (__int64 *)v11 )
        goto LABEL_25;
      if ( (_DWORD)a2 )
      {
        sub_8C7090(11, *(_QWORD *)(result + 88));
        result = sub_8C9930((__int64)v3, v10[3]);
LABEL_25:
        v10 = (_QWORD *)*v10;
        if ( !v10 )
          goto LABEL_30;
      }
      else
      {
        v20 = *(_QWORD *)(v11 + 32);
        if ( !v20 )
          goto LABEL_25;
        sub_8C6400(11, v11);
        result = sub_8D0810(v20);
        *(_QWORD *)(v11 + 32) = 0;
        v10 = (_QWORD *)*v10;
        if ( !v10 )
        {
LABEL_30:
          v10 = (_QWORD *)v3[22];
          goto LABEL_31;
        }
      }
    }
  }
  v14 = v3[24];
  if ( v14 )
  {
    if ( (_DWORD)a2 )
    {
      v24 = (__int64 *)v3[24];
      sub_8C7090(7, (__int64)v24);
      result = sub_8C9930((__int64)v3, *v24);
      v14 = (__int64)v24;
    }
    else
    {
      v15 = *(_QWORD *)(v14 + 32);
      if ( v15 )
      {
        v21 = v3[24];
        sub_8C6400(7, v21);
        result = sub_8D0810(v15);
        v14 = v21;
        *(_QWORD *)(v21 + 32) = 0;
      }
    }
  }
  for ( j = (_QWORD *)v3[23]; j; j = (_QWORD *)*j )
  {
    v17 = j[1];
    if ( *(_BYTE *)(v17 + 80) == 7 )
    {
      v18 = *(_QWORD *)(v17 + 88);
      if ( v14 != v18 )
      {
        if ( (_DWORD)a2 )
        {
          v22 = v14;
          sub_8C7090(7, *(_QWORD *)(v17 + 88));
          result = sub_8C9930((__int64)v3, v17);
          v14 = v22;
        }
        else
        {
          v19 = *(_QWORD *)(v18 + 32);
          if ( v19 )
          {
            v23 = v14;
            sub_8C6400(7, v18);
            result = sub_8D0810(v19);
            *(_QWORD *)(v18 + 32) = 0;
            v14 = v23;
          }
        }
      }
    }
  }
  return result;
}
