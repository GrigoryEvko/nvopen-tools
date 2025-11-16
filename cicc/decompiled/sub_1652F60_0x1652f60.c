// Function: sub_1652F60
// Address: 0x1652f60
//
__int64 __fastcall sub_1652F60(__int64 a1, __int64 a2)
{
  __int64 v3; // rsi
  __int64 v4; // rdx
  __int64 result; // rax
  __int64 v6; // rbx
  int v7; // ebx
  int v8; // r15d
  unsigned __int8 *v9; // r14
  __int64 v10; // r14
  _BYTE *v11; // rax
  __int64 v12; // rdx
  char v13; // al
  __int64 v14; // r12
  _BYTE *v15; // rax
  __int64 v16; // r15
  _BYTE *v17; // rax
  __int64 v18; // rdx
  char v19; // al
  unsigned __int64 v20; // [rsp+0h] [rbp-80h] BYREF
  char v21; // [rsp+8h] [rbp-78h]
  __int64 v22; // [rsp+10h] [rbp-70h] BYREF
  __int64 v23; // [rsp+18h] [rbp-68h]
  char v24; // [rsp+20h] [rbp-60h]
  _QWORD v25[2]; // [rsp+30h] [rbp-50h] BYREF
  char v26; // [rsp+40h] [rbp-40h]
  char v27; // [rsp+41h] [rbp-3Fh]

  v3 = *(_QWORD *)(a2 - 8LL * *(unsigned int *)(a2 + 8));
  if ( !v3 )
  {
    v14 = *(_QWORD *)a1;
    v27 = 1;
    v25[0] = "missing variable";
    v26 = 3;
    if ( v14 )
    {
      sub_16E2CE0(v25, v14);
      v15 = *(_BYTE **)(v14 + 24);
      if ( (unsigned __int64)v15 >= *(_QWORD *)(v14 + 16) )
      {
        sub_16E7DE0(v14, 10);
      }
      else
      {
        *(_QWORD *)(v14 + 24) = v15 + 1;
        *v15 = 10;
      }
    }
    goto LABEL_17;
  }
  sub_1652C50(a1, v3);
  v4 = *(unsigned int *)(a2 + 8);
  result = 1 - v4;
  v6 = *(_QWORD *)(a2 + 8 * (1 - v4));
  if ( !v6 )
    return result;
  if ( !(unsigned __int8)sub_15B1200(*(_QWORD *)(a2 + 8 * (1 - v4))) )
  {
    v10 = *(_QWORD *)a1;
    v27 = 1;
    v25[0] = "invalid expression";
    v26 = 3;
    if ( v10 )
    {
      sub_16E2CE0(v25, v10);
      v11 = *(_BYTE **)(v10 + 24);
      if ( (unsigned __int64)v11 >= *(_QWORD *)(v10 + 16) )
      {
        sub_16E7DE0(v10, 10);
      }
      else
      {
        *(_QWORD *)(v10 + 24) = v11 + 1;
        *v11 = 10;
      }
      v12 = *(_QWORD *)a1;
      v13 = *(_BYTE *)(a1 + 74);
      *(_BYTE *)(a1 + 73) = 1;
      *(_BYTE *)(a1 + 72) |= v13;
      if ( v12 )
        sub_164ED40((__int64 *)a1, (unsigned __int8 *)v6);
    }
    else
    {
      v19 = *(_BYTE *)(a1 + 74);
      *(_BYTE *)(a1 + 73) = 1;
      *(_BYTE *)(a1 + 72) |= v19;
    }
  }
  result = sub_15B1350((__int64)&v22, *(unsigned __int64 **)(v6 + 24), *(unsigned __int64 **)(v6 + 32));
  if ( !v24 )
    return result;
  v7 = v22;
  v8 = v23;
  v9 = *(unsigned __int8 **)(a2 - 8LL * *(unsigned int *)(a2 + 8));
  result = sub_15B1130((__int64)&v20, (__int64)v9);
  if ( !v21 )
    return result;
  result = v20;
  if ( (unsigned int)(v7 + v8) > v20 )
  {
    v16 = *(_QWORD *)a1;
    v27 = 1;
    v25[0] = "fragment is larger than or outside of variable";
    v26 = 3;
    if ( v16 )
    {
      sub_16E2CE0(v25, v16);
      v17 = *(_BYTE **)(v16 + 24);
      if ( (unsigned __int64)v17 >= *(_QWORD *)(v16 + 16) )
      {
        sub_16E7DE0(v16, 10);
      }
      else
      {
        *(_QWORD *)(v16 + 24) = v17 + 1;
        *v17 = 10;
      }
      v18 = *(_QWORD *)a1;
      result = *(unsigned __int8 *)(a1 + 74);
      *(_BYTE *)(a1 + 73) = 1;
      *(_BYTE *)(a1 + 72) |= result;
      if ( v18 )
        goto LABEL_20;
      return result;
    }
LABEL_17:
    result = *(unsigned __int8 *)(a1 + 74);
    *(_BYTE *)(a1 + 73) = 1;
    *(_BYTE *)(a1 + 72) |= result;
    return result;
  }
  if ( v20 == v7 )
  {
    v27 = 1;
    v25[0] = "fragment covers entire variable";
    v26 = 3;
    result = sub_16521E0((__int64 *)a1, (__int64)v25);
    if ( *(_QWORD *)a1 )
    {
LABEL_20:
      result = (__int64)sub_164ED40((__int64 *)a1, (unsigned __int8 *)a2);
      if ( v9 )
        return (__int64)sub_164ED40((__int64 *)a1, v9);
    }
  }
  return result;
}
