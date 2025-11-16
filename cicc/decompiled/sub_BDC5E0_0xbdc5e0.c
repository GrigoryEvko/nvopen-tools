// Function: sub_BDC5E0
// Address: 0xbdc5e0
//
void __fastcall sub_BDC5E0(__int64 a1, __int64 a2, __int64 a3)
{
  char *v3; // r13
  char v4; // al
  __int64 v5; // rax
  __int64 v6; // rax
  const char *v7; // rax
  __int64 v8; // r13
  _BYTE *v9; // rax
  __int64 v10; // rax
  __int64 v11; // r15
  _BYTE *v12; // rax
  __int64 v13; // rax
  _QWORD v14[4]; // [rsp+0h] [rbp-50h] BYREF
  char v15; // [rsp+20h] [rbp-30h]
  char v16; // [rsp+21h] [rbp-2Fh]

  v3 = *(char **)(a2 + 136);
  if ( !v3 )
  {
    v16 = 1;
    v7 = "Expected valid value";
LABEL_12:
    v8 = *(_QWORD *)a1;
    v14[0] = v7;
    v15 = 3;
    if ( v8 )
    {
      sub_CA0E80(v14, v8);
      v9 = *(_BYTE **)(v8 + 32);
      if ( (unsigned __int64)v9 >= *(_QWORD *)(v8 + 24) )
      {
        sub_CB5D20(v8, 10);
      }
      else
      {
        *(_QWORD *)(v8 + 32) = v9 + 1;
        *v9 = 10;
      }
      v10 = *(_QWORD *)a1;
      *(_BYTE *)(a1 + 152) = 1;
      if ( v10 )
        goto LABEL_16;
      return;
    }
LABEL_19:
    *(_BYTE *)(a1 + 152) = 1;
    return;
  }
  if ( *(_BYTE *)(*((_QWORD *)v3 + 1) + 8LL) == 9 )
  {
    v11 = *(_QWORD *)a1;
    v16 = 1;
    v14[0] = "Unexpected metadata round-trip through values";
    v15 = 3;
    if ( !v11 )
      goto LABEL_19;
    sub_CA0E80(v14, v11);
    v12 = *(_BYTE **)(v11 + 32);
    if ( (unsigned __int64)v12 >= *(_QWORD *)(v11 + 24) )
    {
      sub_CB5D20(v11, 10);
    }
    else
    {
      *(_QWORD *)(v11 + 32) = v12 + 1;
      *v12 = 10;
    }
    v13 = *(_QWORD *)a1;
    *(_BYTE *)(a1 + 152) = 1;
    if ( !v13 )
      return;
    goto LABEL_24;
  }
  if ( *(_BYTE *)a2 != 2 )
    return;
  if ( a3 )
  {
    v4 = *v3;
    if ( (unsigned __int8)*v3 <= 0x1Cu )
    {
      if ( v4 == 23 )
      {
        v6 = *((_QWORD *)v3 + 9);
      }
      else
      {
        if ( v4 != 22 )
        {
LABEL_11:
          v16 = 1;
          v7 = "function-local metadata used in wrong function";
          goto LABEL_12;
        }
        v6 = *((_QWORD *)v3 + 3);
      }
LABEL_8:
      if ( v6 == a3 )
        return;
      goto LABEL_11;
    }
    v5 = *((_QWORD *)v3 + 5);
    if ( v5 )
    {
      v6 = *(_QWORD *)(v5 + 72);
      goto LABEL_8;
    }
    v16 = 1;
    v14[0] = "function-local metadata not in basic block";
    v15 = 3;
    sub_BDBF70((__int64 *)a1, (__int64)v14);
    if ( !*(_QWORD *)a1 )
      return;
LABEL_24:
    sub_BD9900((__int64 *)a1, (const char *)a2);
    sub_BDBD80(a1, v3);
    return;
  }
  v16 = 1;
  v14[0] = "function-local metadata used outside a function";
  v15 = 3;
  sub_BDBF70((__int64 *)a1, (__int64)v14);
  if ( *(_QWORD *)a1 )
LABEL_16:
    sub_BD9900((__int64 *)a1, (const char *)a2);
}
