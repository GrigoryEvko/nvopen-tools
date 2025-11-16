// Function: sub_164FFA0
// Address: 0x164ffa0
//
void __fastcall sub_164FFA0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r13
  unsigned __int8 v4; // al
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // r13
  _BYTE *v8; // rax
  __int64 v9; // rax
  __int64 v10; // r15
  _BYTE *v11; // rax
  __int64 v12; // rax
  const char *v13; // rax
  _QWORD v14[2]; // [rsp+0h] [rbp-40h] BYREF
  char v15; // [rsp+10h] [rbp-30h]
  char v16; // [rsp+11h] [rbp-2Fh]

  v3 = *(_QWORD *)(a2 + 136);
  if ( v3 )
  {
    if ( *(_BYTE *)(*(_QWORD *)v3 + 8LL) == 8 )
    {
      v10 = *(_QWORD *)a1;
      v16 = 1;
      v14[0] = "Unexpected metadata round-trip through values";
      v15 = 3;
      if ( !v10 )
        goto LABEL_18;
      sub_16E2CE0(v14, v10);
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
      *(_BYTE *)(a1 + 72) = 1;
      if ( !v12 )
        return;
      goto LABEL_23;
    }
    if ( *(_BYTE *)a2 != 2 )
      return;
    if ( a3 )
    {
      v4 = *(_BYTE *)(v3 + 16);
      if ( v4 <= 0x17u )
      {
        if ( v4 == 18 )
        {
          v6 = *(_QWORD *)(v3 + 56);
        }
        else
        {
          if ( v4 != 17 )
            goto LABEL_11;
          v6 = *(_QWORD *)(v3 + 24);
        }
LABEL_8:
        if ( v6 == a3 )
          return;
LABEL_11:
        v7 = *(_QWORD *)a1;
        v16 = 1;
        v14[0] = "function-local metadata used in wrong function";
        v15 = 3;
        if ( v7 )
        {
          sub_16E2CE0(v14, v7);
          v8 = *(_BYTE **)(v7 + 24);
          if ( (unsigned __int64)v8 >= *(_QWORD *)(v7 + 16) )
          {
            sub_16E7DE0(v7, 10);
          }
          else
          {
            *(_QWORD *)(v7 + 24) = v8 + 1;
            *v8 = 10;
          }
          v9 = *(_QWORD *)a1;
          *(_BYTE *)(a1 + 72) = 1;
          if ( v9 )
            goto LABEL_15;
          return;
        }
LABEL_18:
        *(_BYTE *)(a1 + 72) = 1;
        return;
      }
      v5 = *(_QWORD *)(v3 + 40);
      if ( v5 )
      {
        v6 = *(_QWORD *)(v5 + 56);
        goto LABEL_8;
      }
      v16 = 1;
      v14[0] = "function-local metadata not in basic block";
      v15 = 3;
      sub_164FF40((__int64 *)a1, (__int64)v14);
      if ( !*(_QWORD *)a1 )
        return;
LABEL_23:
      sub_164ED40((__int64 *)a1, (unsigned __int8 *)a2);
      sub_164FA80((__int64 *)a1, v3);
      return;
    }
    v16 = 1;
    v13 = "function-local metadata used outside a function";
  }
  else
  {
    v16 = 1;
    v13 = "Expected valid value";
  }
  v14[0] = v13;
  v15 = 3;
  sub_164FF40((__int64 *)a1, (__int64)v14);
  if ( *(_QWORD *)a1 )
LABEL_15:
    sub_164ED40((__int64 *)a1, (unsigned __int8 *)a2);
}
