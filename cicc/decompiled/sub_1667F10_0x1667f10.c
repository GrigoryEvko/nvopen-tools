// Function: sub_1667F10
// Address: 0x1667f10
//
void __fastcall sub_1667F10(__int64 a1, __int64 a2)
{
  __int64 v3; // r14
  __int64 v4; // rax
  __int64 v5; // rsi
  __int64 v6; // rcx
  _QWORD **v7; // rax
  _QWORD **v8; // rcx
  __int64 v9; // r15
  _BYTE *v10; // rax
  __int64 v11; // rax
  __int64 v12; // r14
  _BYTE *v13; // rax
  __int64 v14; // rax
  __int64 v15; // r12
  _BYTE *v16; // rax
  _QWORD v17[2]; // [rsp+0h] [rbp-40h] BYREF
  char v18; // [rsp+10h] [rbp-30h]
  char v19; // [rsp+11h] [rbp-2Fh]

  v3 = *(_QWORD *)(a2 + 40);
  v4 = *(_QWORD *)(v3 + 48);
  if ( !v4 || a2 != v4 - 24 )
  {
    if ( (*(_QWORD *)(a2 + 24) & 0xFFFFFFFFFFFFFFF8LL) == 0 )
      BUG();
    if ( *(_BYTE *)((*(_QWORD *)(a2 + 24) & 0xFFFFFFFFFFFFFFF8LL) - 8) != 77 )
    {
      v9 = *(_QWORD *)a1;
      v19 = 1;
      v17[0] = "PHI nodes not grouped at top of basic block!";
      v18 = 3;
      if ( v9 )
      {
        sub_16E2CE0(v17, v9);
        v10 = *(_BYTE **)(v9 + 24);
        if ( (unsigned __int64)v10 >= *(_QWORD *)(v9 + 16) )
        {
          sub_16E7DE0(v9, 10);
        }
        else
        {
          *(_QWORD *)(v9 + 24) = v10 + 1;
          *v10 = 10;
        }
        v11 = *(_QWORD *)a1;
        *(_BYTE *)(a1 + 72) = 1;
        if ( v11 )
        {
          sub_164FA80((__int64 *)a1, a2);
          sub_164FA80((__int64 *)a1, v3);
        }
        return;
      }
LABEL_19:
      *(_BYTE *)(a1 + 72) = 1;
      return;
    }
  }
  v5 = *(_QWORD *)a2;
  if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 10 )
  {
    v15 = *(_QWORD *)a1;
    v19 = 1;
    v17[0] = "PHI nodes cannot have token type!";
    v18 = 3;
    if ( v15 )
    {
      sub_16E2CE0(v17, v15);
      v16 = *(_BYTE **)(v15 + 24);
      if ( (unsigned __int64)v16 >= *(_QWORD *)(v15 + 16) )
      {
        sub_16E7DE0(v15, 10);
      }
      else
      {
        *(_QWORD *)(v15 + 24) = v16 + 1;
        *v16 = 10;
      }
    }
    goto LABEL_19;
  }
  v6 = 3LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
  {
    v7 = *(_QWORD ***)(a2 - 8);
    v8 = &v7[v6];
  }
  else
  {
    v7 = (_QWORD **)(a2 - v6 * 8);
    v8 = (_QWORD **)a2;
  }
  while ( 1 )
  {
    if ( v8 == v7 )
    {
      sub_1663F80(a1, a2);
      return;
    }
    if ( v5 != **v7 )
      break;
    v7 += 3;
  }
  v12 = *(_QWORD *)a1;
  v19 = 1;
  v17[0] = "PHI node operands are not the same type as the result!";
  v18 = 3;
  if ( !v12 )
    goto LABEL_19;
  sub_16E2CE0(v17, v12);
  v13 = *(_BYTE **)(v12 + 24);
  if ( (unsigned __int64)v13 >= *(_QWORD *)(v12 + 16) )
  {
    sub_16E7DE0(v12, 10);
  }
  else
  {
    *(_QWORD *)(v12 + 24) = v13 + 1;
    *v13 = 10;
  }
  v14 = *(_QWORD *)a1;
  *(_BYTE *)(a1 + 72) = 1;
  if ( v14 )
    sub_164FA80((__int64 *)a1, a2);
}
