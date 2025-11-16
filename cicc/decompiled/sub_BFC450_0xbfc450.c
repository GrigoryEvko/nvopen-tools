// Function: sub_BFC450
// Address: 0xbfc450
//
void __fastcall sub_BFC450(__int64 a1, __int64 a2)
{
  __int64 v3; // r14
  __int64 v4; // rax
  __int64 v5; // rsi
  __int64 v6; // rcx
  __int64 v7; // rax
  __int64 v8; // rcx
  __int64 v9; // r15
  _BYTE *v10; // rax
  __int64 v11; // rax
  __int64 v12; // r14
  _BYTE *v13; // rax
  __int64 v14; // rax
  __int64 v15; // r12
  _BYTE *v16; // rax
  _QWORD v17[4]; // [rsp+0h] [rbp-50h] BYREF
  char v18; // [rsp+20h] [rbp-30h]
  char v19; // [rsp+21h] [rbp-2Fh]

  v3 = *(_QWORD *)(a2 + 40);
  v4 = *(_QWORD *)(v3 + 56);
  if ( !v4 || a2 != v4 - 24 )
  {
    if ( (*(_QWORD *)(a2 + 24) & 0xFFFFFFFFFFFFFFF8LL) == 0 )
      BUG();
    if ( *(_BYTE *)((*(_QWORD *)(a2 + 24) & 0xFFFFFFFFFFFFFFF8LL) - 24) != 84 )
    {
      v9 = *(_QWORD *)a1;
      v19 = 1;
      v17[0] = "PHI nodes not grouped at top of basic block!";
      v18 = 3;
      if ( v9 )
      {
        sub_CA0E80(v17, v9);
        v10 = *(_BYTE **)(v9 + 32);
        if ( (unsigned __int64)v10 >= *(_QWORD *)(v9 + 24) )
        {
          sub_CB5D20(v9, 10);
        }
        else
        {
          *(_QWORD *)(v9 + 32) = v10 + 1;
          *v10 = 10;
        }
        v11 = *(_QWORD *)a1;
        *(_BYTE *)(a1 + 152) = 1;
        if ( v11 )
        {
          sub_BDBD80(a1, (_BYTE *)a2);
          sub_BDBD80(a1, (_BYTE *)v3);
        }
        return;
      }
LABEL_19:
      *(_BYTE *)(a1 + 152) = 1;
      return;
    }
  }
  v5 = *(_QWORD *)(a2 + 8);
  if ( *(_BYTE *)(v5 + 8) == 11 )
  {
    v15 = *(_QWORD *)a1;
    v19 = 1;
    v17[0] = "PHI nodes cannot have token type!";
    v18 = 3;
    if ( v15 )
    {
      sub_CA0E80(v17, v15);
      v16 = *(_BYTE **)(v15 + 32);
      if ( (unsigned __int64)v16 >= *(_QWORD *)(v15 + 24) )
      {
        sub_CB5D20(v15, 10);
      }
      else
      {
        *(_QWORD *)(v15 + 32) = v16 + 1;
        *v16 = 10;
      }
    }
    goto LABEL_19;
  }
  v6 = 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
  {
    v7 = *(_QWORD *)(a2 - 8);
    v8 = v7 + v6;
  }
  else
  {
    v7 = a2 - v6;
    v8 = a2;
  }
  while ( 1 )
  {
    if ( v8 == v7 )
    {
      sub_BF6FE0(a1, a2);
      return;
    }
    if ( v5 != *(_QWORD *)(*(_QWORD *)v7 + 8LL) )
      break;
    v7 += 32;
  }
  v12 = *(_QWORD *)a1;
  v19 = 1;
  v17[0] = "PHI node operands are not the same type as the result!";
  v18 = 3;
  if ( !v12 )
    goto LABEL_19;
  sub_CA0E80(v17, v12);
  v13 = *(_BYTE **)(v12 + 32);
  if ( (unsigned __int64)v13 >= *(_QWORD *)(v12 + 24) )
  {
    sub_CB5D20(v12, 10);
  }
  else
  {
    *(_QWORD *)(v12 + 32) = v13 + 1;
    *v13 = 10;
  }
  v14 = *(_QWORD *)a1;
  *(_BYTE *)(a1 + 152) = 1;
  if ( v14 )
    sub_BDBD80(a1, (_BYTE *)a2);
}
