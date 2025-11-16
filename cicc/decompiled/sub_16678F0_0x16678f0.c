// Function: sub_16678F0
// Address: 0x16678f0
//
void __fastcall sub_16678F0(__int64 a1, __int64 a2)
{
  __int64 v4; // r9
  char v5; // al
  __int64 v6; // rdx
  const char *v7; // rax
  __int64 v8; // r14
  _BYTE *v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  _DWORD *v12; // rdi
  _DWORD *v13; // rsi
  char v14; // r8
  __int64 v15; // r9
  char v16; // r10
  __int64 v17; // r11
  const char *v18; // rax
  int v19[4]; // [rsp+0h] [rbp-40h] BYREF
  char v20; // [rsp+10h] [rbp-30h]
  char v21; // [rsp+11h] [rbp-2Fh]

  v4 = **(_QWORD **)(a2 - 24);
  v5 = *(_BYTE *)(v4 + 8);
  if ( v5 == 16 )
    v5 = *(_BYTE *)(**(_QWORD **)(v4 + 16) + 8LL);
  if ( v5 == 11 )
  {
    v6 = *(_QWORD *)a2;
    if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 16 )
    {
      v6 = **(_QWORD **)(*(_QWORD *)a2 + 16LL);
      if ( *(_BYTE *)(v6 + 8) != 15 )
        goto LABEL_6;
    }
    else if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) != 15 )
    {
LABEL_6:
      v21 = 1;
      v7 = "IntToPtr result must be a pointer";
      goto LABEL_7;
    }
    v11 = *(_QWORD *)(a1 + 56);
    v12 = *(_DWORD **)(v11 + 408);
    v13 = &v12[*(unsigned int *)(v11 + 416)];
    v19[0] = *(_DWORD *)(v6 + 8) >> 8;
    if ( v13 != sub_164E170(v12, (__int64)v13, v19) )
    {
      v21 = 1;
      *(_QWORD *)v19 = "inttoptr not supported for non-integral pointers";
      v20 = 3;
      sub_164FF40((__int64 *)a1, (__int64)v19);
      return;
    }
    if ( (v16 == 16) == (v14 == 16) )
    {
      if ( v14 != 16 )
        goto LABEL_18;
      if ( v16 != 16 )
        BUG();
      if ( *(_QWORD *)(v17 + 32) == *(_QWORD *)(v15 + 32) )
      {
LABEL_18:
        sub_1663F80(a1, a2);
        return;
      }
      v21 = 1;
      v18 = "IntToPtr Vector width mismatch";
    }
    else
    {
      v21 = 1;
      v18 = "IntToPtr type mismatch";
    }
    *(_QWORD *)v19 = v18;
    v20 = 3;
    sub_164FF40((__int64 *)a1, (__int64)v19);
    if ( *(_QWORD *)a1 )
      goto LABEL_11;
    return;
  }
  v21 = 1;
  v7 = "IntToPtr source must be an integral";
LABEL_7:
  v8 = *(_QWORD *)a1;
  *(_QWORD *)v19 = v7;
  v20 = 3;
  if ( !v8 )
  {
    *(_BYTE *)(a1 + 72) = 1;
    return;
  }
  sub_16E2CE0(v19, v8);
  v9 = *(_BYTE **)(v8 + 24);
  if ( (unsigned __int64)v9 >= *(_QWORD *)(v8 + 16) )
  {
    sub_16E7DE0(v8, 10);
  }
  else
  {
    *(_QWORD *)(v8 + 24) = v9 + 1;
    *v9 = 10;
  }
  v10 = *(_QWORD *)a1;
  *(_BYTE *)(a1 + 72) = 1;
  if ( v10 )
LABEL_11:
    sub_164FA80((__int64 *)a1, a2);
}
