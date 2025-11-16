// Function: sub_16676C0
// Address: 0x16676c0
//
void __fastcall sub_16676C0(__int64 a1, __int64 a2)
{
  __int64 v4; // r9
  __int64 v5; // rdx
  char v6; // al
  __int64 v7; // r14
  _BYTE *v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  _DWORD *v11; // rdi
  _DWORD *v12; // rsi
  char v13; // r8
  __int64 v14; // r9
  __int64 v15; // r10
  __int64 v16; // r11
  char v17; // al
  char v18; // dl
  const char *v19; // rax
  int v20[4]; // [rsp+0h] [rbp-40h] BYREF
  char v21; // [rsp+10h] [rbp-30h]
  char v22; // [rsp+11h] [rbp-2Fh]

  v4 = **(_QWORD **)(a2 - 24);
  v5 = v4;
  v6 = *(_BYTE *)(v4 + 8);
  if ( v6 == 16 )
  {
    v5 = **(_QWORD **)(v4 + 16);
    v6 = *(_BYTE *)(v5 + 8);
  }
  if ( v6 != 15 )
  {
    v7 = *(_QWORD *)a1;
    v22 = 1;
    *(_QWORD *)v20 = "PtrToInt source must be pointer";
    v21 = 3;
    if ( v7 )
    {
      sub_16E2CE0(v20, v7);
      v8 = *(_BYTE **)(v7 + 24);
      if ( (unsigned __int64)v8 < *(_QWORD *)(v7 + 16) )
      {
LABEL_6:
        *(_QWORD *)(v7 + 24) = v8 + 1;
        *v8 = 10;
        v9 = *(_QWORD *)a1;
        goto LABEL_7;
      }
LABEL_22:
      sub_16E7DE0(v7, 10);
      v9 = *(_QWORD *)a1;
LABEL_7:
      *(_BYTE *)(a1 + 72) = 1;
      if ( v9 )
      {
LABEL_8:
        sub_164FA80((__int64 *)a1, a2);
        return;
      }
      return;
    }
    goto LABEL_9;
  }
  v10 = *(_QWORD *)(a1 + 56);
  v11 = *(_DWORD **)(v10 + 408);
  v12 = &v11[*(unsigned int *)(v10 + 416)];
  v20[0] = *(_DWORD *)(v5 + 8) >> 8;
  if ( v12 != sub_164E170(v11, (__int64)v12, v20) )
  {
    v22 = 1;
    *(_QWORD *)v20 = "ptrtoint not supported for non-integral pointers";
    v21 = 3;
    sub_164FF40((__int64 *)a1, v16);
    return;
  }
  v17 = *(_BYTE *)(v15 + 8);
  v18 = v17;
  if ( v17 == 16 )
    v18 = *(_BYTE *)(**(_QWORD **)(v15 + 16) + 8LL);
  if ( v18 != 11 )
  {
    v7 = *(_QWORD *)a1;
    v22 = 1;
    *(_QWORD *)v20 = "PtrToInt result must be integral";
    v21 = 3;
    if ( v7 )
    {
      sub_16E2CE0(v16, v7);
      v8 = *(_BYTE **)(v7 + 24);
      if ( (unsigned __int64)v8 < *(_QWORD *)(v7 + 16) )
        goto LABEL_6;
      goto LABEL_22;
    }
LABEL_9:
    *(_BYTE *)(a1 + 72) = 1;
    return;
  }
  if ( (v17 == 16) == (v13 == 16) )
  {
    if ( v13 != 16 )
      goto LABEL_19;
    if ( v17 != 16 )
      BUG();
    if ( *(_QWORD *)(v15 + 32) == *(_QWORD *)(v14 + 32) )
    {
LABEL_19:
      sub_1663F80(a1, a2);
      return;
    }
    v22 = 1;
    v19 = "PtrToInt Vector width mismatch";
  }
  else
  {
    v22 = 1;
    v19 = "PtrToInt type mismatch";
  }
  *(_QWORD *)v20 = v19;
  v21 = 3;
  sub_164FF40((__int64 *)a1, v16);
  if ( *(_QWORD *)a1 )
    goto LABEL_8;
}
