// Function: sub_BFC120
// Address: 0xbfc120
//
void __fastcall sub_BFC120(__int64 a1, __int64 a2)
{
  __int64 v4; // rdi
  int v5; // r8d
  __int64 v6; // rsi
  __int64 v7; // rcx
  int v8; // r10d
  __int64 v9; // rdx
  const char *v10; // rax
  __int64 v11; // r14
  _BYTE *v12; // rax
  __int64 v13; // rax
  _QWORD v14[4]; // [rsp+0h] [rbp-50h] BYREF
  char v15; // [rsp+20h] [rbp-30h]
  char v16; // [rsp+21h] [rbp-2Fh]

  v4 = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 8LL);
  v5 = *(unsigned __int8 *)(v4 + 8);
  if ( (unsigned int)(v5 - 17) <= 1 )
  {
    v6 = **(_QWORD **)(v4 + 16);
    if ( *(_BYTE *)(v6 + 8) == 14 )
      goto LABEL_3;
LABEL_13:
    v16 = 1;
    v10 = "AddrSpaceCast source must be a pointer";
    goto LABEL_14;
  }
  v6 = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 8LL);
  if ( (_BYTE)v5 != 14 )
    goto LABEL_13;
LABEL_3:
  v7 = *(_QWORD *)(a2 + 8);
  v8 = *(unsigned __int8 *)(v7 + 8);
  if ( (unsigned int)(v8 - 17) > 1 )
  {
    v9 = *(_QWORD *)(a2 + 8);
    if ( (_BYTE)v8 == 14 )
      goto LABEL_5;
LABEL_11:
    v16 = 1;
    v10 = "AddrSpaceCast result must be a pointer";
    goto LABEL_14;
  }
  v9 = **(_QWORD **)(v7 + 16);
  if ( *(_BYTE *)(v9 + 8) != 14 )
    goto LABEL_11;
LABEL_5:
  if ( *(_DWORD *)(v9 + 8) >> 8 == *(_DWORD *)(v6 + 8) >> 8 )
  {
    v16 = 1;
    v14[0] = "AddrSpaceCast must be between different address spaces";
    v15 = 3;
    sub_BDBF70((__int64 *)a1, (__int64)v14);
    if ( !*(_QWORD *)a1 )
      return;
    goto LABEL_18;
  }
  if ( (unsigned int)(v5 - 17) > 1
    || *(_DWORD *)(v4 + 32) == *(_DWORD *)(v7 + 32) && ((_BYTE)v5 == 18) == ((_BYTE)v8 == 18) )
  {
    sub_BF6FE0(a1, a2);
    return;
  }
  v16 = 1;
  v10 = "AddrSpaceCast vector pointer number of elements mismatch";
LABEL_14:
  v11 = *(_QWORD *)a1;
  v14[0] = v10;
  v15 = 3;
  if ( !v11 )
  {
    *(_BYTE *)(a1 + 152) = 1;
    return;
  }
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
  if ( v13 )
LABEL_18:
    sub_BDBD80(a1, (_BYTE *)a2);
}
