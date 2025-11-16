// Function: sub_BFB020
// Address: 0xbfb020
//
void __fastcall sub_BFB020(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  int v4; // edx
  unsigned int v5; // eax
  __int64 v6; // r14
  int v7; // ecx
  char v8; // dl
  char v9; // al
  unsigned int v10; // ebx
  const char *v11; // rax
  __int64 v12; // r14
  _BYTE *v13; // rax
  __int64 v14; // rax
  const char *v15; // [rsp+0h] [rbp-50h] BYREF
  char v16; // [rsp+20h] [rbp-30h]
  char v17; // [rsp+21h] [rbp-2Fh]

  v3 = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 8LL);
  v4 = *(unsigned __int8 *)(v3 + 8);
  v5 = v4 - 17;
  if ( (unsigned int)(v4 - 17) <= 1 )
    LOBYTE(v4) = *(_BYTE *)(**(_QWORD **)(v3 + 16) + 8LL);
  if ( (_BYTE)v4 != 12 )
  {
    v17 = 1;
    v11 = "ZExt only operates on integer";
    goto LABEL_13;
  }
  v6 = *(_QWORD *)(a2 + 8);
  v7 = *(unsigned __int8 *)(v6 + 8);
  v8 = *(_BYTE *)(v6 + 8);
  if ( (unsigned int)(v7 - 17) <= 1 )
    v8 = *(_BYTE *)(**(_QWORD **)(v6 + 16) + 8LL);
  if ( v8 != 12 )
  {
    v17 = 1;
    v11 = "ZExt only produces an integer";
    goto LABEL_13;
  }
  if ( v5 > 1 )
  {
    v9 = 0;
    if ( v7 == 18 )
      goto LABEL_23;
    goto LABEL_9;
  }
  v9 = 1;
  if ( v7 != 18 )
  {
LABEL_9:
    if ( (v7 == 17) == v9 )
      goto LABEL_10;
LABEL_23:
    v17 = 1;
    v11 = "zext source and destination must both be a vector or neither";
    goto LABEL_13;
  }
LABEL_10:
  v10 = sub_BCB060(v3);
  if ( v10 < (unsigned int)sub_BCB060(v6) )
  {
    sub_BF6FE0(a1, a2);
    return;
  }
  v17 = 1;
  v11 = "Type too small for ZExt";
LABEL_13:
  v12 = *(_QWORD *)a1;
  v15 = v11;
  v16 = 3;
  if ( v12 )
  {
    sub_CA0E80(&v15, v12);
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
  else
  {
    *(_BYTE *)(a1 + 152) = 1;
  }
}
