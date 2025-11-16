// Function: sub_1072000
// Address: 0x1072000
//
__int64 __fastcall sub_1072000(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // r15
  __int64 v6; // rcx
  __int64 v7; // r13
  _QWORD *v8; // rax
  void *v9; // rax
  char v10; // si
  char v11; // r14
  char v12; // dl
  __int64 v13; // rdi
  unsigned int v14; // eax
  __int64 v15; // rdi
  char *v16; // rax
  __int64 v17; // rdi
  char *v18; // rax
  __int16 v19; // si
  __int16 v20; // ax
  __int64 v21; // rdi
  __int64 v22; // rdi
  __int64 v24; // rdi
  unsigned __int32 v25; // eax
  __int64 v26; // rax
  void *v27; // rax
  void *v28; // rax
  __int64 v29; // rdi
  void *v30; // rax
  char v32; // [rsp+17h] [rbp-49h]
  _QWORD *v33; // [rsp+18h] [rbp-48h]
  __int64 v34; // [rsp+18h] [rbp-48h]
  unsigned __int8 v35[56]; // [rsp+28h] [rbp-38h] BYREF

  v5 = *(_QWORD *)a2;
  v7 = sub_1071DB0(a1, *(_QWORD *)a2);
  v32 = *(_BYTE *)(a2 + 16);
  if ( v5 == v7 )
  {
    v9 = *(void **)v5;
    if ( !*(_QWORD *)v5 )
    {
      v11 = 0;
      if ( (*(_BYTE *)(v5 + 9) & 0x70) != 0x20 )
        goto LABEL_8;
      if ( *(char *)(v5 + 8) < 0 )
        goto LABEL_8;
      *(_BYTE *)(v5 + 8) |= 8u;
      v9 = sub_E807D0(*(_QWORD *)(v5 + 24));
      *(_QWORD *)v5 = v9;
      if ( !v9 )
        goto LABEL_8;
    }
    goto LABEL_5;
  }
  v8 = sub_1071D30(a1, v7);
  v33 = v8;
  if ( v8 )
    v32 = *((_BYTE *)v8 + 16);
  v9 = *(void **)v7;
  if ( *(_QWORD *)v7
    || (v11 = 10, (*(_BYTE *)(v7 + 9) & 0x70) == 0x20)
    && *(char *)(v7 + 8) >= 0
    && (*(_BYTE *)(v7 + 8) |= 8u, v9 = sub_E807D0(*(_QWORD *)(v7 + 24)), (*(_QWORD *)v7 = v9) != 0) )
  {
LABEL_5:
    v10 = 2;
    if ( off_4C5D170 != v9 )
      v10 = 14;
    v11 = v10;
  }
LABEL_8:
  v12 = *(_BYTE *)(v5 + 8);
  if ( (v12 & 0x40) != 0 )
    v11 |= 0x10u;
  if ( (v12 & 0x20) != 0 )
  {
    v11 |= 1u;
    if ( v5 != v7 )
      goto LABEL_12;
    if ( v9 )
      goto LABEL_13;
  }
  else
  {
    if ( v5 != v7 )
    {
LABEL_12:
      if ( !v9 )
      {
        if ( (*(_BYTE *)(v7 + 9) & 0x70) != 0x20
          || *(char *)(v7 + 8) < 0
          || (*(_BYTE *)(v7 + 8) |= 8u, v28 = sub_E807D0(*(_QWORD *)(v7 + 24)), (*(_QWORD *)v7 = v28) == 0) )
        {
          v34 = v33[1];
          goto LABEL_14;
        }
      }
LABEL_13:
      v34 = sub_1070C50(a1, (_BYTE *)v5, a3, v6);
      goto LABEL_14;
    }
    if ( v9 )
      goto LABEL_13;
    if ( (*(_BYTE *)(v5 + 9) & 0x70) == 0x20 && v12 >= 0 )
    {
      v29 = *(_QWORD *)(v5 + 24);
      *(_BYTE *)(v5 + 8) = v12 | 8;
      v30 = sub_E807D0(v29);
      *(_QWORD *)v5 = v30;
      if ( v30 )
        goto LABEL_13;
    }
    v11 |= 1u;
  }
  LOBYTE(v26) = *(_BYTE *)(v7 + 9) & 0x70;
  if ( (_BYTE)v26 == 32 )
  {
    v34 = 0;
    if ( *(char *)(v7 + 8) < 0 )
      goto LABEL_14;
    *(_BYTE *)(v7 + 8) |= 8u;
    v27 = sub_E807D0(*(_QWORD *)(v7 + 24));
    *(_QWORD *)v7 = v27;
    if ( v27 )
      goto LABEL_13;
    v26 = *(_BYTE *)(v7 + 9) & 0x70;
  }
  v34 = 0;
  if ( (((_BYTE)v26 - 48) & 0xEF) == 0 )
    v34 = *(_QWORD *)(v7 + 24);
LABEL_14:
  v13 = *(_QWORD *)(a1 + 2048);
  v14 = *(_QWORD *)(a2 + 8);
  if ( *(_DWORD *)(a1 + 2056) != 1 )
    v14 = _byteswap_ulong(v14);
  *(_DWORD *)v35 = v14;
  sub_CB6200(v13, v35, 4u);
  v15 = *(_QWORD *)(a1 + 2048);
  v16 = *(char **)(v15 + 32);
  if ( (unsigned __int64)v16 >= *(_QWORD *)(v15 + 24) )
  {
    sub_CB5D20(v15, v11);
  }
  else
  {
    *(_QWORD *)(v15 + 32) = v16 + 1;
    *v16 = v11;
  }
  v17 = *(_QWORD *)(a1 + 2048);
  v18 = *(char **)(v17 + 32);
  if ( (unsigned __int64)v18 >= *(_QWORD *)(v17 + 24) )
  {
    sub_CB5D20(v17, v32);
  }
  else
  {
    *(_QWORD *)(v17 + 32) = v18 + 1;
    *v18 = v32;
  }
  LOBYTE(v19) = 0;
  if ( v5 != v7 )
    v19 = (*(_WORD *)(v5 + 12) >> 9) & 1;
  v20 = sub_1070390(v7, v19);
  v21 = *(_QWORD *)(a1 + 2048);
  if ( *(_DWORD *)(a1 + 2056) != 1 )
    v20 = __ROL2__(v20, 8);
  *(_WORD *)v35 = v20;
  sub_CB6200(v21, v35, 2u);
  if ( (*(_BYTE *)(*(_QWORD *)(a1 + 104) + 8LL) & 1) != 0 )
  {
    v22 = *(_QWORD *)(a1 + 2048);
    if ( *(_DWORD *)(a1 + 2056) != 1 )
      v34 = _byteswap_uint64(v34);
    *(_QWORD *)v35 = v34;
    return sub_CB6200(v22, v35, 8u);
  }
  else
  {
    v24 = *(_QWORD *)(a1 + 2048);
    v25 = v34;
    if ( *(_DWORD *)(a1 + 2056) != 1 )
      v25 = _byteswap_ulong(v34);
    *(_DWORD *)v35 = v25;
    return sub_CB6200(v24, v35, 4u);
  }
}
