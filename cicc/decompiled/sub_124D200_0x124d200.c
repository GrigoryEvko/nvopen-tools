// Function: sub_124D200
// Address: 0x124d200
//
__int64 __fastcall sub_124D200(__int64 *a1, __int64 a2, unsigned int a3, __int64 a4)
{
  __int64 v6; // rbx
  __int64 v7; // r12
  __int64 v8; // r15
  _BYTE *v9; // rdx
  char v10; // al
  char v11; // dl
  char v12; // r15
  char v13; // al
  __int64 v14; // rsi
  unsigned __int8 v15; // r15
  char v16; // cl
  int v17; // eax
  unsigned __int64 v18; // rcx
  __int64 v19; // rdi
  __int64 v20; // rax
  char v21; // al
  unsigned __int64 v22; // r8
  bool v24; // zf
  char v25; // al
  char v26; // al
  _BYTE *v27; // [rsp+8h] [rbp-68h]
  char v28; // [rsp+10h] [rbp-60h]
  unsigned __int64 v29; // [rsp+10h] [rbp-60h]
  __int64 v30; // [rsp+10h] [rbp-60h]
  char v31; // [rsp+1Fh] [rbp-51h]
  char v34; // [rsp+2Ch] [rbp-44h]
  unsigned __int8 v35; // [rsp+2Ch] [rbp-44h]
  _QWORD v36[7]; // [rsp+38h] [rbp-38h] BYREF

  v6 = *(_QWORD *)a4;
  v7 = sub_E5C930(a1, *(_QWORD *)a4);
  if ( v7 )
    v31 = (((*(_BYTE *)(v6 + 9) & 0x70) - 48) & 0xE0) == 0;
  else
    v31 = 1;
  v8 = v6;
  v34 = sub_EA1780(v6);
  v28 = sub_EA1630(v6);
  if ( (unsigned int)sub_EA1630(v6) == 10 )
  {
LABEL_9:
    if ( !v7 )
    {
LABEL_13:
      v11 = 10;
      goto LABEL_14;
    }
    v10 = sub_EA1630(v7);
LABEL_11:
    if ( (unsigned __int8)v10 > 2u )
    {
      v11 = v10;
      if ( v10 != 6 )
        goto LABEL_14;
    }
    goto LABEL_13;
  }
  while ( (*(_BYTE *)(v8 + 9) & 0x70) == 0x20 )
  {
    v9 = *(_BYTE **)(v8 + 24);
    *(_BYTE *)(v8 + 8) |= 8u;
    if ( *v9 != 2 )
      break;
    v27 = v9;
    if ( (*(_DWORD *)v9 & 0xFFFF00) != 0 || (unsigned __int8)sub_EA1630(v8) == 6 )
      break;
    v8 = *((_QWORD *)v27 + 2);
    if ( (unsigned int)sub_EA1630(v8) == 10 )
      goto LABEL_9;
  }
  if ( !v7 )
  {
    v11 = v28;
    goto LABEL_14;
  }
  v10 = sub_EA1630(v7);
  v11 = v10;
  if ( v28 == 2 )
  {
    if ( (unsigned __int8)v10 <= 1u || v10 == 6 )
      v11 = 2;
    goto LABEL_14;
  }
  if ( (unsigned __int8)v28 <= 2u )
  {
    if ( v28 == 1 )
    {
      v24 = v10 == 0;
      v25 = 1;
      if ( !v24 )
        v25 = v11;
      v11 = v25;
    }
    goto LABEL_14;
  }
  if ( v28 == 6 )
  {
    if ( v10 == 10 || (unsigned __int8)v10 <= 2u )
      v11 = 6;
    goto LABEL_14;
  }
  if ( v28 == 10 )
    goto LABEL_11;
LABEL_14:
  v35 = (16 * v34) | v11;
  v12 = sub_EA1680(v6);
  v13 = sub_EA16B0(v6);
  v14 = *(_QWORD *)a4;
  v15 = v12 | v13;
  if ( (((*(_BYTE *)(*(_QWORD *)a4 + 9LL) & 0x70) - 48) & 0xE0) != 0 )
  {
    v30 = *(_QWORD *)a4;
    if ( (unsigned __int8)sub_E5BD10((__int64)a1, v14, (__int64)v36) )
    {
      v26 = sub_E5BBB0((__int64)a1, v30);
      v18 = v36[0];
      if ( v26 )
        v18 = v36[0] | 1LL;
      v14 = *(_QWORD *)a4;
    }
    else
    {
      v14 = *(_QWORD *)a4;
      v18 = 0;
    }
  }
  else
  {
    v16 = 0;
    v17 = (*(_DWORD *)(v14 + 8) >> 15) & 0x1F;
    if ( (_BYTE)v17 )
      v16 = v17 - 1;
    v18 = 1LL << v16;
  }
  v19 = *(_QWORD *)(v14 + 32);
  if ( v19 )
    goto LABEL_25;
  if ( v7 )
  {
    v19 = *(_QWORD *)(v7 + 32);
    while ( (*(_BYTE *)(v6 + 9) & 0x70) == 0x20 )
    {
      v20 = *(_QWORD *)(v6 + 24);
      if ( *(_BYTE *)v20 != 2 )
        break;
      v6 = *(_QWORD *)(v20 + 16);
      if ( *(_QWORD *)(v6 + 32) )
      {
        v19 = *(_QWORD *)(v6 + 32);
        goto LABEL_25;
      }
    }
  }
  v22 = 0;
  if ( v19 )
  {
LABEL_25:
    v29 = v18;
    v21 = sub_E81940(v19, v36, (__int64)a1);
    v18 = v29;
    if ( !v21 )
      sub_C64ED0("Size expression must be absolute.", 1u);
    v22 = v36[0];
  }
  return sub_124CF30(a2, a3, v35, v18, v22, v15, *(_DWORD *)(a4 + 24), v31);
}
