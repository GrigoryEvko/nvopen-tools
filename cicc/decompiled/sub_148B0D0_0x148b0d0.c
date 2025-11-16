// Function: sub_148B0D0
// Address: 0x148b0d0
//
__int64 __fastcall sub_148B0D0(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, char a6)
{
  _QWORD *v9; // rcx
  _QWORD *v10; // rax
  unsigned int v11; // edx
  unsigned int v12; // r8d
  unsigned __int8 v13; // dl
  __int64 v14; // rax
  _QWORD *v16; // rdx
  int v17; // r8d
  __int64 v18; // r12
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r9
  unsigned int v24; // [rsp+8h] [rbp-38h]
  unsigned __int8 v25; // [rsp+Ch] [rbp-34h]

  sub_1412190(a1 + 176, a5);
  v9 = *(_QWORD **)(a1 + 192);
  v10 = *(_QWORD **)(a1 + 184);
  v12 = v11;
  if ( !(_BYTE)v11 )
    return v12;
  v13 = *(_BYTE *)(a5 + 16);
  v12 = 0;
  if ( v13 <= 0x17u )
  {
LABEL_8:
    if ( v9 != v10 )
      goto LABEL_9;
    goto LABEL_12;
  }
  if ( (unsigned int)v13 - 35 > 0x11 )
    goto LABEL_7;
  if ( v13 == 50 )
  {
    if ( a6 )
      goto LABEL_8;
    v12 = sub_148B0D0(a1, a2, a3, a4, *(_QWORD *)(a5 - 48), 0);
    if ( (_BYTE)v12 )
      goto LABEL_29;
    v22 = 0;
    goto LABEL_28;
  }
  if ( v13 == 51 && a6 )
  {
    v12 = sub_148B0D0(a1, a2, a3, a4, *(_QWORD *)(a5 - 48), 1);
    if ( (_BYTE)v12 )
      goto LABEL_29;
    v22 = 1;
LABEL_28:
    v12 = sub_148B0D0(a1, a2, a3, a4, *(_QWORD *)(a5 - 24), v22);
LABEL_29:
    v9 = *(_QWORD **)(a1 + 192);
    v10 = *(_QWORD **)(a1 + 184);
    goto LABEL_8;
  }
LABEL_7:
  v12 = 0;
  if ( v13 != 75 )
    goto LABEL_8;
  v17 = *(_WORD *)(a5 + 18) & 0x7FFF;
  if ( a6 )
    v17 = sub_15FF0F0(*(_WORD *)(a5 + 18) & 0x7FFF);
  v24 = v17;
  v18 = sub_146F1B0(a1, *(_QWORD *)(a5 - 48));
  v19 = sub_146F1B0(a1, *(_QWORD *)(a5 - 24));
  v12 = sub_148AAB0(a1, a2, a3, a4, v24, v18, v19);
  v10 = *(_QWORD **)(a1 + 184);
  if ( *(_QWORD **)(a1 + 192) != v10 )
  {
LABEL_9:
    v25 = v12;
    v10 = (_QWORD *)sub_16CC9F0(a1 + 176, a5);
    v12 = v25;
    if ( a5 == *v10 )
    {
      v20 = *(_QWORD *)(a1 + 192);
      if ( v20 == *(_QWORD *)(a1 + 184) )
        v21 = *(unsigned int *)(a1 + 204);
      else
        v21 = *(unsigned int *)(a1 + 200);
      v16 = (_QWORD *)(v20 + 8 * v21);
    }
    else
    {
      v14 = *(_QWORD *)(a1 + 192);
      if ( v14 != *(_QWORD *)(a1 + 184) )
        return v12;
      v10 = (_QWORD *)(v14 + 8LL * *(unsigned int *)(a1 + 204));
      v16 = v10;
    }
    goto LABEL_16;
  }
LABEL_12:
  v16 = &v10[*(unsigned int *)(a1 + 204)];
  if ( v16 == v10 )
  {
LABEL_30:
    v10 = v16;
  }
  else
  {
    while ( a5 != *v10 )
    {
      if ( v16 == ++v10 )
        goto LABEL_30;
    }
  }
LABEL_16:
  if ( v16 != v10 )
  {
    *v10 = -2;
    ++*(_DWORD *)(a1 + 208);
  }
  return v12;
}
