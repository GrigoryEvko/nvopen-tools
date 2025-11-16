// Function: sub_8111C0
// Address: 0x8111c0
//
__int64 __fastcall sub_8111C0(__int64 a1, unsigned int a2, int a3, int a4, unsigned int a5, _QWORD *a6, __int64 a7)
{
  __int64 v9; // rax
  char v10; // cl
  __int64 i; // rbx
  int v12; // r14d
  __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // r10
  char v16; // dl
  __int64 v17; // r9
  __int64 v18; // rcx
  int v19; // r8d
  _QWORD *v20; // rax
  int v21; // r8d
  char v22; // cl
  char *v23; // rax
  char v24; // si
  __int64 result; // rax
  int v26; // r15d
  unsigned __int8 v27; // dl
  __int64 v28; // rax
  int v29; // r11d
  int v30[2]; // [rsp+0h] [rbp-70h]
  unsigned __int8 v31; // [rsp+0h] [rbp-70h]
  int v32; // [rsp+0h] [rbp-70h]
  __int64 v33; // [rsp+0h] [rbp-70h]
  int v35; // [rsp+8h] [rbp-68h]
  int v36[2]; // [rsp+8h] [rbp-68h]
  __int64 v38; // [rsp+10h] [rbp-60h]
  char v39; // [rsp+18h] [rbp-58h]
  int v40; // [rsp+18h] [rbp-58h]
  int v42; // [rsp+2Ch] [rbp-44h] BYREF
  __int64 *v43; // [rsp+30h] [rbp-40h] BYREF
  _QWORD v44[7]; // [rsp+38h] [rbp-38h] BYREF

  v9 = *(_QWORD *)(a1 + 152);
  v10 = *(_BYTE *)(v9 + 140);
  i = v9;
  v42 = 0;
  v43 = 0;
  if ( v10 == 12 )
  {
    do
      i = *(_QWORD *)(i + 160);
    while ( *(_BYTE *)(i + 140) == 12 );
  }
  v12 = unk_4F0697C;
  if ( !unk_4F0697C )
    goto LABEL_9;
  if ( !*(_QWORD *)(a1 + 240) || (*(_BYTE *)(a1 + 195) & 1) == 0 )
  {
    v12 = 0;
LABEL_9:
    v39 = 0;
    goto LABEL_10;
  }
  v13 = *(_QWORD *)(*(_QWORD *)a1 + 96LL);
  if ( v13 )
  {
    v14 = *(_QWORD *)(v13 + 32);
    switch ( *(_BYTE *)(v14 + 80) )
    {
      case 4:
      case 5:
        v28 = *(_QWORD *)(*(_QWORD *)(v14 + 96) + 80LL);
        break;
      case 6:
        v28 = *(_QWORD *)(*(_QWORD *)(v14 + 96) + 32LL);
        break;
      case 9:
      case 0xA:
        v28 = *(_QWORD *)(*(_QWORD *)(v14 + 96) + 56LL);
        break;
      case 0x13:
      case 0x14:
      case 0x15:
      case 0x16:
        v28 = *(_QWORD *)(v14 + 88);
        break;
      default:
        BUG();
    }
    for ( i = *(_QWORD *)(*(_QWORD *)(v28 + 176) + 152LL); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    v40 = a3;
    v33 = *(_QWORD *)(v28 + 104);
    if ( (unsigned int)sub_80C5A0(v33, 59, 0, 0, v44, (_QWORD *)a7) )
    {
      v12 = 1;
LABEL_34:
      if ( *(_QWORD *)(a1 + 240) )
      {
        v44[0] = *(_QWORD *)(a1 + 240);
        sub_811CB0(v44, 0, 0, a7);
      }
      goto LABEL_36;
    }
    v29 = v40;
    v15 = v33;
    v39 = 1;
    v12 = 1;
    if ( v29 )
      goto LABEL_12;
    goto LABEL_11;
  }
  v39 = 1;
  v12 = 1;
LABEL_10:
  v15 = 0;
  if ( a3 )
    goto LABEL_16;
LABEL_11:
  *(_QWORD *)v30 = v15;
  sub_811730(a1, 11, &v42, &v43, a5, a7);
  v15 = *(_QWORD *)v30;
LABEL_12:
  if ( v15 && !*(_QWORD *)(a7 + 40) )
    sub_80A250(v15, 59, 0, a7);
  v9 = *(_QWORD *)(a1 + 152);
  v10 = *(_BYTE *)(v9 + 140);
LABEL_16:
  v16 = *(_BYTE *)(a1 + 174);
  v17 = 0;
  if ( v16 == 3 )
    v17 = *(_QWORD *)(i + 160);
  if ( v10 == 12 )
  {
    do
      v9 = *(_QWORD *)(v9 + 160);
    while ( *(_BYTE *)(v9 + 140) == 12 );
  }
  v18 = *(_QWORD *)(v9 + 168);
  v19 = 0;
  v20 = *(_QWORD **)v18;
  if ( *(_QWORD *)v18 )
  {
    do
    {
      v20 = (_QWORD *)*v20;
      ++v19;
    }
    while ( v20 );
  }
  v21 = v19 - ((*(_QWORD *)(v18 + 40) == 0) - 1);
  if ( v16 == 5 )
  {
    LOBYTE(a4) = *(_BYTE *)(a1 + 176);
    v22 = 0;
  }
  else if ( (unsigned __int8)(v16 - 1) > 1u )
  {
    LOBYTE(a4) = 0;
    v22 = 0;
  }
  else
  {
    if ( !*(_QWORD *)(a1 + 320) && (*(_BYTE *)(a1 + 205) & 0x1C) == 0 )
    {
      v32 = v21;
      *(_QWORD *)v36 = v17;
      sub_7FA1F0(a1);
      v21 = v32;
      v17 = *(_QWORD *)v36;
    }
    if ( a4 )
    {
      LOBYTE(a4) = 0;
      v22 = 1;
    }
    else
    {
      v22 = (*(_BYTE *)(a1 + 205) >> 2) & 7;
    }
  }
  if ( a6 )
    *a6 = *(_QWORD *)a7;
  v31 = v22;
  v35 = v21;
  v38 = v17;
  sub_80AD10(a1);
  v23 = 0;
  v24 = *(_BYTE *)(a1 + 174);
  if ( v24 == 4 && (*(_BYTE *)(a1 + 89) & 0x40) == 0 )
  {
    if ( (*(_BYTE *)(a1 + 89) & 8) != 0 )
    {
      v23 = *(char **)(a1 + 24);
      if ( v23 )
        v23 += 11;
    }
    else
    {
      v23 = *(char **)(a1 + 8);
      if ( v23 )
        v23 += 11;
    }
  }
  sub_810E90(a1, v24, a4, v31, v35, v38, v23, a7);
  if ( (*(_BYTE *)(a1 + 201) & 4) != 0 )
    sub_80B920(*(__int64 **)(a1 + 104), (_QWORD *)a7);
  if ( v39 )
    goto LABEL_34;
LABEL_36:
  sub_80C110(v42, v43, (_QWORD *)a7);
  result = a2;
  if ( !a2 )
  {
    v26 = *(_DWORD *)(a7 + 52);
    v27 = *(_BYTE *)(a1 + 174) - 1;
    *(_DWORD *)(a7 + 52) = (*(_BYTE *)(a1 + 207) & 0x20) != 0;
    if ( v27 > 2u )
    {
      if ( (*(_BYTE *)(i - 8) & 8) != 0 )
        i = *(_QWORD *)(i + 176);
      if ( v12 )
        sub_80F5E0(*(_QWORD *)(i + 160), 0, (_QWORD *)a7);
    }
    else if ( (*(_BYTE *)(i - 8) & 8) != 0 )
    {
      i = *(_QWORD *)(i + 176);
    }
    result = sub_80FC70(*(_QWORD *)(i + 168), (_QWORD *)a7);
    *(_DWORD *)(a7 + 52) = v26;
  }
  return result;
}
