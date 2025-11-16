// Function: sub_8851B0
// Address: 0x8851b0
//
__int64 __fastcall sub_8851B0(
        __int64 a1,
        __int64 *a2,
        unsigned int a3,
        int a4,
        __int64 a5,
        _QWORD *a6,
        _BYTE *a7,
        _DWORD *a8,
        _BOOL4 *a9,
        _DWORD *a10)
{
  _QWORD *v10; // rax
  __int64 v11; // r13
  _QWORD *v12; // rbx
  __int64 *v13; // rax
  char v14; // r10
  unsigned __int8 v15; // di
  __int64 v16; // rcx
  char v17; // dl
  __int64 *v18; // r15
  __int64 v19; // rcx
  __int64 v20; // r8
  char v21; // dl
  __int64 v22; // rdx
  char v23; // r8
  __int64 v24; // rcx
  char v25; // dl
  __int64 v26; // rsi
  char v27; // si
  _QWORD *v28; // r14
  __int64 v29; // r12
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rdi
  char v33; // al
  bool v34; // zf
  _BOOL4 v35; // eax
  _QWORD *v36; // r12
  __int64 **v37; // rdi

  v10 = sub_8848A0(a1, a2, a3, a4);
  if ( !v10 )
    return 0;
  v11 = v10[1];
  v12 = v10;
  v13 = (__int64 *)*v10;
  if ( !v13 )
  {
    v18 = v12;
    v34 = (*(_BYTE *)(v11 + 82) & 4) == 0;
    *a8 = (*(_BYTE *)(v11 + 82) & 4) != 0;
    if ( v34 )
      goto LABEL_50;
    goto LABEL_35;
  }
  v14 = *(_BYTE *)(v11 + 80);
  v15 = *((_BYTE *)v12 + 24);
  v16 = v11;
  v17 = v14;
  if ( v14 == 16 )
  {
    v16 = **(_QWORD **)(v11 + 88);
    v17 = *(_BYTE *)(v16 + 80);
    if ( v17 != 24 )
    {
LABEL_5:
      v18 = v12;
      if ( v17 == 3 )
        goto LABEL_17;
      goto LABEL_6;
    }
  }
  else if ( v14 != 24 )
  {
    goto LABEL_5;
  }
  v18 = v12;
  v17 = *(_BYTE *)(*(_QWORD *)(v16 + 88) + 80LL);
  if ( v17 == 3 )
    goto LABEL_17;
LABEL_6:
  v18 = v13;
  if ( dword_4F077C4 != 2 )
    goto LABEL_9;
  v18 = v12;
  if ( (unsigned __int8)(v17 - 4) <= 2u )
  {
LABEL_17:
    v22 = v11;
    if ( v14 == 16 )
      goto LABEL_57;
LABEL_18:
    if ( v14 == 24 )
      goto LABEL_58;
    goto LABEL_19;
  }
  v18 = v13;
  do
  {
LABEL_9:
    v19 = v18[1];
    v20 = v19;
    v21 = *(_BYTE *)(v19 + 80);
    if ( v21 == 16 )
    {
      v20 = **(_QWORD **)(v19 + 88);
      v21 = *(_BYTE *)(v20 + 80);
    }
    if ( v21 == 24 )
      v21 = *(_BYTE *)(*(_QWORD *)(v20 + 88) + 80LL);
    if ( v21 == 3 || dword_4F077C4 == 2 && (unsigned __int8)(v21 - 4) <= 2u )
    {
      v13 = (__int64 *)*v18;
      v14 = *(_BYTE *)(v19 + 80);
      v11 = v18[1];
      goto LABEL_17;
    }
    v18 = (__int64 *)*v18;
  }
  while ( v18 );
  v18 = v12;
  v22 = v11;
  if ( v14 != 16 )
    goto LABEL_18;
LABEL_57:
  v22 = **(_QWORD **)(v11 + 88);
  v14 = *(_BYTE *)(v22 + 80);
  if ( v14 == 24 )
  {
LABEL_58:
    v23 = 1;
    v14 = *(_BYTE *)(*(_QWORD *)(v22 + 88) + 80LL);
    if ( v14 == 3 )
      goto LABEL_22;
LABEL_20:
    v23 = 0;
    if ( dword_4F077C4 == 2 )
      v23 = (unsigned __int8)(v14 - 4) <= 2u;
    goto LABEL_22;
  }
LABEL_19:
  v23 = 1;
  if ( v14 != 3 )
    goto LABEL_20;
LABEL_22:
  while ( v13 )
  {
    if ( *((_BYTE *)v13 + 24) < v15 )
    {
      v24 = v13[1];
      v25 = *(_BYTE *)(v24 + 80);
      v26 = v24;
      if ( v25 == 16 )
      {
        v26 = **(_QWORD **)(v24 + 88);
        v25 = *(_BYTE *)(v26 + 80);
      }
      if ( v25 == 24 )
        v25 = *(_BYTE *)(*(_QWORD *)(v26 + 88) + 80LL);
      v27 = 1;
      if ( v25 != 3 )
      {
        v27 = 0;
        if ( dword_4F077C4 == 2 )
          v27 = (unsigned __int8)(v25 - 4) <= 2u;
      }
      if ( v23 == v27 )
      {
        v18 = v13;
        v11 = v13[1];
        v15 = *(_BYTE *)(v24 + 96) & 3;
      }
    }
    v13 = (__int64 *)*v13;
  }
  *a8 = 1;
LABEL_35:
  if ( !dword_4D047DC )
    goto LABEL_50;
  v28 = v12;
  v29 = 0;
  *a10 = 1;
  while ( 1 )
  {
    v32 = v28[1];
    v33 = *(_BYTE *)(v32 + 80);
    if ( v33 != 16 )
      goto LABEL_37;
    if ( (*(_BYTE *)(v32 + 82) & 4) != 0 && (*(_BYTE *)(v32 + 96) & 0x20) == 0 )
      break;
    v32 = **(_QWORD **)(v32 + 88);
    v33 = *(_BYTE *)(v32 + 80);
    if ( v33 == 24 )
    {
      v32 = *(_QWORD *)(v32 + 88);
      v33 = *(_BYTE *)(v32 + 80);
    }
LABEL_37:
    if ( v33 != 3 )
      break;
    if ( !*(_BYTE *)(v32 + 104) )
      break;
    v30 = *(_QWORD *)(v32 + 88);
    if ( (*(_BYTE *)(v30 + 177) & 0x10) == 0 || !*(_QWORD *)(*(_QWORD *)(v30 + 168) + 168LL) )
      break;
    v31 = sub_880FE0(v32);
    if ( v29 )
    {
      if ( v31 != v29 )
        break;
    }
    else
    {
      v29 = v31;
    }
    v28 = (_QWORD *)*v28;
    if ( !v28 )
      goto LABEL_50;
  }
  *a10 = 0;
LABEL_50:
  *a7 = *((_BYTE *)v18 + 24);
  *a6 = v18[2];
  v35 = 0;
  v18[2] = 0;
  if ( *(_BYTE *)(v11 + 80) == 16 )
    v35 = (*(_BYTE *)(v11 + 96) & 0xC) != 0;
  *a9 = v35;
  do
  {
    v36 = v12;
    v12 = (_QWORD *)*v12;
    v37 = (__int64 **)v36[2];
    if ( v37 )
      sub_5EBA80(v37);
    *v36 = qword_4F5FE50;
    qword_4F5FE50 = (__int64)v36;
  }
  while ( v12 );
  return v11;
}
