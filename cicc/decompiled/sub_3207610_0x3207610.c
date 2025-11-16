// Function: sub_3207610
// Address: 0x3207610
//
__int64 __fastcall sub_3207610(__int64 a1, __int64 a2)
{
  __int64 v4; // rdx
  __int64 v5; // rsi
  int v6; // r9d
  unsigned int i; // eax
  __int64 v8; // rcx
  unsigned int v9; // eax
  unsigned __int8 v11; // di
  __int64 v12; // r15
  bool v13; // dl
  __int64 v14; // rcx
  __int64 v15; // r14
  __int64 v16; // rax
  unsigned __int64 v17; // rdx
  unsigned __int64 v18; // r9
  unsigned __int64 v19; // r8
  __int64 v20; // rsi
  __int64 v21; // rcx
  unsigned __int8 *v22; // rbx
  __int64 v23; // r10
  __int64 v24; // r9
  __int64 v25; // rax
  int v26; // ebx
  __int64 v27; // r15
  int v28; // eax
  int v29; // eax
  unsigned __int64 v30; // rax
  int v31; // r10d
  char v32; // cl
  unsigned __int64 v33; // [rsp+8h] [rbp-78h]
  unsigned __int64 v34; // [rsp+8h] [rbp-78h]
  unsigned __int64 v35; // [rsp+8h] [rbp-78h]
  __int64 v36; // [rsp+10h] [rbp-70h]
  __int64 v37; // [rsp+18h] [rbp-68h]
  unsigned int v38; // [rsp+2Ch] [rbp-54h]
  _BYTE v39[6]; // [rsp+30h] [rbp-50h] BYREF
  int v40; // [rsp+36h] [rbp-4Ah]
  __int64 v41; // [rsp+40h] [rbp-40h]
  unsigned __int64 v42; // [rsp+48h] [rbp-38h]

  v4 = *(unsigned int *)(a1 + 1240);
  v5 = *(_QWORD *)(a1 + 1224);
  if ( (_DWORD)v4 )
  {
    v6 = 1;
    for ( i = (v4 - 1) & (969526130 * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4))); ; i = (v4 - 1) & v9 )
    {
      v8 = v5 + 24LL * i;
      if ( a2 == *(_QWORD *)v8 && !*(_QWORD *)(v8 + 8) )
        break;
      if ( *(_QWORD *)v8 == -4096 && *(_QWORD *)(v8 + 8) == -4096 )
        goto LABEL_10;
      v9 = v6 + i;
      ++v6;
    }
    if ( v8 != v5 + 24 * v4 )
      return *(unsigned int *)(v8 + 16);
  }
LABEL_10:
  v11 = *(_BYTE *)(a2 - 16);
  v12 = a2 - 16;
  v13 = (v11 & 2) != 0;
  if ( (v11 & 2) != 0 )
    v14 = *(_QWORD *)(a2 - 32);
  else
    v14 = v12 - 8LL * ((v11 >> 2) & 0xF);
  v15 = *(_QWORD *)(v14 + 16);
  if ( !v15 )
  {
    v19 = 0;
    goto LABEL_15;
  }
  v16 = sub_B91420(*(_QWORD *)(v14 + 16));
  v11 = *(_BYTE *)(a2 - 16);
  v18 = v17;
  v19 = v17;
  v20 = v16;
  v15 = v16;
  v13 = (v11 & 2) != 0;
  if ( !v18 )
    goto LABEL_15;
  if ( *(_BYTE *)(v16 + v18 - 1) != 62 )
    goto LABEL_15;
  LODWORD(v30) = v18 - 1;
  if ( (int)v18 - 1 < 0 )
    goto LABEL_15;
  v30 = (int)v30;
  v31 = 0;
  while ( 1 )
  {
    v32 = *(_BYTE *)(v20 + v30);
    if ( v32 != 62 )
      break;
    ++v31;
LABEL_33:
    if ( (_DWORD)--v30 == -1 )
      goto LABEL_15;
  }
  if ( v32 != 60 )
    goto LABEL_33;
  if ( --v31 )
    goto LABEL_33;
  if ( v18 <= v30 )
    v30 = v18;
  v19 = v30;
LABEL_15:
  if ( v13 )
    v21 = *(_QWORD *)(a2 - 32);
  else
    v21 = v12 - 8LL * ((v11 >> 2) & 0xF);
  v22 = *(unsigned __int8 **)(v21 + 8);
  v23 = a1 + 632;
  v24 = a1 + 648;
  if ( !v22 )
    goto LABEL_24;
  if ( *v22 == 14 )
  {
    v33 = v19;
    v36 = a1 + 632;
    *(_WORD *)v39 = 5634;
    *(_DWORD *)&v39[2] = sub_3206530(a1, v22, 0);
    v40 = sub_3207400(a1, a2, v22);
    v41 = v15;
    v42 = v33;
    v25 = sub_370AE70(a1 + 648, v39);
    goto LABEL_20;
  }
  if ( (*v22 & 0xFD) != 0x10 )
  {
    v35 = v19;
    v29 = sub_3205980(a1, v22);
    v11 = *(_BYTE *)(a2 - 16);
    v19 = v35;
    v23 = a1 + 632;
    v24 = a1 + 648;
    v26 = v29;
    v13 = (v11 & 2) != 0;
  }
  else
  {
LABEL_24:
    *(_DWORD *)v39 = 0;
    v26 = 0;
  }
  if ( v13 )
    v27 = *(_QWORD *)(a2 - 32);
  else
    v27 = v12 - 8LL * ((v11 >> 2) & 0xF);
  v34 = v19;
  v37 = v24;
  v36 = v23;
  v28 = sub_3206530(a1, *(unsigned __int8 **)(v27 + 32), 0);
  *(_DWORD *)&v39[2] = v26;
  v40 = v28;
  *(_WORD *)v39 = 5633;
  v41 = v15;
  v42 = v34;
  v25 = sub_370ABE0(v37, v39);
LABEL_20:
  v38 = sub_3707F80(v36, v25);
  return sub_31FEC80(a1, a2, v38, 0);
}
