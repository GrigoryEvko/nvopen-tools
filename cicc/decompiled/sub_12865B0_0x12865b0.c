// Function: sub_12865B0
// Address: 0x12865b0
//
__int64 __fastcall sub_12865B0(__int64 a1, _QWORD *a2, __int64 a3)
{
  __int64 *v5; // r15
  __int64 *v6; // r14
  __int64 v7; // rax
  char v8; // r10
  __int64 v9; // rax
  __int64 *v10; // rax
  unsigned __int64 v11; // rax
  unsigned __int64 v12; // r13
  unsigned __int64 v13; // rax
  bool v14; // sf
  __int64 v15; // rax
  char v16; // r10
  _BYTE *v17; // r14
  __int64 v18; // rax
  __int64 v19; // rax
  bool v20; // cc
  __int64 v21; // r11
  _QWORD *v22; // r15
  _BOOL4 v23; // edx
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // r11
  __int64 v30; // r10
  __int64 v31; // rax
  __int64 v32; // rdi
  unsigned __int64 *v33; // r14
  __int64 v34; // rax
  unsigned __int64 v35; // rcx
  __int64 v36; // rsi
  __int64 v37; // rsi
  __int64 v38; // rax
  unsigned int v39; // r14d
  unsigned __int64 v40; // rcx
  unsigned __int64 j; // rdx
  unsigned __int64 v42; // rax
  unsigned __int64 v43; // rdx
  __int64 v44; // rax
  __int64 v45; // rax
  unsigned __int64 i; // rax
  unsigned __int64 v47; // rdx
  __int64 v48; // [rsp+8h] [rbp-B8h]
  char v49; // [rsp+10h] [rbp-B0h]
  __int64 v50; // [rsp+10h] [rbp-B0h]
  char v51; // [rsp+10h] [rbp-B0h]
  __int64 v52; // [rsp+18h] [rbp-A8h]
  __int64 v53; // [rsp+18h] [rbp-A8h]
  __int64 *v54; // [rsp+20h] [rbp-A0h]
  __int64 v55; // [rsp+20h] [rbp-A0h]
  unsigned int v56; // [rsp+28h] [rbp-98h]
  unsigned int v57; // [rsp+2Ch] [rbp-94h]
  __int64 v58; // [rsp+30h] [rbp-90h] BYREF
  __int64 v59; // [rsp+38h] [rbp-88h] BYREF
  _QWORD v60[2]; // [rsp+40h] [rbp-80h] BYREF
  char v61; // [rsp+50h] [rbp-70h]
  char v62; // [rsp+51h] [rbp-6Fh]
  _BYTE v63[8]; // [rsp+60h] [rbp-60h] BYREF
  _BYTE *v64; // [rsp+68h] [rbp-58h]
  unsigned int v65; // [rsp+70h] [rbp-50h]

  v5 = *(__int64 **)(a3 + 72);
  v6 = (__int64 *)v5[2];
  v54 = v6;
  v7 = sub_127A030(a2[4] + 8LL, *v5, 0);
  v8 = 1;
  if ( *(_BYTE *)(v7 + 8) != 15 )
  {
    v9 = sub_127F8B0(a2, v5);
    v8 = 0;
    v52 = v9;
    v10 = v5;
    v5 = v6;
    v54 = v10;
  }
  v11 = *v5;
  if ( *(_BYTE *)(*v5 + 140) == 12 )
  {
    do
      v11 = *(_QWORD *)(v11 + 160);
    while ( *(_BYTE *)(v11 + 140) == 12 );
  }
  v13 = *(_QWORD *)(v11 + 160);
  v12 = v13;
  if ( *(_BYTE *)(v13 + 140) == 12 )
  {
    do
      v13 = *(_QWORD *)(v13 + 160);
    while ( *(_BYTE *)(v13 + 140) == 12 );
    v14 = *(char *)(v13 + 142) < 0;
    v13 = v12;
    if ( v14 )
    {
      do
      {
        v13 = *(_QWORD *)(v13 + 160);
        if ( *(_BYTE *)(v13 + 140) != 12 )
          break;
        v13 = *(_QWORD *)(v13 + 160);
      }
      while ( *(_BYTE *)(v13 + 140) == 12 );
    }
    else
    {
      do
        v13 = *(_QWORD *)(v13 + 160);
      while ( *(_BYTE *)(v13 + 140) == 12 );
    }
  }
  v57 = *(_DWORD *)(v13 + 136);
  if ( *((_BYTE *)v5 + 24) != 1 || *((_BYTE *)v5 + 56) != 21 )
  {
    v49 = v8;
    v15 = sub_128F980(a2, v5);
    v16 = v49;
    v17 = (_BYTE *)v15;
    goto LABEL_12;
  }
  v51 = v8;
  sub_1286D80(v63, a2, v5[9]);
  v39 = v65;
  v57 = v65;
  if ( !sub_127C7B0((__int64)v54, &v59) )
  {
    v40 = v57;
    for ( i = v12; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    v42 = *(_QWORD *)(i + 128);
    if ( v57 )
    {
      while ( 1 )
      {
        v47 = v42 % v40;
        v42 = v40;
        if ( !v47 )
          break;
        v40 = v47;
      }
      goto LABEL_46;
    }
    goto LABEL_54;
  }
  if ( v59 )
  {
    v40 = v39;
    for ( j = v12; *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
      ;
    v42 = *(_QWORD *)(j + 128) * v59;
    if ( v39 )
    {
      while ( 1 )
      {
        v43 = v42 % v40;
        v42 = v40;
        if ( !v43 )
          break;
        v40 = v43;
      }
      goto LABEL_46;
    }
LABEL_54:
    LODWORD(v40) = v42;
LABEL_46:
    v57 = v40;
  }
  v62 = 1;
  v60[0] = "arraydecay";
  v61 = 3;
  v44 = sub_1286300(a2 + 6, 0, v64, 0, 0, (__int64)v60);
  v16 = v51;
  v17 = (_BYTE *)v44;
LABEL_12:
  if ( v16 )
    v52 = sub_127F8B0(a2, v54);
  v62 = 1;
  v60[0] = "arrayidx";
  v18 = a2[4];
  v61 = 3;
  v19 = sub_127A030(v18 + 8, v12, 0);
  v20 = v17[16] <= 0x10u;
  v21 = v19;
  v58 = v52;
  if ( v20 && *(_BYTE *)(v52 + 16) <= 0x10u )
  {
    v63[4] = 0;
    v59 = v52;
    v22 = (_QWORD *)sub_15A2E80(v19, (_DWORD)v17, (unsigned int)&v59, 1, 1, (unsigned int)v63, 0);
  }
  else
  {
    LOWORD(v65) = 257;
    if ( !v19 )
    {
      v45 = *(_QWORD *)v17;
      if ( *(_BYTE *)(*(_QWORD *)v17 + 8LL) == 16 )
        v45 = **(_QWORD **)(v45 + 16);
      v21 = *(_QWORD *)(v45 + 24);
    }
    v55 = v21;
    v25 = sub_1648A60(72, 2);
    v22 = (_QWORD *)v25;
    if ( v25 )
    {
      v53 = v25;
      v50 = v25 - 48;
      v26 = *(_QWORD *)v17;
      if ( *(_BYTE *)(*(_QWORD *)v17 + 8LL) == 16 )
        v26 = **(_QWORD **)(v26 + 16);
      v56 = *(_DWORD *)(v26 + 8) >> 8;
      v27 = sub_15F9F50(v55, &v58, 1);
      v28 = sub_1646BA0(v27, v56);
      v29 = v55;
      v30 = v28;
      v31 = *(_QWORD *)v17;
      if ( *(_BYTE *)(*(_QWORD *)v17 + 8LL) == 16 || (v31 = *(_QWORD *)v58, *(_BYTE *)(*(_QWORD *)v58 + 8LL) == 16) )
      {
        v38 = sub_16463B0(v30, *(_QWORD *)(v31 + 32));
        v29 = v55;
        v30 = v38;
      }
      v48 = v29;
      sub_15F1EA0(v22, v30, 32, v50, 2, 0);
      v22[7] = v48;
      v22[8] = sub_15F9F50(v48, &v58, 1);
      sub_15F9CE0(v22, v17, &v58, 1, v63);
    }
    else
    {
      v53 = 0;
    }
    sub_15FA2E0(v22, 1);
    v32 = a2[7];
    if ( v32 )
    {
      v33 = (unsigned __int64 *)a2[8];
      sub_157E9D0(v32 + 40, v22);
      v34 = v22[3];
      v35 = *v33;
      v22[4] = v33;
      v35 &= 0xFFFFFFFFFFFFFFF8LL;
      v22[3] = v35 | v34 & 7;
      *(_QWORD *)(v35 + 8) = v22 + 3;
      *v33 = *v33 & 7 | (unsigned __int64)(v22 + 3);
    }
    sub_164B780(v53, v60);
    v36 = a2[6];
    if ( v36 )
    {
      v59 = a2[6];
      sub_1623A60(&v59, v36, 2);
      if ( v22[6] )
        sub_161E7C0(v22 + 6);
      v37 = v59;
      v22[6] = v59;
      if ( v37 )
        sub_1623210(&v59, v37, v22 + 6);
    }
  }
  v23 = 0;
  if ( (*(_BYTE *)(v12 + 140) & 0xFB) == 8 )
    v23 = (sub_8D4C10(v12, dword_4F077C4 != 2) & 2) != 0;
  *(_QWORD *)(a1 + 8) = v22;
  *(_DWORD *)a1 = 0;
  *(_DWORD *)(a1 + 16) = v57;
  *(_DWORD *)(a1 + 40) = v23;
  return a1;
}
