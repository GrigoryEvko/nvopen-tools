// Function: sub_926CF0
// Address: 0x926cf0
//
__int64 __fastcall sub_926CF0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 *v6; // r14
  __int64 *v7; // r13
  __int64 v8; // rax
  char v9; // r10
  _BYTE *v10; // rax
  __int64 *v11; // rax
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // r15
  unsigned __int64 v14; // rax
  bool v15; // sf
  __int64 *v16; // rsi
  unsigned int **v17; // r14
  __int64 v18; // rax
  __int64 v19; // rcx
  char v20; // r10
  __int64 v21; // r11
  __int64 v22; // rdi
  __int64 v23; // rax
  __int64 v24; // rax
  _BOOL4 v25; // edx
  __int64 v26; // rbx
  _BYTE *v28; // rax
  unsigned int v29; // r14d
  unsigned __int64 v30; // rcx
  unsigned __int64 j; // rdx
  unsigned __int64 v32; // rax
  unsigned __int64 v33; // rdx
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // rax
  unsigned __int64 i; // rax
  unsigned __int64 v38; // rdx
  char v39; // [rsp+10h] [rbp-C0h]
  __int64 v40; // [rsp+10h] [rbp-C0h]
  __int64 v41; // [rsp+10h] [rbp-C0h]
  char v42; // [rsp+10h] [rbp-C0h]
  __int64 v43; // [rsp+10h] [rbp-C0h]
  char v44; // [rsp+1Bh] [rbp-B5h]
  unsigned int v45; // [rsp+1Ch] [rbp-B4h]
  _BYTE *v46; // [rsp+20h] [rbp-B0h] BYREF
  __int64 v47; // [rsp+28h] [rbp-A8h] BYREF
  const char *v48; // [rsp+30h] [rbp-A0h] BYREF
  char v49; // [rsp+50h] [rbp-80h]
  char v50; // [rsp+51h] [rbp-7Fh]
  _QWORD v51[3]; // [rsp+60h] [rbp-70h] BYREF
  unsigned int v52; // [rsp+78h] [rbp-58h]
  char v53; // [rsp+80h] [rbp-50h]
  char v54; // [rsp+81h] [rbp-4Fh]

  v6 = *(__int64 **)(a3 + 72);
  v7 = (__int64 *)v6[2];
  v8 = sub_91A390(*(_QWORD *)(a2 + 32) + 8LL, *v6, 0, a4);
  v9 = 1;
  if ( *(_BYTE *)(v8 + 8) != 14 )
  {
    v10 = (_BYTE *)sub_9217E0(a2, v6);
    v9 = 0;
    v46 = v10;
    v11 = v6;
    v6 = v7;
    v7 = v11;
  }
  v12 = *v6;
  if ( *(_BYTE *)(*v6 + 140) == 12 )
  {
    do
      v12 = *(_QWORD *)(v12 + 160);
    while ( *(_BYTE *)(v12 + 140) == 12 );
  }
  v14 = *(_QWORD *)(v12 + 160);
  v13 = v14;
  if ( *(_BYTE *)(v14 + 140) == 12 )
  {
    do
      v14 = *(_QWORD *)(v14 + 160);
    while ( *(_BYTE *)(v14 + 140) == 12 );
    v15 = *(char *)(v14 + 142) < 0;
    v14 = v13;
    if ( v15 )
    {
      do
      {
        v14 = *(_QWORD *)(v14 + 160);
        if ( *(_BYTE *)(v14 + 140) != 12 )
          break;
        v14 = *(_QWORD *)(v14 + 160);
      }
      while ( *(_BYTE *)(v14 + 140) == 12 );
    }
    else
    {
      do
        v14 = *(_QWORD *)(v14 + 160);
      while ( *(_BYTE *)(v14 + 140) == 12 );
    }
  }
  v45 = *(_DWORD *)(v14 + 136);
  if ( *((_BYTE *)v6 + 24) != 1 || *((_BYTE *)v6 + 56) != 21 )
  {
    v16 = v6;
    v39 = v9;
    v17 = (unsigned int **)(a2 + 48);
    v18 = sub_92F410(a2, v16);
    v20 = v39;
    v21 = v18;
    goto LABEL_12;
  }
  v42 = v9;
  sub_926800((__int64)v51, a2, v6[9]);
  v29 = v52;
  v45 = v52;
  if ( !sub_91CB00((__int64)v7, &v47) )
  {
    v30 = v45;
    for ( i = v13; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    v32 = *(_QWORD *)(i + 128);
    if ( v45 )
    {
      while ( 1 )
      {
        v38 = v32 % v30;
        v32 = v30;
        if ( !v38 )
          break;
        v30 = v38;
      }
      goto LABEL_28;
    }
    goto LABEL_33;
  }
  if ( v47 )
  {
    v30 = v29;
    for ( j = v13; *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
      ;
    v32 = *(_QWORD *)(j + 128) * v47;
    if ( v29 )
    {
      while ( 1 )
      {
        v33 = v32 % v30;
        v32 = v30;
        if ( !v33 )
          break;
        v30 = v33;
      }
      goto LABEL_28;
    }
LABEL_33:
    v30 = v32;
LABEL_28:
    v45 = v30;
  }
  v50 = 1;
  v44 = v42;
  v17 = (unsigned int **)(a2 + 48);
  v48 = "arraydecay";
  v34 = *(_QWORD *)(a2 + 32);
  v43 = v51[1];
  v49 = 3;
  v35 = sub_91A390(v34 + 8, v51[2], 0, v30);
  v36 = sub_9213A0((unsigned int **)(a2 + 48), v35, v43, 0, 0, (__int64)&v48, 7u);
  v20 = v44;
  v21 = v36;
LABEL_12:
  if ( v20 )
  {
    v41 = v21;
    v28 = (_BYTE *)sub_9217E0(a2, v7);
    v21 = v41;
    v46 = v28;
  }
  v22 = *(_QWORD *)(a2 + 32);
  v40 = v21;
  v54 = 1;
  v51[0] = "arrayidx";
  v53 = 3;
  v23 = sub_91A390(v22 + 8, v13, 0, v19);
  v24 = sub_921130(v17, v23, v40, &v46, 1, (__int64)v51, 3u);
  v25 = 0;
  v26 = v24;
  if ( (*(_BYTE *)(v13 + 140) & 0xFB) == 8 )
    v25 = (sub_8D4C10(v13, dword_4F077C4 != 2) & 2) != 0;
  *(_QWORD *)(a1 + 8) = v26;
  *(_QWORD *)(a1 + 16) = v13;
  *(_DWORD *)(a1 + 24) = v45;
  *(_DWORD *)a1 = 0;
  *(_DWORD *)(a1 + 48) = v25;
  return a1;
}
