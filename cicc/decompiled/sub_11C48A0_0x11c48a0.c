// Function: sub_11C48A0
// Address: 0x11c48a0
//
char __fastcall sub_11C48A0(__int64 a1, unsigned __int8 *a2)
{
  unsigned __int64 v2; // rax
  __int64 v3; // rcx
  __int64 v5; // rbx
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // r13
  int v9; // r13d
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // r15
  __int64 v14; // r14
  unsigned __int64 v15; // rbx
  __int64 v16; // rax
  __int64 v17; // rsi
  unsigned int v18; // ebx
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  unsigned __int64 v23; // rax
  __int64 v24; // r14
  __int64 v25; // r15
  unsigned __int64 v26; // rbx
  __int64 v27; // rax
  __int64 v28; // rsi
  unsigned int v29; // ebx
  __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // r8
  __int64 v33; // r9
  unsigned __int64 v34; // rax
  __int64 v35; // rax
  unsigned int v36; // r15d
  __int64 v37; // rax
  __int64 v38; // rcx
  __int64 v39; // rax
  unsigned int v40; // r15d
  __int64 v41; // rax
  __int64 v42; // rcx
  __int128 v44; // [rsp-118h] [rbp-118h]
  __int128 v45; // [rsp-118h] [rbp-118h]
  __int128 v46; // [rsp-118h] [rbp-118h]
  __int128 v47; // [rsp-118h] [rbp-118h]
  __int64 v48; // [rsp-F8h] [rbp-F8h]
  unsigned __int64 v49; // [rsp-D8h] [rbp-D8h]
  __int64 v50; // [rsp-B8h] [rbp-B8h]
  __int64 v51; // [rsp-98h] [rbp-98h]
  _QWORD v52[4]; // [rsp-78h] [rbp-78h] BYREF
  __int64 *v53; // [rsp-58h] [rbp-58h] BYREF
  unsigned __int64 v54; // [rsp-50h] [rbp-50h]
  __int64 v55; // [rsp-48h] [rbp-48h]

  LODWORD(v2) = *a2;
  if ( (unsigned __int8)(v2 - 34) > 0x33u )
    return v2;
  v3 = 0x8000000000041LL;
  if ( _bittest64(&v3, (unsigned int)(v2 - 34)) )
  {
    v52[0] = a2;
    v53 = v52;
    v54 = a1;
    if ( (_DWORD)v2 == 40 )
    {
      v5 = 32LL * (unsigned int)sub_B491D0((__int64)a2);
      if ( (a2[7] & 0x80u) == 0 )
        goto LABEL_14;
    }
    else
    {
      v5 = 0;
      if ( (_DWORD)v2 != 85 )
      {
        v5 = 64;
        if ( (_DWORD)v2 != 34 )
          BUG();
      }
      if ( (a2[7] & 0x80u) == 0 )
        goto LABEL_14;
    }
    v6 = sub_BD2BC0((__int64)a2);
    v8 = v6 + v7;
    if ( (a2[7] & 0x80u) == 0 )
    {
      if ( (unsigned int)(v8 >> 4) )
        goto LABEL_41;
    }
    else if ( (unsigned int)((v8 - sub_BD2BC0((__int64)a2)) >> 4) )
    {
      if ( (a2[7] & 0x80u) != 0 )
      {
        v9 = *(_DWORD *)(sub_BD2BC0((__int64)a2) + 8);
        if ( (a2[7] & 0x80u) == 0 )
          BUG();
        v10 = sub_BD2BC0((__int64)a2);
        v12 = 32LL * (unsigned int)(*(_DWORD *)(v10 + v11 - 4) - v9);
        goto LABEL_15;
      }
LABEL_41:
      BUG();
    }
LABEL_14:
    v12 = 0;
LABEL_15:
    sub_11C44E0(&v53, *(_QWORD *)(v52[0] + 72LL), (32LL * (*((_DWORD *)a2 + 1) & 0x7FFFFFF) - 32 - v5 - v12) >> 5);
    v2 = *(_QWORD *)(v52[0] - 32LL);
    if ( v2 && !*(_BYTE *)v2 && *(_QWORD *)(v2 + 24) == *(_QWORD *)(v52[0] + 80LL) )
      LOBYTE(v2) = sub_11C44E0(&v53, *(_QWORD *)(v2 + 120), *(_QWORD *)(v2 + 104));
    return v2;
  }
  if ( (_BYTE)v2 == 61 )
  {
    v13 = *((_QWORD *)a2 + 1);
    v14 = *((_QWORD *)a2 - 4);
    v15 = 1LL << (*((_WORD *)a2 + 1) >> 1);
    v16 = sub_B43CA0((__int64)a2);
    v17 = v13;
    _BitScanReverse64(&v15, v15);
    v18 = v15 ^ 0x3F;
    v53 = (__int64 *)sub_9208B0(v16 + 312, v13);
    v23 = ((unsigned __int64)v53 + 7) >> 3;
    v54 = v19;
    if ( (_DWORD)v23 )
    {
      LODWORD(v48) = 90;
      *((_QWORD *)&v46 + 1) = (unsigned int)v23;
      *(_QWORD *)&v46 = v48;
      sub_11C1FA0(a1, v13, v19, v20, v21, v22, v46, (unsigned __int8 *)v14);
      v35 = *(_QWORD *)(v14 + 8);
      if ( (unsigned int)*(unsigned __int8 *)(v35 + 8) - 17 <= 1 )
        v35 = **(_QWORD **)(v35 + 16);
      v36 = *(_DWORD *)(v35 + 8);
      v37 = sub_B43CB0((__int64)a2);
      v17 = v36 >> 8;
      if ( !sub_B2F070(v37, v17) )
      {
        LODWORD(v49) = 43;
        sub_11C1FA0(a1, v17, v19, v38, v21, v22, v49, (unsigned __int8 *)v14);
      }
    }
    v2 = 0x8000000000000000LL >> v18;
    if ( 0x8000000000000000LL >> v18 != 1 )
    {
      LODWORD(v50) = 86;
      *((_QWORD *)&v44 + 1) = 0x8000000000000000LL >> v18;
      *(_QWORD *)&v44 = v50;
      LOBYTE(v2) = sub_11C1FA0(a1, v17, v19, v18, v21, v22, v44, (unsigned __int8 *)v14);
    }
  }
  else if ( (_BYTE)v2 == 62 )
  {
    v24 = *((_QWORD *)a2 - 4);
    v25 = *(_QWORD *)(*((_QWORD *)a2 - 8) + 8LL);
    v26 = 1LL << (*((_WORD *)a2 + 1) >> 1);
    v27 = sub_B43CA0((__int64)a2);
    v28 = v25;
    _BitScanReverse64(&v26, v26);
    v29 = v26 ^ 0x3F;
    v53 = (__int64 *)sub_9208B0(v27 + 312, v25);
    v34 = ((unsigned __int64)v53 + 7) >> 3;
    v54 = v30;
    if ( (_DWORD)v34 )
    {
      LODWORD(v51) = 90;
      *((_QWORD *)&v47 + 1) = (unsigned int)v34;
      *(_QWORD *)&v47 = v51;
      sub_11C1FA0(a1, v25, v30, v31, v32, v33, v47, (unsigned __int8 *)v24);
      v39 = *(_QWORD *)(v24 + 8);
      if ( (unsigned int)*(unsigned __int8 *)(v39 + 8) - 17 <= 1 )
        v39 = **(_QWORD **)(v39 + 16);
      v40 = *(_DWORD *)(v39 + 8);
      v41 = sub_B43CB0((__int64)a2);
      v28 = v40 >> 8;
      if ( !sub_B2F070(v41, v28) )
      {
        LODWORD(v52[0]) = 43;
        v52[1] = 0;
        v52[2] = v24;
        sub_11C1FA0(a1, v28, v30, v42, v32, v33, v52[0], (unsigned __int8 *)v24);
      }
    }
    v2 = 0x8000000000000000LL >> v29;
    if ( 0x8000000000000000LL >> v29 != 1 )
    {
      LODWORD(v53) = 86;
      v54 = 0x8000000000000000LL >> v29;
      v55 = v24;
      *((_QWORD *)&v45 + 1) = 0x8000000000000000LL >> v29;
      *(_QWORD *)&v45 = v53;
      LOBYTE(v2) = sub_11C1FA0(a1, v28, v30, v29, v32, v33, v45, (unsigned __int8 *)v24);
    }
  }
  return v2;
}
