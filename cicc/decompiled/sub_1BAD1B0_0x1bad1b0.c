// Function: sub_1BAD1B0
// Address: 0x1bad1b0
//
void __fastcall sub_1BAD1B0(__int64 a1)
{
  __int64 v2; // rdx
  __int64 v3; // r12
  __int64 v4; // rdi
  int v5; // eax
  __int64 v6; // rcx
  __int64 v7; // rsi
  int v8; // edi
  unsigned int v9; // edx
  __int64 *v10; // rax
  __int64 v11; // r8
  __int64 v12; // rax
  __int64 v13; // rcx
  __int64 v14; // rdi
  __int64 v15; // rax
  __int64 v16; // rsi
  __int64 v17; // r10
  int v18; // r8d
  unsigned int v19; // r9d
  __int64 *v20; // rdx
  __int64 v21; // r11
  __int64 *v22; // rax
  __int64 v23; // r12
  unsigned int v24; // r9d
  __int64 *v25; // rdx
  __int64 v26; // r10
  __int64 v27; // r13
  __int64 v28; // rax
  _BYTE *v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rcx
  int v32; // r8d
  int v33; // r9d
  _BYTE *v34; // rsi
  __int64 v35; // rdi
  __int64 v36; // rax
  __int64 v37; // rsi
  int v38; // r8d
  __int64 v39; // r10
  unsigned int v40; // r9d
  __int64 *v41; // rdx
  __int64 v42; // r11
  __int64 *v43; // rax
  __int64 v44; // rbx
  unsigned int v45; // r9d
  __int64 *v46; // rdx
  __int64 v47; // r10
  __int64 v48; // r12
  __int64 v49; // rax
  _BYTE *v50; // rax
  __int64 v51; // rdx
  __int64 v52; // rcx
  int v53; // r8d
  int v54; // r9d
  _BYTE *v55; // rsi
  int v56; // edx
  int v57; // r11d
  int v58; // edx
  int v59; // edx
  int v60; // r11d
  int v61; // eax
  int v62; // edx
  int v63; // ebx
  int v64; // r12d
  int v65; // r9d
  __int64 v66[5]; // [rsp+8h] [rbp-28h] BYREF

  sub_1465150(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 112LL), *(_QWORD *)(a1 + 8));
  v2 = *(_QWORD *)(a1 + 24);
  v3 = *(_QWORD *)(a1 + 32);
  v4 = 0;
  v5 = *(_DWORD *)(v2 + 24);
  if ( v5 )
  {
    v6 = *(_QWORD *)(a1 + 200);
    v7 = *(_QWORD *)(v2 + 8);
    v8 = v5 - 1;
    v9 = (v5 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
    v10 = (__int64 *)(v7 + 16LL * v9);
    v11 = *v10;
    if ( v6 == *v10 )
    {
LABEL_3:
      v4 = v10[1];
    }
    else
    {
      v61 = 1;
      while ( v11 != -8 )
      {
        v65 = v61 + 1;
        v9 = v8 & (v61 + v9);
        v10 = (__int64 *)(v7 + 16LL * v9);
        v11 = *v10;
        if ( v6 == *v10 )
          goto LABEL_3;
        v61 = v65;
      }
      v4 = 0;
    }
  }
  v12 = sub_13FCB50(v4);
  sub_1BACEB0(v3, *(_QWORD *)(a1 + 184), v12);
  sub_1BACEB0(*(_QWORD *)(a1 + 32), *(_QWORD *)(a1 + 176), **(_QWORD **)(a1 + 216));
  v13 = *(_QWORD *)(a1 + 32);
  v14 = *(_QWORD *)(a1 + 208);
  v15 = *(unsigned int *)(v13 + 48);
  v16 = *(_QWORD *)(v13 + 32);
  if ( !(_DWORD)v15 )
  {
LABEL_58:
    *(_BYTE *)(v13 + 72) = 0;
    BUG();
  }
  v17 = *(_QWORD *)(a1 + 176);
  v18 = v15 - 1;
  v19 = (v15 - 1) & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
  v20 = (__int64 *)(v16 + 16LL * v19);
  v21 = *v20;
  if ( v17 == *v20 )
  {
LABEL_6:
    v22 = (__int64 *)(v16 + 16 * v15);
    if ( v22 != v20 )
    {
      v23 = v20[1];
      goto LABEL_8;
    }
  }
  else
  {
    v62 = 1;
    while ( v21 != -8 )
    {
      v64 = v62 + 1;
      v19 = v18 & (v62 + v19);
      v20 = (__int64 *)(v16 + 16LL * v19);
      v21 = *v20;
      if ( v17 == *v20 )
        goto LABEL_6;
      v62 = v64;
    }
    v22 = (__int64 *)(v16 + 16 * v15);
  }
  v23 = 0;
LABEL_8:
  v24 = v18 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
  v25 = (__int64 *)(v16 + 16LL * v24);
  v26 = *v25;
  if ( v14 != *v25 )
  {
    v56 = 1;
    while ( v26 != -8 )
    {
      v57 = v56 + 1;
      v24 = v18 & (v56 + v24);
      v25 = (__int64 *)(v16 + 16LL * v24);
      v26 = *v25;
      if ( v14 == *v25 )
        goto LABEL_9;
      v56 = v57;
    }
    goto LABEL_58;
  }
LABEL_9:
  if ( v25 == v22 )
    goto LABEL_58;
  v27 = v25[1];
  *(_BYTE *)(v13 + 72) = 0;
  v28 = *(_QWORD *)(v27 + 8);
  if ( v28 != v23 )
  {
    v66[0] = v27;
    v29 = sub_1B8E500(*(_QWORD **)(v28 + 24), *(_QWORD *)(v28 + 32), v66);
    sub_15CDF70(*(_QWORD *)(v27 + 8) + 24LL, v29);
    *(_QWORD *)(v27 + 8) = v23;
    v66[0] = v27;
    v34 = *(_BYTE **)(v23 + 32);
    if ( v34 == *(_BYTE **)(v23 + 40) )
    {
      sub_15CE310(v23 + 24, v34, v66);
    }
    else
    {
      if ( v34 )
      {
        *(_QWORD *)v34 = v27;
        v34 = *(_BYTE **)(v23 + 32);
      }
      v34 += 8;
      *(_QWORD *)(v23 + 32) = v34;
    }
    if ( *(_DWORD *)(v27 + 16) != *(_DWORD *)(*(_QWORD *)(v27 + 8) + 16LL) + 1 )
      sub_1B8E2B0(v27, (__int64)v34, v30, v31, v32, v33);
  }
  v13 = *(_QWORD *)(a1 + 32);
  v35 = *(_QWORD *)(a1 + 192);
  v36 = *(unsigned int *)(v13 + 48);
  v37 = *(_QWORD *)(v13 + 32);
  if ( !(_DWORD)v36 )
    goto LABEL_58;
  v38 = v36 - 1;
  v39 = **(_QWORD **)(a1 + 216);
  v40 = (v36 - 1) & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4));
  v41 = (__int64 *)(v37 + 16LL * v40);
  v42 = *v41;
  if ( v39 == *v41 )
  {
LABEL_19:
    v43 = (__int64 *)(v37 + 16 * v36);
    if ( v43 != v41 )
    {
      v44 = v41[1];
      goto LABEL_21;
    }
  }
  else
  {
    v58 = 1;
    while ( v42 != -8 )
    {
      v63 = v58 + 1;
      v40 = v38 & (v58 + v40);
      v41 = (__int64 *)(v37 + 16LL * v40);
      v42 = *v41;
      if ( v39 == *v41 )
        goto LABEL_19;
      v58 = v63;
    }
    v43 = (__int64 *)(v37 + 16 * v36);
  }
  v44 = 0;
LABEL_21:
  v45 = v38 & (((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4));
  v46 = (__int64 *)(v37 + 16LL * v45);
  v47 = *v46;
  if ( v35 != *v46 )
  {
    v59 = 1;
    while ( v47 != -8 )
    {
      v60 = v59 + 1;
      v45 = v38 & (v59 + v45);
      v46 = (__int64 *)(v37 + 16LL * v45);
      v47 = *v46;
      if ( v35 == *v46 )
        goto LABEL_22;
      v59 = v60;
    }
    goto LABEL_58;
  }
LABEL_22:
  if ( v46 == v43 )
    goto LABEL_58;
  v48 = v46[1];
  *(_BYTE *)(v13 + 72) = 0;
  v49 = *(_QWORD *)(v48 + 8);
  if ( v49 != v44 )
  {
    v66[0] = v48;
    v50 = sub_1B8E500(*(_QWORD **)(v49 + 24), *(_QWORD *)(v49 + 32), v66);
    sub_15CDF70(*(_QWORD *)(v48 + 8) + 24LL, v50);
    *(_QWORD *)(v48 + 8) = v44;
    v66[0] = v48;
    v55 = *(_BYTE **)(v44 + 32);
    if ( v55 == *(_BYTE **)(v44 + 40) )
    {
      sub_15CE310(v44 + 24, v55, v66);
    }
    else
    {
      if ( v55 )
      {
        *(_QWORD *)v55 = v48;
        v55 = *(_BYTE **)(v44 + 32);
      }
      v55 += 8;
      *(_QWORD *)(v44 + 32) = v55;
    }
    if ( *(_DWORD *)(v48 + 16) != *(_DWORD *)(*(_QWORD *)(v48 + 8) + 16LL) + 1 )
      sub_1B8E2B0(v48, (__int64)v55, v51, v52, v53, v54);
  }
}
