// Function: sub_2B6F180
// Address: 0x2b6f180
//
bool __fastcall sub_2B6F180(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r13
  __int64 v5; // rdi
  unsigned __int8 v6; // al
  __int64 v8; // rbx
  unsigned int v9; // r14d
  unsigned int v10; // eax
  __int64 v12; // r13
  unsigned int v13; // r14d
  unsigned int v14; // edi
  __int64 v15; // rsi
  int v16; // r10d
  __int64 v17; // r8
  unsigned int v18; // eax
  __int64 v19; // rdx
  __int64 v20; // rcx
  unsigned __int64 v21; // r8
  _QWORD *v22; // rcx
  unsigned __int64 v23; // r15
  int v24; // r11d
  __int64 v25; // r10
  unsigned int v26; // eax
  __int64 v27; // rdx
  __int64 v28; // r9
  unsigned __int64 v29; // rax
  _QWORD *v30; // r13
  int v31; // eax
  int v32; // edx
  __int64 *v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rbx
  unsigned __int8 *v36; // r8
  unsigned __int8 *v37; // r14
  unsigned int v38; // eax
  unsigned int v39; // edx
  __int64 v40; // rax
  __int64 v41; // r10
  __int64 v42; // rsi
  unsigned int v43; // r9d
  __int64 v44; // rdx
  __int64 v45; // rax
  __int64 v46; // r10
  unsigned int v47; // eax
  __int64 v48; // rax
  __int64 *v49; // rdx
  __int64 v50; // rax
  __int64 v51; // rdx
  int v52; // eax
  int v53; // edx
  __int64 *v54; // rax
  __int64 v55; // rdx
  unsigned int v56; // eax
  unsigned __int8 v57; // si
  int v58; // esi
  _QWORD *v59; // [rsp+0h] [rbp-60h]
  _QWORD *v60; // [rsp+0h] [rbp-60h]
  _QWORD *v61; // [rsp+0h] [rbp-60h]
  unsigned __int8 *v62; // [rsp+8h] [rbp-58h]
  unsigned __int64 v63; // [rsp+8h] [rbp-58h]
  unsigned __int64 v64; // [rsp+8h] [rbp-58h]
  __int64 v65; // [rsp+10h] [rbp-50h] BYREF
  __int64 v66; // [rsp+18h] [rbp-48h] BYREF
  __int64 v67[8]; // [rsp+20h] [rbp-40h] BYREF

  v4 = *(_QWORD *)(a3 + 8);
  v5 = *(_QWORD *)(a2 + 8);
  v65 = a2;
  v66 = a3;
  v6 = *(_BYTE *)(v4 + 8);
  if ( *(_BYTE *)(v5 + 8) < v6 )
    return 1;
  if ( *(_BYTE *)(v5 + 8) > v6 )
    return 0;
  v8 = a3;
  v9 = sub_BCB060(v5);
  v10 = sub_BCB060(v4);
  if ( v9 < v10 )
    return 1;
  if ( v9 > v10 )
    return 0;
  v12 = a1[1];
  v13 = *(_DWORD *)(v12 + 24);
  if ( !v13 )
  {
    v67[0] = 0;
    ++*(_QWORD *)v12;
LABEL_81:
    v58 = 2 * v13;
LABEL_82:
    sub_2B5BE90(v12, v58);
    sub_2B41380(v12, &v65, v67);
    v53 = *(_DWORD *)(v12 + 16) + 1;
    goto LABEL_55;
  }
  v14 = v13 - 1;
  v15 = *(_QWORD *)(v12 + 8);
  v16 = 1;
  v17 = 0;
  v18 = (v13 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v19 = v15 + 56LL * v18;
  v20 = *(_QWORD *)v19;
  if ( a2 == *(_QWORD *)v19 )
  {
LABEL_8:
    v21 = *(unsigned int *)(v19 + 16);
    v22 = *(_QWORD **)(v19 + 8);
    v23 = v21;
    goto LABEL_9;
  }
  while ( v20 != -4096 )
  {
    if ( !v17 && v20 == -8192 )
      v17 = v19;
    v18 = v14 & (v16 + v18);
    v19 = v15 + 56LL * v18;
    v20 = *(_QWORD *)v19;
    if ( a2 == *(_QWORD *)v19 )
      goto LABEL_8;
    ++v16;
  }
  if ( !v17 )
    v17 = v19;
  v67[0] = v17;
  v52 = *(_DWORD *)(v12 + 16);
  ++*(_QWORD *)v12;
  v53 = v52 + 1;
  if ( 4 * (v52 + 1) >= 3 * v13 )
    goto LABEL_81;
  if ( v13 - *(_DWORD *)(v12 + 20) - v53 <= v13 >> 3 )
  {
    v58 = v13;
    goto LABEL_82;
  }
LABEL_55:
  *(_DWORD *)(v12 + 16) = v53;
  v54 = (__int64 *)v67[0];
  if ( *(_QWORD *)v67[0] != -4096 )
    --*(_DWORD *)(v12 + 20);
  v55 = v65;
  v22 = v54 + 3;
  v54[1] = (__int64)(v54 + 3);
  *v54 = v55;
  v54[2] = 0x400000000LL;
  v12 = a1[1];
  v13 = *(_DWORD *)(v12 + 24);
  if ( !v13 )
  {
    v67[0] = 0;
    v56 = 0;
    v21 = 0;
    ++*(_QWORD *)v12;
    goto LABEL_59;
  }
  v15 = *(_QWORD *)(v12 + 8);
  v8 = v66;
  v14 = v13 - 1;
  v21 = 0;
  v23 = 0;
LABEL_9:
  v24 = 1;
  v25 = 0;
  v26 = v14 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
  v27 = v15 + 56LL * v26;
  v28 = *(_QWORD *)v27;
  if ( *(_QWORD *)v27 == v8 )
  {
LABEL_10:
    v29 = *(unsigned int *)(v27 + 16);
    v30 = *(_QWORD **)(v27 + 8);
    if ( v29 > v21 )
      return 1;
    goto LABEL_26;
  }
  while ( v28 != -4096 )
  {
    if ( !v25 && v28 == -8192 )
      v25 = v27;
    v26 = v14 & (v24 + v26);
    v27 = v15 + 56LL * v26;
    v28 = *(_QWORD *)v27;
    if ( *(_QWORD *)v27 == v8 )
      goto LABEL_10;
    ++v24;
  }
  if ( !v25 )
    v25 = v27;
  v67[0] = v25;
  v31 = *(_DWORD *)(v12 + 16);
  ++*(_QWORD *)v12;
  v32 = v31 + 1;
  if ( 4 * (v31 + 1) < 3 * v13 )
  {
    if ( v13 - *(_DWORD *)(v12 + 20) - v32 <= v13 >> 3 )
    {
      v61 = v22;
      v64 = v21;
      sub_2B5BE90(v12, v13);
      sub_2B41380(v12, &v66, v67);
      v22 = v61;
      v21 = v64;
      v32 = *(_DWORD *)(v12 + 16) + 1;
    }
    goto LABEL_23;
  }
  v56 = v13;
  v13 = v23;
LABEL_59:
  v60 = v22;
  v23 = v13;
  v63 = v21;
  sub_2B5BE90(v12, 2 * v56);
  sub_2B41380(v12, &v66, v67);
  v21 = v63;
  v22 = v60;
  v32 = *(_DWORD *)(v12 + 16) + 1;
LABEL_23:
  *(_DWORD *)(v12 + 16) = v32;
  v33 = (__int64 *)v67[0];
  if ( *(_QWORD *)v67[0] != -4096 )
    --*(_DWORD *)(v12 + 20);
  v34 = v66;
  v30 = v33 + 3;
  v33[1] = (__int64)(v33 + 3);
  *v33 = v34;
  v33[2] = 0x400000000LL;
  v29 = 0;
LABEL_26:
  if ( v21 > v29 || (int)v23 <= 0 )
    return 0;
  v35 = 0;
  while ( 1 )
  {
    v36 = (unsigned __int8 *)v22[v35];
    v37 = (unsigned __int8 *)v30[v35];
    v38 = *v36;
    v39 = *v37;
    if ( (unsigned __int8)v38 <= 0x1Cu )
      break;
    if ( (unsigned __int8)v39 <= 0x1Cu )
      return 1;
    v40 = *((_QWORD *)v36 + 5);
    v41 = 0;
    v42 = *(_QWORD *)(*a1 + 40LL);
    if ( v40 )
    {
      v41 = (unsigned int)(*(_DWORD *)(v40 + 44) + 1);
      LODWORD(v40) = *(_DWORD *)(v40 + 44) + 1;
    }
    v43 = *(_DWORD *)(v42 + 32);
    v44 = 0;
    if ( (unsigned int)v40 < v43 )
      v44 = *(_QWORD *)(*(_QWORD *)(v42 + 24) + 8 * v41);
    v45 = *((_QWORD *)v37 + 5);
    if ( v45 )
    {
      v46 = (unsigned int)(*(_DWORD *)(v45 + 44) + 1);
      v47 = *(_DWORD *)(v45 + 44) + 1;
    }
    else
    {
      v46 = 0;
      v47 = 0;
    }
    if ( v43 <= v47 )
    {
      if ( v44 )
        return 0;
      return v44 != 0;
    }
    v48 = *(_QWORD *)(*(_QWORD *)(v42 + 24) + 8 * v46);
    if ( !v44 )
    {
      v44 = *(_QWORD *)(*(_QWORD *)(v42 + 24) + 8 * v46);
      return v44 != 0;
    }
    if ( !v48 )
      return 0;
    if ( v48 != v44 )
      return *(_DWORD *)(v44 + 72) < *(_DWORD *)(v48 + 72);
    v49 = *(__int64 **)(*a1 + 16LL);
    v67[0] = v22[v35];
    v62 = v36;
    v59 = v22;
    v67[1] = (__int64)v37;
    v50 = sub_2B5F980(v67, 2u, v49);
    if ( v50 == 0 || v51 == 0 )
      return (unsigned int)*v62 - 29 < (unsigned int)*v37 - 29;
    v22 = v59;
    if ( v50 != v51 )
      return (unsigned int)*v62 - 29 < (unsigned int)*v37 - 29;
LABEL_66:
    if ( v23 == ++v35 )
      return 0;
  }
  if ( (unsigned __int8)v39 > 0x1Cu )
    return 0;
  if ( (unsigned __int8)v38 > 0x15u )
  {
    if ( (unsigned __int8)v39 > 0x15u )
    {
      if ( v38 != v39 )
        return v38 <= v39;
      goto LABEL_66;
    }
    if ( (_BYTE)v39 == 12 )
      return 1;
    if ( (_BYTE)v39 != 13 )
      return 0;
LABEL_65:
    if ( v38 - 12 > 1 )
      return 1;
    goto LABEL_66;
  }
  v57 = v38 - 12;
  if ( (unsigned __int8)v39 <= 0x15u )
  {
    if ( (_BYTE)v39 != 12 && (_BYTE)v39 != 13 )
    {
      if ( v57 <= 1u )
        return 0;
      goto LABEL_66;
    }
    if ( v57 > 1u )
      return 1;
    goto LABEL_65;
  }
  return v57 > 1u;
}
