// Function: sub_30F48E0
// Address: 0x30f48e0
//
char __fastcall sub_30F48E0(__int64 a1, __int64 a2)
{
  _QWORD *v4; // rax
  char *v5; // rsi
  __int64 v6; // rdi
  __int64 v7; // rbx
  int v8; // eax
  __int64 v9; // rcx
  int v10; // r8d
  unsigned int v11; // edx
  __int64 *v12; // rax
  __int64 v13; // r9
  char *v14; // r13
  char v15; // al
  __int64 v16; // r8
  __int64 *v17; // r15
  __int64 v18; // rax
  __int64 v19; // r8
  __int64 v20; // r9
  _QWORD *v21; // r15
  _QWORD *v22; // rax
  __int64 v23; // r9
  __int64 v24; // rdx
  unsigned __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // r11
  __int64 v30; // rsi
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // r8
  __int64 v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // r8
  __int64 *v38; // rsi
  __int64 v39; // rax
  __int64 v40; // rsi
  __int64 v41; // rax
  __int64 v42; // r9
  __int64 v43; // rdx
  __int64 v44; // rax
  __int64 v45; // r14
  __int64 *v46; // rbx
  __int64 *v47; // r15
  __int64 v48; // rax
  __int64 v49; // r14
  __int64 *v50; // r14
  __int64 v51; // rcx
  __int64 v52; // r8
  __int64 v53; // rcx
  __int64 v54; // r8
  __int64 v55; // rcx
  __int64 v56; // r8
  char result; // al
  int v58; // eax
  __int64 v59; // rax
  __int64 v60; // rdx
  int v61; // r10d
  __int64 *v62; // rax
  unsigned int v63; // [rsp+4h] [rbp-4Ch]
  __int64 v64; // [rsp+8h] [rbp-48h]
  __int64 v65; // [rsp+8h] [rbp-48h]
  __int64 v66; // [rsp+8h] [rbp-48h]
  __int64 v67; // [rsp+10h] [rbp-40h]
  __int64 v68; // [rsp+10h] [rbp-40h]
  __int64 *v69; // [rsp+10h] [rbp-40h]
  __int64 v70; // [rsp+10h] [rbp-40h]
  __int64 *v71; // [rsp+10h] [rbp-40h]
  __int64 v72; // [rsp+18h] [rbp-38h]

  v4 = sub_DCADF0(*(__int64 **)(a1 + 104), *(_QWORD *)(a1 + 8));
  v5 = *(char **)(a1 + 8);
  v6 = *(_QWORD *)(a2 + 8);
  v7 = (__int64)v4;
  v8 = *(_DWORD *)(a2 + 24);
  v9 = *((_QWORD *)v5 + 5);
  if ( !v8 )
    return 0;
  v10 = v8 - 1;
  v11 = (v8 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
  v12 = (__int64 *)(v6 + 16LL * v11);
  v13 = *v12;
  if ( v9 != *v12 )
  {
    v58 = 1;
    while ( v13 != -4096 )
    {
      v61 = v58 + 1;
      v11 = v10 & (v58 + v11);
      v12 = (__int64 *)(v6 + 16LL * v11);
      v13 = *v12;
      if ( v9 == *v12 )
        goto LABEL_3;
      v58 = v61;
    }
    return 0;
  }
LABEL_3:
  v14 = (char *)v12[1];
  if ( !v14 )
    return 0;
  v15 = *v5;
  v16 = 0;
  if ( (unsigned __int8)*v5 > 0x1Cu )
  {
    if ( v15 == 61 || v15 == 62 )
    {
      v16 = *((_QWORD *)v5 - 4);
    }
    else if ( v15 == 63 )
    {
      v16 = *(_QWORD *)&v5[-32 * (*((_DWORD *)v5 + 1) & 0x7FFFFFF)];
    }
  }
  v17 = sub_DDFBA0(*(_QWORD *)(a1 + 104), v16, v14);
  v18 = sub_D97190(*(_QWORD *)(a1 + 104), (__int64)v17);
  if ( *(_WORD *)(v18 + 24) != 15 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    return 0;
  }
  *(_QWORD *)(a1 + 16) = v18;
  v72 = a1 + 64;
  if ( (unsigned __int8)sub_30F3FE0(a1, (__int64)v17, a1 + 24) )
  {
    v59 = *(unsigned int *)(a1 + 72);
    if ( v59 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 76) )
    {
      sub_C8D5F0(v72, (const void *)(a1 + 80), v59 + 1, 8u, v19, v20);
      v59 = *(unsigned int *)(a1 + 72);
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 64) + 8 * v59) = v7;
    v60 = *(_QWORD *)(a1 + 16);
    ++*(_DWORD *)(a1 + 72);
    v21 = sub_DCC810(*(__int64 **)(a1 + 104), (__int64)v17, v60, 0, 0);
  }
  else
  {
    v21 = sub_DCC810(*(__int64 **)(a1 + 104), (__int64)v17, *(_QWORD *)(a1 + 16), 0, 0);
    v22 = sub_DCADF0(*(__int64 **)(a1 + 104), *(_QWORD *)(a1 + 8));
    sub_30B8650(*(__int64 **)(a1 + 104), (__int64)v21, a1 + 24, v72, (__int64)v22, v23);
  }
  v27 = *(unsigned int *)(a1 + 32);
  if ( !(_DWORD)v27 || (v24 = *(unsigned int *)(a1 + 72), (_DWORD)v24 != (_DWORD)v27) || !(_DWORD)v24 )
  {
    if ( *((_WORD *)v21 + 12) != 8 )
      goto LABEL_46;
    if ( v21[5] != 2 )
      goto LABEL_46;
    v64 = *(_QWORD *)(a1 + 104);
    v67 = *(_QWORD *)v21[4];
    v28 = sub_D33D80(v21, v64, v24, v25, v26);
    v29 = v67;
    if ( *(_WORD *)(v67 + 24) == 8 )
      goto LABEL_46;
    v68 = v28;
    if ( *(_WORD *)(v28 + 24) == 8 )
      goto LABEL_46;
    if ( !sub_DADE90(v64, v29, (__int64)v14) )
      goto LABEL_46;
    v30 = v68;
    v69 = (__int64 *)v64;
    if ( !sub_DADE90(v64, v30, (__int64)v14) )
      goto LABEL_46;
    v34 = sub_D33D80(v21, v64, v31, v32, v33);
    v38 = (__int64 *)v34;
    if ( v34 )
    {
      v65 = v34;
      v38 = (__int64 *)v34;
      if ( (unsigned __int8)sub_DBEC00((__int64)v69, v34) )
        v38 = sub_DCAF50(v69, v65, 0);
    }
    if ( (__int64 *)v7 != v38 )
    {
LABEL_46:
      *(_DWORD *)(a1 + 32) = 0;
      *(_DWORD *)(a1 + 72) = 0;
      return 0;
    }
    if ( *((_WORD *)v21 + 12) == 8 )
    {
      v39 = sub_D33D80(v21, *(_QWORD *)(a1 + 104), v35, v36, v37);
      if ( v39 )
      {
        v40 = v39;
        if ( (unsigned __int8)sub_DBEC00(*(_QWORD *)(a1 + 104), v39) )
        {
          v66 = v21[6];
          v63 = *((_WORD *)v21 + 14) & 7;
          v71 = *(__int64 **)(a1 + 104);
          v62 = sub_DCAF50(v71, v40, 0);
          v21 = sub_DC1960((__int64)v71, *(_QWORD *)v21[4], (__int64)v62, v66, v63);
        }
      }
    }
    v41 = sub_DCC290(*(__int64 **)(a1 + 104), (__int64)v21, v7);
    v43 = *(unsigned int *)(a1 + 32);
    v26 = v43 + 1;
    if ( v43 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 36) )
    {
      v70 = v41;
      sub_C8D5F0(a1 + 24, (const void *)(a1 + 40), v43 + 1, 8u, v26, v42);
      v43 = *(unsigned int *)(a1 + 32);
      v41 = v70;
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8 * v43) = v41;
    v44 = *(unsigned int *)(a1 + 72);
    v25 = *(unsigned int *)(a1 + 76);
    ++*(_DWORD *)(a1 + 32);
    if ( v44 + 1 > v25 )
    {
      sub_C8D5F0(v72, (const void *)(a1 + 80), v44 + 1, 8u, v26, v42);
      v44 = *(unsigned int *)(a1 + 72);
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 64) + 8 * v44) = v7;
    v27 = *(unsigned int *)(a1 + 32);
    ++*(_DWORD *)(a1 + 72);
  }
  v45 = 8 * v27;
  v46 = *(__int64 **)(a1 + 24);
  v47 = &v46[v27];
  v48 = (8 * v27) >> 3;
  v49 = v45 >> 5;
  if ( v49 )
  {
    v50 = &v46[4 * v49];
    while ( sub_30F4860(a1, *v46, (__int64)v14, v25, v26) )
    {
      if ( !sub_30F4860(a1, v46[1], (__int64)v14, v55, v56) )
        return v47 == v46 + 1;
      if ( !sub_30F4860(a1, v46[2], (__int64)v14, v51, v52) )
        return v47 == v46 + 2;
      if ( !sub_30F4860(a1, v46[3], (__int64)v14, v53, v54) )
        return v47 == v46 + 3;
      v46 += 4;
      if ( v50 == v46 )
      {
        v48 = v47 - v46;
        goto LABEL_53;
      }
    }
    return v47 == v46;
  }
LABEL_53:
  if ( v48 == 2 )
    goto LABEL_61;
  if ( v48 == 3 )
  {
    if ( !sub_30F4860(a1, *v46, (__int64)v14, v25, v26) )
      return v47 == v46;
    ++v46;
LABEL_61:
    if ( sub_30F4860(a1, *v46, (__int64)v14, v25, v26) )
    {
      ++v46;
      goto LABEL_63;
    }
    return v47 == v46;
  }
  if ( v48 != 1 )
    return 1;
LABEL_63:
  result = sub_30F4860(a1, *v46, (__int64)v14, v25, v26);
  if ( !result )
    return v47 == v46;
  return result;
}
