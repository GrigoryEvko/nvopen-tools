// Function: sub_1EE8590
// Address: 0x1ee8590
//
void __fastcall sub_1EE8590(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r13
  __int64 v6; // r8
  unsigned int *v7; // r15
  unsigned int *v8; // r13
  unsigned int v9; // r12d
  __int64 v10; // r9
  unsigned int v11; // esi
  _DWORD *v12; // rdx
  __int64 v13; // rcx
  unsigned int v14; // eax
  __int64 v15; // rdi
  int v16; // r10d
  unsigned int v17; // eax
  __int64 v18; // rcx
  unsigned int *v19; // r8
  __int64 v20; // r15
  unsigned int *v21; // r13
  unsigned int v22; // r12d
  __int64 v23; // r11
  unsigned int v24; // ecx
  unsigned int v25; // esi
  unsigned __int64 v26; // r9
  _BYTE *v27; // r10
  unsigned int v28; // eax
  __int64 v29; // rdi
  _DWORD *v30; // rdx
  int v31; // r9d
  __int64 v32; // rcx
  int *v33; // rbx
  int *j; // r13
  int v35; // r12d
  unsigned int v36; // edi
  __int64 v37; // rcx
  unsigned int v38; // eax
  unsigned int *v39; // rdx
  _BYTE *v40; // r9
  unsigned int v41; // esi
  unsigned int v42; // eax
  unsigned int *v43; // rcx
  __int64 v44; // rax
  char *v45; // rax
  __int64 v46; // rcx
  char *v47; // rsi
  __int64 v48; // rdi
  char *v49; // rdx
  __int64 v50; // rax
  char *v51; // rax
  __int64 v52; // rdx
  char *v53; // rsi
  __int64 v54; // rdi
  __int64 v55; // rdx
  char *v56; // rdx
  __int64 v57; // rax
  __int64 v58; // rdx
  __int64 v59; // r8
  int v60; // r9d
  unsigned __int64 v61; // rdx
  __int64 i; // rcx
  __int64 v63; // rdi
  unsigned int v64; // ecx
  unsigned int v65; // esi
  __int64 *v66; // rax
  __int64 v67; // r9
  __int64 v68; // rax
  int v69; // eax
  int v70; // r10d
  __int64 v72; // [rsp+8h] [rbp-48h]
  int v73; // [rsp+10h] [rbp-40h]
  unsigned int v74; // [rsp+10h] [rbp-40h]
  int v75; // [rsp+10h] [rbp-40h]
  int v76; // [rsp+10h] [rbp-40h]
  unsigned __int64 v77; // [rsp+10h] [rbp-40h]
  unsigned __int64 v78; // [rsp+18h] [rbp-38h]
  int v79; // [rsp+18h] [rbp-38h]
  int v80; // [rsp+18h] [rbp-38h]

  v4 = a2;
  sub_1EE7580(a1, *(int **)(a2 + 160), *(unsigned int *)(a2 + 168));
  v6 = *(_QWORD *)(a2 + 80);
  if ( v6 != v6 + 8LL * *(unsigned int *)(a2 + 88) )
  {
    v7 = (unsigned int *)(v6 + 8LL * *(unsigned int *)(a2 + 88));
    v8 = *(unsigned int **)(a2 + 80);
    while ( 1 )
    {
      v9 = *v8;
      v10 = v8[1];
      v11 = *v8;
      if ( (*v8 & 0x80000000) != 0 )
        v11 = *(_DWORD *)(a1 + 192) + (v11 & 0x7FFFFFFF);
      v12 = *(_DWORD **)(a1 + 176);
      v13 = *(unsigned int *)(a1 + 104);
      v14 = *((unsigned __int8 *)v12 + v11);
      if ( v14 >= (unsigned int)v13 )
        goto LABEL_49;
      v15 = *(_QWORD *)(a1 + 96);
      while ( 1 )
      {
        v12 = (_DWORD *)(v15 + 8LL * v14);
        if ( v11 == *v12 )
          break;
        v14 += 256;
        if ( (unsigned int)v13 <= v14 )
          goto LABEL_49;
      }
      if ( v12 == (_DWORD *)(v15 + 8 * v13) )
      {
LABEL_49:
        v16 = 0;
        if ( !(_DWORD)v10 )
          goto LABEL_50;
        v18 = 0;
      }
      else
      {
        v16 = v12[1];
        v12[1] = v16 & ~(_DWORD)v10;
        v17 = v8[1];
        v18 = v16 & ~v17;
        v10 = v17 & ~v16;
        if ( (v17 & ~v16) == 0 )
          goto LABEL_11;
      }
      v75 = v18;
      v79 = v10;
      sub_1EE7570(a1, (v10 << 32) | v9, (__int64)v12, v18, v6, v10);
      sub_1EE5C30((__int64 *)(a1 + 72), *(_QWORD **)(a1 + 24), v9);
      LODWORD(v10) = v79;
      LODWORD(v18) = v75;
      v16 = v79;
LABEL_11:
      if ( (_DWORD)v18 )
        goto LABEL_12;
LABEL_50:
      if ( !*(_BYTE *)(a1 + 58) || !a3 )
      {
        LODWORD(v18) = 0;
        goto LABEL_12;
      }
      v45 = *(char **)a3;
      v46 = *(unsigned int *)(a3 + 8);
      v47 = (char *)(*(_QWORD *)a3 + 8 * v46);
      v48 = (8 * v46) >> 3;
      if ( (8 * v46) >> 5 )
      {
        v49 = &v45[32 * ((8 * v46) >> 5)];
        while ( v9 != *(_DWORD *)v45 )
        {
          if ( v9 == *((_DWORD *)v45 + 2) )
          {
            v45 += 8;
            goto LABEL_60;
          }
          if ( v9 == *((_DWORD *)v45 + 4) )
          {
            v45 += 16;
            goto LABEL_60;
          }
          if ( v9 == *((_DWORD *)v45 + 6) )
          {
            v45 += 24;
            goto LABEL_60;
          }
          v45 += 32;
          if ( v49 == v45 )
          {
            v48 = (v47 - v45) >> 3;
            goto LABEL_96;
          }
        }
        goto LABEL_60;
      }
LABEL_96:
      if ( v48 == 2 )
        goto LABEL_113;
      if ( v48 != 3 )
      {
        if ( v48 != 1 )
          goto LABEL_99;
LABEL_115:
        if ( v9 != *(_DWORD *)v45 )
        {
LABEL_99:
          v68 = v9;
          if ( (unsigned int)v46 >= *(_DWORD *)(a3 + 12) )
          {
            v80 = v16;
            sub_16CD150(a3, (const void *)(a3 + 16), 0, 8, v6, v10);
            v68 = v9;
            v16 = v80;
            v47 = (char *)(*(_QWORD *)a3 + 8LL * *(unsigned int *)(a3 + 8));
          }
          *(_QWORD *)v47 = v68;
          LODWORD(v18) = 0;
          ++*(_DWORD *)(a3 + 8);
          goto LABEL_12;
        }
        goto LABEL_60;
      }
      if ( v9 != *(_DWORD *)v45 )
      {
        v45 += 8;
LABEL_113:
        if ( v9 != *(_DWORD *)v45 )
        {
          v45 += 8;
          goto LABEL_115;
        }
      }
LABEL_60:
      if ( v47 == v45 )
        goto LABEL_99;
      *((_DWORD *)v45 + 1) = 0;
      LODWORD(v18) = 0;
LABEL_12:
      v8 += 2;
      sub_1EE5E20(a1, v9, v16, v18);
      if ( v7 == v8 )
      {
        v4 = a2;
        break;
      }
    }
  }
  v78 = 0;
  if ( *(_BYTE *)(a1 + 56) )
  {
    v61 = *(_QWORD *)(a1 + 64);
    for ( i = *(_QWORD *)(*(_QWORD *)(a1 + 32) + 272LL);
          (*(_BYTE *)(v61 + 46) & 4) != 0;
          v61 = *(_QWORD *)v61 & 0xFFFFFFFFFFFFFFF8LL )
    {
      ;
    }
    v63 = *(_QWORD *)(i + 368);
    v64 = *(_DWORD *)(i + 384);
    if ( v64 )
    {
      v65 = (v64 - 1) & (((unsigned int)v61 >> 9) ^ ((unsigned int)v61 >> 4));
      v66 = (__int64 *)(v63 + 16LL * v65);
      v67 = *v66;
      if ( *v66 == v61 )
      {
LABEL_91:
        v78 = v66[1] & 0xFFFFFFFFFFFFFFF8LL | 4;
        goto LABEL_15;
      }
      v69 = 1;
      while ( v67 != -8 )
      {
        v70 = v69 + 1;
        v65 = (v64 - 1) & (v69 + v65);
        v66 = (__int64 *)(v63 + 16LL * v65);
        v67 = *v66;
        if ( *v66 == v61 )
          goto LABEL_91;
        v69 = v70;
      }
    }
    v78 = *(_QWORD *)(v63 + 16LL * v64 + 8) & 0xFFFFFFFFFFFFFFF8LL | 4;
  }
LABEL_15:
  v19 = *(unsigned int **)v4;
  if ( *(_QWORD *)v4 != *(_QWORD *)v4 + 8LL * *(unsigned int *)(v4 + 8) )
  {
    v72 = v4;
    v20 = *(_QWORD *)v4 + 8LL * *(unsigned int *)(v4 + 8);
    v21 = *(unsigned int **)v4;
    while ( 1 )
    {
      v22 = *v21;
      v23 = v21[1];
      v24 = *v21;
      if ( (*v21 & 0x80000000) != 0 )
        v24 = *(_DWORD *)(a1 + 192) + (v24 & 0x7FFFFFFF);
      v25 = *(_DWORD *)(a1 + 104);
      v26 = v24 | (unsigned __int64)(v23 << 32);
      v27 = (_BYTE *)(*(_QWORD *)(a1 + 176) + v24);
      v28 = (unsigned __int8)*v27;
      if ( v28 >= v25 )
        goto LABEL_62;
      v29 = *(_QWORD *)(a1 + 96);
      while ( 1 )
      {
        v30 = (_DWORD *)(v29 + 8LL * v28);
        if ( v24 == *v30 )
          break;
        v28 += 256;
        if ( v25 <= v28 )
          goto LABEL_62;
      }
      if ( v30 == (_DWORD *)(v29 + 8LL * v25) )
      {
LABEL_62:
        *v27 = v25;
        v50 = *(unsigned int *)(a1 + 104);
        if ( (unsigned int)v50 >= *(_DWORD *)(a1 + 108) )
        {
          v77 = v24 | (unsigned __int64)(v23 << 32);
          sub_16CD150(a1 + 96, (const void *)(a1 + 112), 0, 8, (int)v19, v24);
          v50 = *(unsigned int *)(a1 + 104);
          v26 = v77;
        }
        *(_QWORD *)(*(_QWORD *)(a1 + 96) + 8 * v50) = v26;
        ++*(_DWORD *)(a1 + 104);
        v32 = v21[1];
        if ( !(_DWORD)v32 )
          goto LABEL_27;
        goto LABEL_65;
      }
      v31 = v30[1];
      v30[1] = v31 | v23;
      v32 = v31 | v21[1];
      if ( (_DWORD)v32 != v31 )
        break;
LABEL_27:
      v21 += 2;
      if ( (unsigned int *)v20 == v21 )
      {
        v4 = v72;
        goto LABEL_29;
      }
    }
    if ( v31 )
    {
LABEL_26:
      sub_1EE5D10(a1, v22, v31, v32);
      goto LABEL_27;
    }
LABEL_65:
    if ( !a3 )
    {
LABEL_76:
      v31 = 0;
      if ( *(_BYTE *)(a1 + 56) )
      {
        v74 = v32;
        v57 = sub_1EE8560(a1, v22, v78);
        LODWORD(v32) = v74;
        if ( (_DWORD)v57 )
        {
          sub_1EE7570(a1, (v57 << 32) | v22, v58, v74, v59, v60);
          LODWORD(v32) = v74;
        }
        v31 = 0;
      }
      goto LABEL_26;
    }
    if ( !*(_BYTE *)(a1 + 58) )
      goto LABEL_86;
    v51 = *(char **)a3;
    v52 = 8LL * *(unsigned int *)(a3 + 8);
    v53 = (char *)(*(_QWORD *)a3 + v52);
    v54 = v52 >> 3;
    v55 = v52 >> 5;
    if ( v55 )
    {
      v56 = &v51[32 * v55];
      while ( v22 != *(_DWORD *)v51 )
      {
        if ( v22 == *((_DWORD *)v51 + 2) )
        {
          v51 += 8;
          break;
        }
        if ( v22 == *((_DWORD *)v51 + 4) )
        {
          v51 += 16;
          break;
        }
        if ( v22 == *((_DWORD *)v51 + 6) )
        {
          v51 += 24;
          break;
        }
        v51 += 32;
        if ( v56 == v51 )
        {
          v54 = (v53 - v51) >> 3;
          goto LABEL_83;
        }
      }
LABEL_74:
      if ( v53 != v51 )
      {
        v73 = v32;
        sub_1EE5780(a3, (v32 << 32) | v22);
        LODWORD(v32) = v73;
        goto LABEL_76;
      }
      goto LABEL_86;
    }
LABEL_83:
    if ( v54 != 2 )
    {
      if ( v54 != 3 )
      {
        if ( v54 != 1 )
          goto LABEL_86;
        goto LABEL_109;
      }
      if ( v22 == *(_DWORD *)v51 )
        goto LABEL_74;
      v51 += 8;
    }
    if ( v22 == *(_DWORD *)v51 )
      goto LABEL_74;
    v51 += 8;
LABEL_109:
    if ( v22 == *(_DWORD *)v51 )
      goto LABEL_74;
LABEL_86:
    v76 = v32;
    sub_1EE58A0(a3, (v32 << 32) | v22);
    LODWORD(v32) = v76;
    goto LABEL_76;
  }
LABEL_29:
  if ( *(_BYTE *)(a1 + 57) )
  {
    v33 = *(int **)(v4 + 80);
    for ( j = &v33[2 * *(unsigned int *)(v4 + 88)]; j != v33; v33 += 2 )
    {
      v35 = *v33;
      if ( *v33 < 0 )
      {
        v36 = v35 & 0x7FFFFFFF;
        v37 = *(unsigned int *)(a1 + 104);
        v38 = *(unsigned __int8 *)(*(_QWORD *)(a1 + 176) + (v35 & 0x7FFFFFFFu) + *(_DWORD *)(a1 + 192));
        if ( v38 >= (unsigned int)v37 )
          goto LABEL_40;
        v19 = *(unsigned int **)(a1 + 96);
        while ( 1 )
        {
          v39 = &v19[2 * v38];
          if ( (v35 & 0x7FFFFFFF) + *(_DWORD *)(a1 + 192) == *v39 )
            break;
          v38 += 256;
          if ( (unsigned int)v37 <= v38 )
            goto LABEL_40;
        }
        if ( v39 == &v19[2 * v37] || (v39[1] & v33[1]) == 0 )
        {
LABEL_40:
          v40 = (_BYTE *)(*(_QWORD *)(a1 + 248) + v36);
          v41 = *(_DWORD *)(a1 + 208);
          v42 = (unsigned __int8)*v40;
          if ( v42 >= v41 )
            goto LABEL_45;
          v19 = *(unsigned int **)(a1 + 200);
          while ( 1 )
          {
            v43 = &v19[v42];
            if ( v36 == (*v43 & 0x7FFFFFFF) )
              break;
            v42 += 256;
            if ( v41 <= v42 )
              goto LABEL_45;
          }
          if ( v43 == &v19[v41] )
          {
LABEL_45:
            *v40 = v41;
            v44 = *(unsigned int *)(a1 + 208);
            if ( (unsigned int)v44 >= *(_DWORD *)(a1 + 212) )
            {
              sub_16CD150(a1 + 200, (const void *)(a1 + 216), 0, 4, (int)v19, (int)v40);
              v44 = *(unsigned int *)(a1 + 208);
            }
            *(_DWORD *)(*(_QWORD *)(a1 + 200) + 4 * v44) = v35;
            ++*(_DWORD *)(a1 + 208);
          }
        }
      }
    }
  }
}
