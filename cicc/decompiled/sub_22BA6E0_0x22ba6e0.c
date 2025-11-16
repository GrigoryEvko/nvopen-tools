// Function: sub_22BA6E0
// Address: 0x22ba6e0
//
void __fastcall sub_22BA6E0(
        unsigned int *a1,
        unsigned int *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7)
{
  _DWORD *v7; // r14
  unsigned int *v8; // r12
  __int64 v9; // rax
  unsigned int *v10; // r13
  __int64 v11; // r15
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // r10
  __int64 v16; // r11
  unsigned int *v17; // rbx
  __int64 v18; // rcx
  __int64 v19; // rbx
  char **v20; // r12
  unsigned __int64 v21; // r15
  __int64 v22; // rdx
  char **v23; // rsi
  __int64 v24; // rdi
  __int64 v25; // rcx
  __int64 v26; // rdx
  __int64 v27; // rax
  char **v28; // rsi
  _DWORD *v29; // rbx
  _DWORD *v30; // r15
  char **v31; // r13
  unsigned int *j; // rbx
  __int64 v33; // rdx
  __int64 v34; // rdi
  __int64 v35; // rdx
  unsigned __int64 v36; // r12
  unsigned int *v37; // rbx
  char **v38; // r15
  unsigned int v39; // eax
  char **v40; // rsi
  __int64 v41; // rdi
  __int64 v42; // rcx
  __int64 v43; // r13
  char **v44; // rbx
  unsigned __int64 v45; // r15
  __int64 v46; // rdx
  char **v47; // rsi
  __int64 v48; // rdi
  __int64 v49; // rcx
  unsigned int *v50; // rbx
  __int64 i; // r15
  char **v52; // rsi
  __int64 v53; // rdx
  __int64 v54; // rcx
  __int64 v55; // r8
  __int64 v56; // r9
  __int64 v57; // r8
  __int64 v58; // r9
  __int64 v59; // rcx
  __int64 v60; // rdx
  char **v61; // r13
  unsigned int *v62; // r12
  unsigned __int64 v63; // rbx
  __int64 v64; // rcx
  char **v65; // rsi
  __int64 v66; // rdi
  __int64 v67; // rax
  __int64 v68; // r8
  __int64 v69; // r9
  __int64 v70; // rdx
  __int64 v71; // r10
  __int64 v72; // rcx
  char **v73; // r12
  unsigned int *v74; // r13
  unsigned __int64 v75; // rbx
  __int64 v76; // rcx
  char **v77; // rsi
  __int64 v78; // rdi
  __int64 v79; // rax
  char **v80; // rax
  unsigned __int64 v81; // r12
  char **v82; // rbx
  __int64 v83; // r13
  __int64 v84; // rdi
  char **v85; // r14
  __int64 v86; // [rsp+0h] [rbp-80h]
  __int64 v87; // [rsp+8h] [rbp-78h]
  __int64 v88; // [rsp+10h] [rbp-70h]
  int v89; // [rsp+10h] [rbp-70h]
  __int64 v90; // [rsp+10h] [rbp-70h]
  int v91; // [rsp+10h] [rbp-70h]
  __int64 v92; // [rsp+10h] [rbp-70h]
  __int64 v93; // [rsp+18h] [rbp-68h]
  __int64 v94; // [rsp+18h] [rbp-68h]
  __int64 v95; // [rsp+18h] [rbp-68h]
  __int64 v96; // [rsp+18h] [rbp-68h]
  __int64 v97; // [rsp+18h] [rbp-68h]
  __int64 v98; // [rsp+20h] [rbp-60h]
  __int64 v99; // [rsp+28h] [rbp-58h]
  __int64 v100; // [rsp+30h] [rbp-50h]
  unsigned int *v101; // [rsp+38h] [rbp-48h]
  unsigned int *v102; // [rsp+38h] [rbp-48h]
  unsigned int *v103; // [rsp+38h] [rbp-48h]
  unsigned int *v104; // [rsp+40h] [rbp-40h]
  __int64 v105; // [rsp+40h] [rbp-40h]
  __int64 v106; // [rsp+40h] [rbp-40h]
  unsigned int *v107; // [rsp+48h] [rbp-38h]

  while ( 1 )
  {
    v7 = (_DWORD *)a6;
    v8 = a1;
    v107 = (unsigned int *)a3;
    v9 = a7;
    if ( a5 <= a7 )
      v9 = a5;
    if ( v9 >= a4 )
    {
      v42 = (char *)a2 - (char *)a1;
      v106 = (char *)a2 - (char *)a1;
      v43 = a6 + 8;
      if ( (char *)a2 - (char *)a1 > 0 )
      {
        v103 = a2;
        v44 = (char **)(a1 + 2);
        v45 = 0x8E38E38E38E38E39LL * (((char *)a2 - (char *)a1) >> 3);
        do
        {
          v46 = *((unsigned int *)v44 - 2);
          v47 = v44;
          v48 = v43;
          v44 += 9;
          v43 += 72;
          *(_DWORD *)(v43 - 80) = v46;
          sub_22AD4A0(v48, v47, v46, v42, a5, a6);
          --v45;
        }
        while ( v45 );
        v49 = v106;
        v50 = v103;
        if ( v106 <= 0 )
          v49 = 72;
        for ( i = (__int64)v7 + v49; v7 != (_DWORD *)i; v8 += 18 )
        {
          if ( v107 == v50 )
          {
            sub_22BA640((__int64)v7, i, (__int64)v8, v49, a5, a6);
            return;
          }
          v53 = *v50;
          if ( (unsigned int)v53 > *v7 )
          {
            *v8 = v53;
            v52 = (char **)(v50 + 2);
            v50 += 18;
          }
          else
          {
            *v8 = *v7;
            v52 = (char **)(v7 + 2);
            v7 += 18;
          }
          sub_22AD4A0((__int64)(v8 + 2), v52, v53, v49, a5, a6);
        }
      }
      return;
    }
    v10 = a2;
    v11 = a5;
    if ( a5 <= a7 )
      break;
    if ( a5 >= a4 )
    {
      v99 = a5 / 2;
      v101 = &a2[18 * (a5 / 2)];
      v104 = sub_22AD8D0(a1, (__int64)a2, v101);
      v100 = 0x8E38E38E38E38E39LL * (((char *)v104 - (char *)a1) >> 3);
    }
    else
    {
      v100 = a4 / 2;
      v104 = &a1[18 * (a4 / 2)];
      v101 = sub_22AD870(a2, a3, v104);
      v99 = 0x8E38E38E38E38E39LL * (((char *)v101 - (char *)a2) >> 3);
    }
    v98 = v16 - v100;
    if ( v16 - v100 <= v99 || v15 < v99 )
    {
      if ( v15 < v98 )
      {
        v97 = v15;
        v80 = sub_22AE020((char **)v104, (char **)a2, (char **)v101, v12, v13, v14);
        v15 = v97;
        v17 = (unsigned int *)v80;
      }
      else
      {
        v17 = v101;
        if ( v98 )
        {
          v88 = v15;
          v94 = sub_22BA640((__int64)v104, (__int64)a2, (__int64)v7, v12, v13, v14);
          sub_22BA640((__int64)a2, (__int64)v101, (__int64)v104, v54, v55, v56);
          v59 = v94;
          v15 = v88;
          v95 = v94 - (_QWORD)v7;
          v60 = 0x8E38E38E38E38E39LL * (v95 >> 3);
          if ( v95 > 0 )
          {
            v86 = v88;
            v61 = (char **)(v59 - 64);
            v89 = (int)a1;
            v62 = v101 - 16;
            v63 = 0x8E38E38E38E38E39LL * (v95 >> 3);
            do
            {
              v64 = *((unsigned int *)v61 - 2);
              v65 = v61;
              v66 = (__int64)v62;
              v61 -= 9;
              v62 -= 18;
              v62[16] = v64;
              sub_22AD4A0(v66, v65, v60, v64, v57, v58);
              --v63;
            }
            while ( v63 );
            LODWORD(v8) = v89;
            v15 = v86;
            v17 = &v101[-2 * (v95 >> 3)];
          }
        }
      }
    }
    else
    {
      v17 = v104;
      if ( v99 )
      {
        v90 = v15;
        v67 = sub_22BA640((__int64)a2, (__int64)v101, (__int64)v7, v12, v13, v14);
        v70 = (char *)a2 - (char *)v104;
        v96 = v67;
        v71 = v90;
        v72 = 0x8E38E38E38E38E39LL * (((char *)a2 - (char *)v104) >> 3);
        if ( (char *)a2 - (char *)v104 > 0 )
        {
          v87 = v90;
          v91 = (int)a1;
          v73 = (char **)(a2 - 16);
          v74 = v101 - 16;
          v75 = 0x8E38E38E38E38E39LL * (v70 >> 3);
          do
          {
            v76 = *((unsigned int *)v73 - 2);
            v77 = v73;
            v78 = (__int64)v74;
            v73 -= 9;
            v74 -= 18;
            v74[16] = v76;
            sub_22AD4A0(v78, v77, v70, v76, v68, v69);
            --v75;
          }
          while ( v75 );
          LODWORD(v8) = v91;
          v71 = v87;
        }
        v92 = v71;
        v79 = sub_22BA640((__int64)v7, v96, (__int64)v104, v72, v68, v69);
        v15 = v92;
        v17 = (unsigned int *)v79;
      }
    }
    v93 = v15;
    sub_22BA6E0((_DWORD)v8, (_DWORD)v104, (_DWORD)v17, v100, v99, (_DWORD)v7, v15);
    a4 = v98;
    a6 = (__int64)v7;
    a2 = v101;
    a5 = v11 - v99;
    a1 = v17;
    a7 = v93;
    a3 = (__int64)v107;
  }
  v18 = a3 - (_QWORD)a2;
  v105 = a3 - (_QWORD)a2;
  if ( a3 - (__int64)a2 <= 0 )
    return;
  v102 = a1;
  v19 = a6 + 8;
  v20 = (char **)(a2 + 2);
  v21 = 0x8E38E38E38E38E39LL * ((a3 - (__int64)a2) >> 3);
  do
  {
    v22 = *((unsigned int *)v20 - 2);
    v23 = v20;
    v24 = v19;
    v20 += 9;
    v19 += 72;
    *(_DWORD *)(v19 - 80) = v22;
    sub_22AD4A0(v24, v23, v22, v18, a5, a6);
    --v21;
  }
  while ( v21 );
  v25 = v105;
  v26 = 8;
  v27 = v105 - 64;
  if ( v105 <= 0 )
    v27 = 8;
  v28 = (char **)((char *)v7 + v27);
  if ( v105 <= 0 )
    v25 = 72;
  v29 = (_DWORD *)((char *)v7 + v25);
  if ( v102 != v10 )
  {
    if ( v7 == v29 )
      return;
    v30 = v29 - 18;
    v31 = (char **)(v10 - 18);
    for ( j = v107 - 18; ; j -= 18 )
    {
      v33 = *(unsigned int *)v31;
      v34 = (__int64)(j + 2);
      if ( *v30 > (unsigned int)v33 )
      {
        *j = v33;
        sub_22AD4A0(v34, v31 + 1, v33, v25, a5, a6);
        if ( v31 == (char **)v102 )
        {
          v35 = 0x8E38E38E38E38E39LL;
          v36 = 0x8E38E38E38E38E39LL * (((char *)(v30 + 18) - (char *)v7) >> 3);
          if ( (char *)(v30 + 18) - (char *)v7 > 0 )
          {
            v37 = j - 16;
            v38 = (char **)(v30 + 2);
            do
            {
              v39 = *((_DWORD *)v38 - 2);
              v40 = v38;
              v41 = (__int64)v37;
              v38 -= 9;
              v37 -= 18;
              v37[16] = v39;
              sub_22AD4A0(v41, v40, v35, v25, a5, a6);
              --v36;
            }
            while ( v36 );
          }
          return;
        }
        v31 -= 9;
      }
      else
      {
        *j = *v30;
        sub_22AD4A0(v34, (char **)v30 + 1, v33, v25, a5, a6);
        if ( v7 == v30 )
          return;
        v30 -= 18;
      }
    }
  }
  v81 = 0x8E38E38E38E38E39LL * (v25 >> 3);
  if ( v25 > 0 )
  {
    v82 = (char **)(v29 - 34);
    v83 = (__int64)(v107 - 16);
    while ( 1 )
    {
      v84 = v83;
      v85 = v82;
      v83 -= 72;
      *(_DWORD *)(v83 + 64) = *((_DWORD *)v82 + 16);
      sub_22AD4A0(v84, v28, v26, v25, a5, a6);
      if ( !--v81 )
        break;
      v82 -= 9;
      v28 = v85;
    }
  }
}
