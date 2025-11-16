// Function: sub_3517D50
// Address: 0x3517d50
//
__int64 *__fastcall sub_3517D50(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // r14
  __int64 v9; // rax
  __int64 v10; // rcx
  unsigned int v11; // edx
  __int64 *v12; // r13
  __int64 v13; // rsi
  __int64 v14; // r15
  __int64 *v15; // rsi
  int v16; // ecx
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 *v19; // rdi
  _QWORD *v20; // rax
  __int64 v21; // r13
  char *v22; // rdi
  __int64 v23; // rax
  char *v24; // rdx
  __int64 v25; // rsi
  char *v26; // rcx
  char **v27; // r15
  int v28; // ecx
  char *v29; // rsi
  int v30; // r8d
  __int64 v31; // r11
  int v32; // r8d
  unsigned int v33; // ecx
  __int64 *v34; // r9
  __int64 v35; // r15
  __int64 v36; // rcx
  __int64 v37; // rdx
  __int64 v38; // rcx
  __int64 v39; // r14
  __int64 *result; // rax
  __int64 v41; // rsi
  unsigned int v42; // edx
  __int64 *v43; // r13
  __int64 v44; // rdi
  __int64 i; // r15
  char *v46; // rdx
  char *v47; // rdi
  __int64 v48; // rax
  __int64 v49; // rcx
  char *v50; // rax
  bool v51; // zf
  __int64 *v52; // rsi
  __int64 *v53; // rdx
  __int64 v54; // rcx
  _QWORD *v55; // rax
  __int64 v56; // r13
  unsigned __int64 v57; // rdx
  unsigned __int64 v58; // rax
  int v59; // r15d
  __int64 v60; // rcx
  __int64 v61; // rax
  __int64 v62; // rdx
  __int64 v63; // r14
  int v64; // ecx
  __int64 v65; // r11
  int v66; // r15d
  unsigned int v67; // ecx
  __int64 *v68; // r8
  __int64 v69; // r9
  __int64 v70; // rcx
  __int64 v71; // rdx
  int v72; // ecx
  __int64 v73; // r11
  int v74; // r14d
  unsigned int v75; // ecx
  __int64 *v76; // r8
  __int64 v77; // r9
  __int64 v78; // rcx
  __int64 v79; // rdx
  int v80; // r9d
  int v81; // r9d
  int v82; // r14d
  int v83; // r8d
  int v84; // r8d
  int v85; // [rsp+8h] [rbp-38h]
  __int64 v86; // [rsp+8h] [rbp-38h]
  int v87; // [rsp+8h] [rbp-38h]
  int v88; // [rsp+8h] [rbp-38h]

  **(_BYTE **)a1 = 1;
  v8 = *(_QWORD *)(a1 + 8);
  v9 = *(unsigned int *)(v8 + 912);
  v10 = *(_QWORD *)(v8 + 896);
  if ( !(_DWORD)v9 )
    goto LABEL_71;
  v11 = (v9 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v12 = (__int64 *)(v10 + 16LL * v11);
  v13 = *v12;
  if ( a2 != *v12 )
  {
    a5 = 1;
    while ( v13 != -4096 )
    {
      a6 = (unsigned int)(a5 + 1);
      v11 = (v9 - 1) & (a5 + v11);
      v12 = (__int64 *)(v10 + 16LL * v11);
      v13 = *v12;
      if ( a2 == *v12 )
        goto LABEL_3;
      a5 = (unsigned int)a6;
    }
    goto LABEL_71;
  }
LABEL_3:
  if ( v12 == (__int64 *)(v10 + 16 * v9) )
  {
LABEL_71:
    v55 = *(_QWORD **)(a1 + 16);
    if ( a2 != *v55 )
    {
      v56 = v8 + 200;
      if ( *(_BYTE *)(a2 + 216) )
      {
LABEL_82:
        v57 = *(unsigned int *)(v8 + 352);
        v58 = *(unsigned int *)(v8 + 208);
        v59 = *(_DWORD *)(v8 + 352);
        if ( v57 <= v58 )
        {
          if ( *(_DWORD *)(v8 + 352) )
            memmove(*(void **)(v8 + 200), *(const void **)(v8 + 344), 8 * v57);
        }
        else
        {
          if ( v57 > *(unsigned int *)(v8 + 212) )
          {
            *(_DWORD *)(v8 + 208) = 0;
            sub_C8D5F0(v56, (const void *)(v8 + 216), v57, 8u, a5, a6);
            v57 = *(unsigned int *)(v8 + 352);
            v60 = 0;
          }
          else
          {
            v60 = 8 * v58;
            if ( *(_DWORD *)(v8 + 208) )
            {
              v86 = 8 * v58;
              memmove(*(void **)(v8 + 200), *(const void **)(v8 + 344), 8 * v58);
              v57 = *(unsigned int *)(v8 + 352);
              v60 = v86;
            }
          }
          v61 = *(_QWORD *)(v8 + 344);
          v62 = 8 * v57;
          if ( v61 + v60 != v62 + v61 )
            memcpy((void *)(v60 + *(_QWORD *)(v8 + 200)), (const void *)(v61 + v60), v62 - v60);
        }
        *(_DWORD *)(v8 + 208) = v59;
      }
LABEL_73:
      sub_3517BE0(v56, a2);
      goto LABEL_9;
    }
    *v55 = *(_QWORD *)(a2 + 8);
LABEL_81:
    v8 = *(_QWORD *)(a1 + 8);
    v56 = v8 + 200;
    if ( *(_BYTE *)(a2 + 216) )
      goto LABEL_82;
    goto LABEL_73;
  }
  v14 = v12[1];
  v15 = *(__int64 **)v14;
  v16 = *(_DWORD *)(v14 + 56);
  a5 = *(unsigned int *)(v14 + 8);
  v17 = *(_QWORD *)v14 + 8 * a5;
  while ( v15 != (__int64 *)v17 )
  {
    v18 = *v15;
    v19 = v15++;
    if ( a2 == v18 )
    {
      if ( v15 != (__int64 *)v17 )
      {
        v85 = *(_DWORD *)(v14 + 56);
        memmove(v19, v15, v17 - (_QWORD)v15);
        LODWORD(a5) = *(_DWORD *)(v14 + 8);
        v16 = v85;
      }
      a5 = (unsigned int)(a5 - 1);
      *(_DWORD *)(v14 + 8) = a5;
      v8 = *(_QWORD *)(a1 + 8);
      break;
    }
  }
  *v12 = -8192;
  --*(_DWORD *)(v8 + 904);
  ++*(_DWORD *)(v8 + 908);
  v20 = *(_QWORD **)(a1 + 16);
  if ( a2 != *v20 )
  {
    if ( v16 )
      goto LABEL_9;
    goto LABEL_81;
  }
  *v20 = *(_QWORD *)(a2 + 8);
  if ( !v16 )
    goto LABEL_81;
LABEL_9:
  v21 = **(_QWORD **)(a1 + 24);
  if ( !v21 )
    goto LABEL_29;
  v22 = *(char **)(v21 + 32);
  v23 = *(unsigned int *)(v21 + 40);
  v24 = &v22[8 * v23];
  v25 = (8 * v23) >> 3;
  if ( (8 * v23) >> 5 )
  {
    v26 = &v22[32 * ((8 * v23) >> 5)];
    while ( a2 != *(_QWORD *)v22 )
    {
      if ( a2 == *((_QWORD *)v22 + 1) )
      {
        v22 += 8;
        if ( v24 != v22 )
          goto LABEL_18;
        goto LABEL_29;
      }
      if ( a2 == *((_QWORD *)v22 + 2) )
      {
        v22 += 16;
        if ( v24 != v22 )
          goto LABEL_18;
        goto LABEL_29;
      }
      if ( a2 == *((_QWORD *)v22 + 3) )
      {
        v22 += 24;
        if ( v24 != v22 )
          goto LABEL_18;
        goto LABEL_29;
      }
      v22 += 32;
      if ( v26 == v22 )
      {
        v25 = (v24 - v22) >> 3;
        goto LABEL_97;
      }
    }
    goto LABEL_17;
  }
LABEL_97:
  if ( v25 == 2 )
    goto LABEL_121;
  if ( v25 == 3 )
  {
    if ( a2 == *(_QWORD *)v22 )
      goto LABEL_17;
    v22 += 8;
LABEL_121:
    if ( a2 == *(_QWORD *)v22 )
      goto LABEL_17;
    v22 += 8;
    goto LABEL_100;
  }
  if ( v25 != 1 )
    goto LABEL_29;
LABEL_100:
  if ( a2 != *(_QWORD *)v22 )
    goto LABEL_29;
LABEL_17:
  if ( v24 == v22 )
    goto LABEL_29;
LABEL_18:
  v27 = *(char ***)(a1 + 32);
  v28 = *(_DWORD *)(v21 + 16);
  v29 = v22 + 8;
  if ( *v27 <= v22 )
  {
    if ( *v27 != v22 )
    {
      if ( v28 )
      {
        v30 = *(_DWORD *)(v21 + 24);
        v31 = *(_QWORD *)(v21 + 8);
        if ( v30 )
        {
          v32 = v30 - 1;
          v33 = v32 & (((unsigned int)*(_QWORD *)v22 >> 9) ^ ((unsigned int)*(_QWORD *)v22 >> 4));
          v34 = (__int64 *)(v31 + 8LL * v33);
          v35 = *v34;
          if ( *(_QWORD *)v22 == *v34 )
          {
LABEL_23:
            *v34 = -8192;
            v36 = *(unsigned int *)(v21 + 40);
            --*(_DWORD *)(v21 + 16);
            v37 = *(_QWORD *)(v21 + 32);
            ++*(_DWORD *)(v21 + 20);
            LODWORD(v23) = v36;
            v24 = (char *)(v37 + 8 * v36);
          }
          else
          {
            v81 = 1;
            while ( v35 != -4096 )
            {
              v82 = v81 + 1;
              v33 = v32 & (v33 + v81);
              v34 = (__int64 *)(v31 + 8LL * v33);
              v35 = *v34;
              if ( *(_QWORD *)v22 == *v34 )
                goto LABEL_23;
              v81 = v82;
            }
          }
        }
        if ( v29 == v24 )
          goto LABEL_28;
      }
      else if ( v24 == v29 )
      {
LABEL_28:
        *(_DWORD *)(v21 + 40) = v23 - 1;
        goto LABEL_29;
      }
      memmove(v22, v29, v24 - v29);
      LODWORD(v23) = *(_DWORD *)(v21 + 40);
      goto LABEL_28;
    }
    if ( v28 )
    {
      v72 = *(_DWORD *)(v21 + 24);
      v73 = *(_QWORD *)(v21 + 8);
      if ( v72 )
      {
        v74 = v72 - 1;
        v75 = (v72 - 1) & (((unsigned int)*(_QWORD *)v22 >> 9) ^ ((unsigned int)*(_QWORD *)v22 >> 4));
        v76 = (__int64 *)(v73 + 8LL * v75);
        v77 = *v76;
        if ( *v76 == *(_QWORD *)v22 )
        {
LABEL_111:
          *v76 = -8192;
          v78 = *(unsigned int *)(v21 + 40);
          --*(_DWORD *)(v21 + 16);
          v79 = *(_QWORD *)(v21 + 32);
          ++*(_DWORD *)(v21 + 20);
          LODWORD(v23) = v78;
          v24 = (char *)(v79 + 8 * v78);
        }
        else
        {
          v83 = 1;
          while ( v77 != -4096 )
          {
            v75 = v74 & (v83 + v75);
            v87 = v83 + 1;
            v76 = (__int64 *)(v73 + 8LL * v75);
            v77 = *v76;
            if ( *(_QWORD *)v22 == *v76 )
              goto LABEL_111;
            v83 = v87;
          }
        }
      }
      if ( v29 == v24 )
        goto LABEL_114;
    }
    else if ( v24 == v29 )
    {
      goto LABEL_114;
    }
    v22 = (char *)memmove(v22, v29, v24 - v29);
    LODWORD(v23) = *(_DWORD *)(v21 + 40);
LABEL_114:
    *(_DWORD *)(v21 + 40) = v23 - 1;
    *v27 = v22;
    goto LABEL_29;
  }
  v63 = *v27 - v22;
  if ( v28 )
  {
    v64 = *(_DWORD *)(v21 + 24);
    v65 = *(_QWORD *)(v21 + 8);
    if ( v64 )
    {
      v66 = v64 - 1;
      v67 = (v64 - 1) & (((unsigned int)*(_QWORD *)v22 >> 9) ^ ((unsigned int)*(_QWORD *)v22 >> 4));
      v68 = (__int64 *)(v65 + 8LL * v67);
      v69 = *v68;
      if ( *v68 == *(_QWORD *)v22 )
      {
LABEL_92:
        *v68 = -8192;
        v70 = *(unsigned int *)(v21 + 40);
        --*(_DWORD *)(v21 + 16);
        v71 = *(_QWORD *)(v21 + 32);
        ++*(_DWORD *)(v21 + 20);
        LODWORD(v23) = v70;
        v24 = (char *)(v71 + 8 * v70);
      }
      else
      {
        v84 = 1;
        while ( v69 != -4096 )
        {
          v67 = v66 & (v84 + v67);
          v88 = v84 + 1;
          v68 = (__int64 *)(v65 + 8LL * v67);
          v69 = *v68;
          if ( *(_QWORD *)v22 == *v68 )
            goto LABEL_92;
          v84 = v88;
        }
      }
    }
  }
  if ( v24 != v29 )
  {
    v22 = (char *)memmove(v22, v29, v24 - v29);
    LODWORD(v23) = *(_DWORD *)(v21 + 40);
  }
  *(_DWORD *)(v21 + 40) = v23 - 1;
  **(_QWORD **)(a1 + 32) = &v22[v63 - 8];
LABEL_29:
  v38 = *(_QWORD *)(a1 + 8);
  v39 = *(_QWORD *)(v38 + 544);
  result = (__int64 *)*(unsigned int *)(v39 + 24);
  v41 = *(_QWORD *)(v39 + 8);
  if ( (_DWORD)result )
  {
    v42 = ((_DWORD)result - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v43 = (__int64 *)(v41 + 16LL * v42);
    v44 = *v43;
    if ( a2 == *v43 )
    {
LABEL_31:
      result = (__int64 *)(v41 + 16LL * (_QWORD)result);
      if ( v43 != result )
      {
        for ( i = v43[1]; i; i = *(_QWORD *)i )
        {
          v46 = *(char **)(i + 40);
          v47 = *(char **)(i + 32);
          v48 = (v46 - v47) >> 5;
          v49 = (v46 - v47) >> 3;
          if ( v48 > 0 )
          {
            v50 = &v47[32 * v48];
            while ( a2 != *(_QWORD *)v47 )
            {
              if ( a2 == *((_QWORD *)v47 + 1) )
              {
                v47 += 8;
                goto LABEL_40;
              }
              if ( a2 == *((_QWORD *)v47 + 2) )
              {
                v47 += 16;
                goto LABEL_40;
              }
              if ( a2 == *((_QWORD *)v47 + 3) )
              {
                v47 += 24;
                goto LABEL_40;
              }
              v47 += 32;
              if ( v50 == v47 )
              {
                v49 = (v46 - v47) >> 3;
                goto LABEL_55;
              }
            }
            goto LABEL_40;
          }
LABEL_55:
          if ( v49 != 2 )
          {
            if ( v49 != 3 )
            {
              if ( v49 == 1 )
                goto LABEL_66;
              v47 = *(char **)(i + 40);
              goto LABEL_40;
            }
            if ( a2 == *(_QWORD *)v47 )
              goto LABEL_40;
            v47 += 8;
          }
          if ( a2 != *(_QWORD *)v47 )
          {
            v47 += 8;
LABEL_66:
            if ( a2 != *(_QWORD *)v47 )
              v47 = *(char **)(i + 40);
          }
LABEL_40:
          if ( v47 + 8 != v46 )
          {
            memmove(v47, v47 + 8, v46 - (v47 + 8));
            v46 = *(char **)(i + 40);
          }
          v51 = *(_BYTE *)(i + 84) == 0;
          *(_QWORD *)(i + 40) = v46 - 8;
          if ( v51 )
          {
            result = sub_C8CA60(i + 56, a2);
            if ( result )
            {
              *result = -2;
              ++*(_DWORD *)(i + 80);
              ++*(_QWORD *)(i + 56);
            }
          }
          else
          {
            v52 = *(__int64 **)(i + 64);
            v53 = &v52[*(unsigned int *)(i + 76)];
            result = v52;
            if ( v52 != v53 )
            {
              while ( a2 != *result )
              {
                if ( v53 == ++result )
                  goto LABEL_48;
              }
              v54 = (unsigned int)(*(_DWORD *)(i + 76) - 1);
              *(_DWORD *)(i + 76) = v54;
              *result = v52[v54];
              ++*(_QWORD *)(i + 56);
            }
          }
LABEL_48:
          ;
        }
        *v43 = -8192;
        --*(_DWORD *)(v39 + 16);
        ++*(_DWORD *)(v39 + 20);
        v38 = *(_QWORD *)(a1 + 8);
      }
    }
    else
    {
      v80 = 1;
      while ( v44 != -4096 )
      {
        v42 = ((_DWORD)result - 1) & (v80 + v42);
        v43 = (__int64 *)(v41 + 16LL * v42);
        v44 = *v43;
        if ( a2 == *v43 )
          goto LABEL_31;
        ++v80;
      }
    }
  }
  if ( a2 == *(_QWORD *)(v38 + 552) )
    *(_QWORD *)(v38 + 552) = 0;
  return result;
}
