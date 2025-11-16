// Function: sub_38E48B0
// Address: 0x38e48b0
//
__int64 __fastcall sub_38E48B0(
        __int64 a1,
        __int64 a2,
        unsigned __int8 *a3,
        size_t a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        unsigned __int64 a8,
        char a9,
        __int64 a10)
{
  size_t v10; // r13
  size_t v12; // rbx
  unsigned __int8 v13; // dl
  size_t v14; // rax
  size_t v15; // r12
  void *v16; // rdi
  char *v17; // r12
  int v18; // r14d
  size_t v19; // rdx
  __int64 v20; // r15
  __int64 v21; // r8
  __int64 v22; // r14
  char *v23; // rsi
  int v24; // r9d
  size_t v25; // r12
  unsigned int v26; // ebx
  unsigned int v27; // ebx
  unsigned int v28; // eax
  int v29; // eax
  unsigned int v30; // ecx
  int **v31; // rax
  int *v32; // r15
  __int64 v33; // rbx
  int *v34; // r14
  int v35; // eax
  char *v36; // rsi
  void *v37; // rdi
  size_t v38; // r12
  size_t v40; // rax
  int v41; // r12d
  int v42; // ebx
  unsigned __int8 *i; // r15
  __int64 v44; // rax
  size_t v45; // r14
  _BYTE *v46; // rax
  __int64 v47; // r14
  void *v48; // rdi
  _QWORD *v49; // rsi
  char *v50; // rsi
  size_t v51; // r9
  unsigned __int64 v52; // rax
  const char *v53; // rdx
  size_t v54; // rsi
  size_t v55; // r12
  __int64 v56; // r15
  size_t v57; // rbx
  const char *v58; // rax
  char v59; // r14
  unsigned __int64 v60; // rcx
  unsigned __int64 v61; // r13
  unsigned __int64 v62; // r12
  unsigned __int64 v63; // rax
  int v64; // eax
  unsigned __int64 v65; // rax
  __int64 v66; // rax
  _BYTE *v67; // rax
  __int64 *v68; // rax
  __int64 v69; // r12
  unsigned __int8 *v70; // r15
  __int64 v71; // rbx
  void *v72; // rdi
  unsigned __int64 v73; // r14
  void *v74; // rsi
  size_t v75; // rdx
  __int64 v76; // [rsp+0h] [rbp-D0h]
  int *v77; // [rsp+8h] [rbp-C8h]
  int *v78; // [rsp+10h] [rbp-C0h]
  size_t v80; // [rsp+38h] [rbp-98h]
  const char *v81; // [rsp+38h] [rbp-98h]
  int v82; // [rsp+40h] [rbp-90h]
  unsigned __int8 *v83; // [rsp+48h] [rbp-88h]
  unsigned __int8 *v84; // [rsp+48h] [rbp-88h]
  int v85; // [rsp+50h] [rbp-80h]
  char v86; // [rsp+55h] [rbp-7Bh]
  bool v87; // [rsp+57h] [rbp-79h]
  size_t v88; // [rsp+58h] [rbp-78h]
  size_t v89; // [rsp+58h] [rbp-78h]
  unsigned __int8 *v90; // [rsp+58h] [rbp-78h]
  unsigned __int8 *v91; // [rsp+58h] [rbp-78h]
  __int64 v92; // [rsp+60h] [rbp-70h]
  bool v93; // [rsp+60h] [rbp-70h]
  char *v94; // [rsp+60h] [rbp-70h]
  unsigned __int8 *v95; // [rsp+60h] [rbp-70h]
  size_t v96; // [rsp+60h] [rbp-70h]
  unsigned __int8 *v97; // [rsp+60h] [rbp-70h]
  size_t v98; // [rsp+60h] [rbp-70h]
  size_t v99; // [rsp+60h] [rbp-70h]
  unsigned __int8 *v101; // [rsp+70h] [rbp-60h]
  size_t v102; // [rsp+70h] [rbp-60h]
  unsigned __int8 *v103; // [rsp+70h] [rbp-60h]
  unsigned __int8 *v104; // [rsp+70h] [rbp-60h]
  const char *v106; // [rsp+80h] [rbp-50h] BYREF
  size_t v107; // [rsp+88h] [rbp-48h]
  _QWORD v108[8]; // [rsp+90h] [rbp-40h] BYREF

  v85 = a6;
  if ( (_DWORD)a6 )
  {
    v86 = *(_BYTE *)(48 * a6 + a5 - 7);
  }
  else
  {
    v86 = *(_BYTE *)(a1 + 844);
    if ( v86 )
    {
      v86 = 0;
      goto LABEL_4;
    }
  }
  if ( a8 != (unsigned int)a6 )
  {
    v106 = "Wrong number of arguments";
    LOWORD(v108[0]) = 259;
    return sub_3909790(a1, a10, &v106, 0, 0);
  }
LABEL_4:
  v10 = a4;
  v87 = (_DWORD)a6 == 0;
  if ( a4 )
  {
    while ( 1 )
    {
      v12 = 0;
      v13 = *a3;
      while ( 1 )
      {
        v14 = v12 + 1;
        if ( (*(_BYTE *)(a1 + 844) & v87) == 0 )
          break;
        if ( v13 != 36 )
          goto LABEL_6;
        if ( v14 == v10 )
          goto LABEL_12;
        v13 = a3[v14];
        if ( v13 == 36 || v13 == 110 || (unsigned int)v13 - 48 <= 9 )
          goto LABEL_55;
LABEL_8:
        ++v12;
      }
      if ( v13 != 92 )
      {
LABEL_6:
        if ( v14 == v10 )
          goto LABEL_12;
        v13 = a3[v14];
        goto LABEL_8;
      }
      if ( v14 == v10 )
      {
LABEL_12:
        v15 = v10;
        v12 = v10;
        goto LABEL_13;
      }
LABEL_55:
      if ( v12 )
      {
        v15 = v10;
        if ( v12 <= v10 )
          v15 = v12;
LABEL_13:
        v16 = *(void **)(a2 + 24);
        v101 = a3;
        if ( v15 <= *(_QWORD *)(a2 + 16) - (_QWORD)v16 )
        {
          memcpy(v16, a3, v15);
          a3 = v101;
          *(_QWORD *)(a2 + 24) += v15;
        }
        else
        {
          sub_16E7EE0(a2, (char *)a3, v15);
          a3 = v101;
        }
        if ( v12 == v10 )
          return 0;
        v14 = v12 + 1;
      }
      v17 = (char *)&a3[v14];
      if ( *(_BYTE *)(a1 + 844) && v87 )
      {
        v64 = *v17;
        if ( *v17 != 36 )
        {
          if ( (_BYTE)v64 == 110 )
          {
            v103 = a3;
            sub_16E7A90(a2, a8);
            a3 = v103;
            v102 = v12 + 2;
            goto LABEL_47;
          }
          v65 = (unsigned int)(v64 - 48);
          if ( v65 >= a8 || (v68 = (__int64 *)(a7 + 24 * v65), v69 = *v68, v68[1] == *v68) )
          {
            v102 = v12 + 2;
            goto LABEL_47;
          }
          v70 = a3;
          v99 = v12;
          v71 = v68[1];
          while ( 1 )
          {
            v72 = *(void **)(a2 + 24);
            v73 = *(_QWORD *)(v69 + 16);
            v74 = *(void **)(v69 + 8);
            if ( v73 <= *(_QWORD *)(a2 + 16) - (_QWORD)v72 )
            {
              if ( v73 )
              {
                memcpy(v72, v74, *(_QWORD *)(v69 + 16));
                *(_QWORD *)(a2 + 24) += v73;
              }
              v69 += 40;
              if ( v71 == v69 )
              {
LABEL_128:
                a3 = v70;
                v102 = v99 + 2;
                goto LABEL_47;
              }
            }
            else
            {
              v75 = *(_QWORD *)(v69 + 16);
              v69 += 40;
              sub_16E7EE0(a2, (char *)v74, v75);
              if ( v71 == v69 )
                goto LABEL_128;
            }
          }
        }
        v67 = *(_BYTE **)(a2 + 24);
        if ( (unsigned __int64)v67 >= *(_QWORD *)(a2 + 16) )
        {
          v104 = a3;
          sub_16E7DE0(a2, 36);
          a3 = v104;
        }
        else
        {
          *(_QWORD *)(a2 + 24) = v67 + 1;
          *v67 = 36;
        }
        v102 = v12 + 2;
      }
      else
      {
        v18 = a3[(unsigned int)v14];
        if ( a9 && (_BYTE)v18 == 64 && (v102 = (unsigned int)(v12 + 2), v102 != v10) )
        {
          v19 = v102 + ~v12;
          if ( v19 != 1 )
            goto LABEL_23;
LABEL_66:
          if ( *v17 == 64 )
          {
            v95 = a3;
            sub_16E7A90(a2, *(unsigned int *)(a1 + 556));
            a3 = v95;
            v102 = v12 + 2;
            goto LABEL_47;
          }
        }
        else
        {
          v94 = (char *)&a3[v14];
          v40 = (unsigned int)(v12 + 1);
          v41 = v18;
          v89 = v12;
          v42 = v12 + 1;
          for ( i = a3; ; v41 = i[v42] )
          {
            v45 = v40;
            if ( !isalnum((unsigned __int8)v41) )
            {
              if ( (unsigned __int8)(v41 - 36) > 0x3Bu )
                break;
              v44 = 0x800000000000401LL;
              if ( !_bittest64(&v44, (unsigned int)(v41 - 36)) )
                break;
            }
            v40 = (unsigned int)++v42;
            if ( v42 == v10 )
              break;
          }
          v12 = v89;
          v17 = v94;
          v102 = v45;
          a3 = i;
          v19 = v45 + ~v89;
          if ( v19 == 1 )
            goto LABEL_66;
        }
LABEL_23:
        if ( !v85 )
          goto LABEL_70;
        v20 = a5;
        v21 = (unsigned int)(v85 - 1);
        v22 = 0;
        v23 = v17;
        v80 = v12;
        v24 = v85 - 1;
        v25 = v19;
        v26 = 0;
        if ( v19 != *(_QWORD *)(a5 + 8) )
          goto LABEL_25;
LABEL_27:
        v92 = v21;
        if ( !v25
          || (v82 = v24, v83 = a3, v29 = memcmp(*(const void **)v20, v23, v25), a3 = v83, v24 = v82, v21 = v92, !v29) )
        {
          v30 = v26;
          goto LABEL_30;
        }
LABEL_25:
        while ( 1 )
        {
          v27 = v26 + 1;
          v28 = v22 + 1;
          v20 += 48;
          if ( v21 == v22 )
            break;
          ++v22;
          v26 = v28;
          if ( v25 == *(_QWORD *)(v20 + 8) )
            goto LABEL_27;
        }
        v19 = v25;
        v30 = v27;
        v17 = v23;
        v12 = v80;
        if ( v85 == v30 )
        {
LABEL_70:
          if ( *v17 == 40 && a3[v12 + 2] == 41 )
          {
            v102 = v12 + 3;
          }
          else
          {
            v46 = *(_BYTE **)(a2 + 24);
            if ( (unsigned __int64)v46 >= *(_QWORD *)(a2 + 16) )
            {
              v91 = a3;
              v98 = v19;
              v66 = sub_16E7DE0(a2, 92);
              v19 = v98;
              a3 = v91;
              v47 = v66;
            }
            else
            {
              *(_QWORD *)(a2 + 24) = v46 + 1;
              v47 = a2;
              *v46 = 92;
            }
            v48 = *(void **)(v47 + 24);
            if ( v19 > *(_QWORD *)(v47 + 16) - (_QWORD)v48 )
            {
              v97 = a3;
              sub_16E7EE0(v47, v17, v19);
              a3 = v97;
            }
            else if ( v19 )
            {
              v90 = a3;
              v96 = v19;
              memcpy(v48, v17, v19);
              a3 = v90;
              *(_QWORD *)(v47 + 24) += v96;
            }
          }
          goto LABEL_47;
        }
        v22 = v30;
LABEL_30:
        v93 = 0;
        if ( v86 )
          v93 = v30 == v24;
        v31 = (int **)(a7 + 24 * v22);
        v32 = v31[1];
        if ( v32 != *v31 )
        {
          v88 = v10;
          v33 = a2;
          v34 = *v31;
          v84 = a3;
          while ( 1 )
          {
            v35 = *v34;
            if ( !*(_BYTE *)(a1 + 272) )
              goto LABEL_40;
            v36 = (char *)*((_QWORD *)v34 + 1);
            if ( *v36 == 37 )
            {
              if ( v35 == 4 )
              {
                v49 = (_QWORD *)*((_QWORD *)v34 + 3);
                if ( (unsigned int)v34[8] > 0x40 )
                  v49 = (_QWORD *)*v49;
                sub_16E7AB0(v33, (__int64)v49);
                goto LABEL_36;
              }
LABEL_40:
              if ( v35 != 3 || v93 )
              {
                v36 = (char *)*((_QWORD *)v34 + 1);
                goto LABEL_43;
              }
              v62 = *((_QWORD *)v34 + 2);
              if ( v62 )
              {
                v63 = v62 - 1;
                if ( v62 == 1 )
                  v63 = 1;
                v37 = *(void **)(v33 + 24);
                if ( v63 > v62 )
                  v63 = *((_QWORD *)v34 + 2);
                v38 = v63 - 1;
                v36 = (char *)(*((_QWORD *)v34 + 1) + 1LL);
                if ( *(_QWORD *)(v33 + 16) - (_QWORD)v37 < v63 - 1 )
                {
                  sub_16E7EE0(v33, v36, v38);
                  goto LABEL_45;
                }
LABEL_34:
                if ( v38 )
                {
                  memcpy(v37, v36, v38);
                  *(_QWORD *)(v33 + 24) += v38;
                }
              }
LABEL_36:
              v34 += 10;
              if ( v32 == v34 )
                goto LABEL_46;
            }
            else
            {
              if ( *v36 != 60 )
                goto LABEL_40;
              if ( v35 == 3 )
              {
                v107 = 0;
                v106 = (const char *)v108;
                v50 = (char *)v108;
                LOBYTE(v108[0]) = 0;
                v51 = *((_QWORD *)v34 + 2);
                v52 = v51 - 1;
                if ( v51 )
                {
                  if ( v51 == 1 )
                    v52 = 1;
                  if ( v52 <= v51 )
                    v51 = v52;
                  if ( --v51 )
                  {
                    v77 = v32;
                    v53 = (const char *)v108;
                    v54 = 0;
                    v55 = 0;
                    v76 = v33;
                    v56 = *((_QWORD *)v34 + 1) + 1LL;
                    v57 = v51;
                    v78 = v34;
                    v58 = (const char *)v108;
                    while ( 1 )
                    {
                      v59 = *(_BYTE *)(v56 + v55);
                      if ( v59 == 33 )
                      {
                        ++v55;
                        v59 = *(_BYTE *)(v56 + v55);
                      }
                      v60 = 15;
                      v61 = v54 + 1;
                      if ( v53 != v58 )
                        v60 = v108[0];
                      if ( v61 > v60 )
                      {
                        v81 = v58;
                        sub_2240BB0((unsigned __int64 *)&v106, v54, 0, 0, 1u);
                        v53 = v106;
                        v58 = v81;
                      }
                      v53[v54] = v59;
                      ++v55;
                      v107 = v54 + 1;
                      v106[v61] = 0;
                      if ( v55 >= v57 )
                        break;
                      v54 = v107;
                      v53 = v106;
                    }
                    v34 = v78;
                    v32 = v77;
                    v33 = v76;
                    v51 = v107;
                    v50 = (char *)v106;
                  }
                }
                sub_16E7EE0(v33, v50, v51);
                if ( v106 != (const char *)v108 )
                  j_j___libc_free_0((unsigned __int64)v106);
                goto LABEL_36;
              }
LABEL_43:
              v37 = *(void **)(v33 + 24);
              v38 = *((_QWORD *)v34 + 2);
              if ( v38 <= *(_QWORD *)(v33 + 16) - (_QWORD)v37 )
                goto LABEL_34;
              sub_16E7EE0(v33, v36, *((_QWORD *)v34 + 2));
LABEL_45:
              v34 += 10;
              if ( v32 == v34 )
              {
LABEL_46:
                v10 = v88;
                a3 = v84;
                break;
              }
            }
          }
        }
      }
LABEL_47:
      if ( v102 <= v10 )
      {
        a3 += v102;
        v10 -= v102;
        if ( v10 )
          continue;
      }
      return 0;
    }
  }
  return 0;
}
