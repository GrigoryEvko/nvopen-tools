// Function: sub_C8EE80
// Address: 0xc8ee80
//
_BYTE *__fastcall sub_C8EE80(__int64 a1, const char *a2, _QWORD *a3, unsigned __int8 a4, char a5, char a6)
{
  __int64 v7; // r12
  unsigned int v9; // eax
  _BYTE *v10; // rax
  _BYTE *result; // rax
  __int64 v12; // r8
  __int64 v13; // rsi
  char *v14; // rcx
  signed __int64 v15; // rdx
  char *v16; // rax
  unsigned int *v17; // r15
  unsigned int *i; // r13
  unsigned __int64 v19; // rdx
  __int64 v20; // rdi
  size_t v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // rsi
  __int64 v24; // rdx
  signed __int64 v25; // rax
  unsigned __int64 v26; // rsi
  unsigned __int64 v27; // r15
  unsigned __int64 *v28; // r12
  unsigned __int64 *v29; // rbx
  unsigned __int64 v30; // r14
  int v31; // esi
  __int64 v32; // rcx
  unsigned __int64 v33; // r9
  unsigned __int64 v34; // rax
  size_t v35; // rdx
  unsigned int v36; // edx
  size_t v37; // rdx
  unsigned __int64 v38; // rax
  unsigned __int64 v39; // rcx
  size_t v40; // rax
  _QWORD *v41; // r8
  size_t v42; // rdx
  _WORD *v43; // rdi
  unsigned __int64 v44; // rax
  __int64 v45; // rax
  int v46; // r15d
  __int64 v47; // r12
  unsigned __int64 v48; // rbx
  __int64 v49; // r14
  __int64 v50; // rsi
  _BYTE *v51; // rax
  __int64 v52; // rsi
  _BYTE *v53; // rax
  _BYTE *v54; // rax
  __int64 v55; // r14
  unsigned __int64 v56; // r13
  unsigned __int64 v57; // r15
  _QWORD *v58; // rax
  __int64 v59; // rsi
  _BYTE *v60; // rax
  __int64 v61; // rsi
  _BYTE *v62; // rax
  _QWORD *v63; // rdi
  __int64 v64; // rdx
  _BYTE *v65; // rax
  _QWORD *v66; // rdi
  _WORD *v67; // rdx
  unsigned __int64 v68; // r10
  _BYTE *v69; // rax
  unsigned __int64 v70; // [rsp+0h] [rbp-F0h]
  unsigned __int64 v71; // [rsp+8h] [rbp-E8h]
  unsigned __int64 v72; // [rsp+8h] [rbp-E8h]
  __int64 v73; // [rsp+10h] [rbp-E0h]
  __int64 v74; // [rsp+10h] [rbp-E0h]
  unsigned __int64 v75; // [rsp+18h] [rbp-D8h]
  unsigned __int64 v76; // [rsp+18h] [rbp-D8h]
  __int64 v77; // [rsp+28h] [rbp-C8h]
  __int64 v78; // [rsp+30h] [rbp-C0h]
  unsigned int v79; // [rsp+3Ch] [rbp-B4h]
  _QWORD *v80; // [rsp+48h] [rbp-A8h]
  _QWORD *v81; // [rsp+50h] [rbp-A0h]
  size_t v82; // [rsp+50h] [rbp-A0h]
  signed __int64 v84; // [rsp+58h] [rbp-98h]
  unsigned __int64 v86; // [rsp+60h] [rbp-90h]
  int v87; // [rsp+68h] [rbp-88h]
  __int64 v88; // [rsp+68h] [rbp-88h]
  __int64 v89; // [rsp+70h] [rbp-80h] BYREF
  unsigned __int64 v90; // [rsp+78h] [rbp-78h]
  _QWORD *v91; // [rsp+80h] [rbp-70h] BYREF
  unsigned __int64 v92; // [rsp+88h] [rbp-68h]
  _QWORD v93[2]; // [rsp+90h] [rbp-60h] BYREF
  _QWORD *v94; // [rsp+A0h] [rbp-50h] BYREF
  unsigned __int64 v95; // [rsp+A8h] [rbp-48h]
  _QWORD v96[8]; // [rsp+B0h] [rbp-40h] BYREF

  v7 = (__int64)a3;
  v94 = a3;
  v87 = 2 * (a4 == 0);
  LODWORD(v95) = v87;
  sub_CA58F0(&v94, 16, 1, 0);
  if ( a2 && *a2 )
  {
    v81 = v94;
    v40 = strlen(a2);
    v41 = v81;
    v42 = v40;
    v43 = (_WORD *)v81[4];
    v44 = v81[3] - (_QWORD)v43;
    if ( v42 > v44 )
    {
      sub_CB6200(v81, a2, v42);
      v41 = v94;
      v43 = (_WORD *)v94[4];
      v44 = v94[3] - (_QWORD)v43;
    }
    else if ( v42 )
    {
      v80 = v81;
      v82 = v42;
      memcpy(v43, a2, v42);
      v80[4] += v82;
      v41 = v94;
      v43 = (_WORD *)v94[4];
      v44 = v94[3] - (_QWORD)v43;
    }
    if ( v44 <= 1 )
    {
      sub_CB6200(v41, ": ", 2);
    }
    else
    {
      *v43 = 8250;
      v41[4] += 2LL;
    }
  }
  if ( a6 && *(_QWORD *)(a1 + 24) )
  {
    if ( (unsigned int)sub_2241AC0(a1 + 16, "-") )
    {
      sub_CB6200(v94, *(_QWORD *)(a1 + 16), *(_QWORD *)(a1 + 24));
    }
    else
    {
      v63 = v94;
      v64 = v94[4];
      if ( (unsigned __int64)(v94[3] - v64) <= 6 )
      {
        sub_CB6200(v94, "<stdin>", 7);
      }
      else
      {
        *(_DWORD *)v64 = 1685353276;
        *(_WORD *)(v64 + 4) = 28265;
        *(_BYTE *)(v64 + 6) = 62;
        v63[4] += 7LL;
      }
    }
    if ( *(_DWORD *)(a1 + 48) != -1 )
    {
      v65 = (_BYTE *)v94[4];
      if ( (unsigned __int64)v65 >= v94[3] )
      {
        sub_CB5D20(v94, 58);
      }
      else
      {
        v94[4] = v65 + 1;
        *v65 = 58;
      }
      sub_CB59F0(v94, *(int *)(a1 + 48));
      if ( *(_DWORD *)(a1 + 52) != -1 )
      {
        v69 = (_BYTE *)v94[4];
        if ( (unsigned __int64)v69 >= v94[3] )
        {
          sub_CB5D20(v94, 58);
        }
        else
        {
          v94[4] = v69 + 1;
          *v69 = 58;
        }
        sub_CB59F0(v94, *(_DWORD *)(a1 + 52) + 1);
      }
    }
    v66 = v94;
    v67 = (_WORD *)v94[4];
    if ( v94[3] - (_QWORD)v67 <= 1u )
    {
      sub_CB6200(v94, ": ", 2);
    }
    else
    {
      *v67 = 8250;
      v66[4] += 2LL;
    }
  }
  sub_CA5960(&v94);
  if ( a5 )
  {
    v9 = *(_DWORD *)(a1 + 56);
    if ( v9 == 2 )
    {
      sub_CA5D40(v7, byte_3F871B3, 0, a4 ^ 1u);
    }
    else if ( v9 > 2 )
    {
      if ( v9 == 3 )
        sub_CA5BF0(v7, byte_3F871B3, 0, a4 ^ 1u);
    }
    else if ( v9 )
    {
      sub_CA5AA0(v7, byte_3F871B3, 0, a4 ^ 1u);
    }
    else
    {
      sub_CA5970(v7, byte_3F871B3, 0, a4 ^ 1u);
    }
  }
  v94 = (_QWORD *)v7;
  LODWORD(v95) = v87;
  sub_CA58F0(&v94, 16, 1, 0);
  sub_CB6200(v94, *(_QWORD *)(a1 + 64), *(_QWORD *)(a1 + 72));
  v10 = (_BYTE *)v94[4];
  if ( (unsigned __int64)v10 >= v94[3] )
  {
    sub_CB5D20(v94, 10);
  }
  else
  {
    v94[4] = v10 + 1;
    *v10 = 10;
  }
  result = (_BYTE *)sub_CA5960(&v94);
  if ( *(_DWORD *)(a1 + 48) == -1 || *(_DWORD *)(a1 + 52) == -1 )
    return result;
  v13 = *(_QWORD *)(a1 + 96);
  v84 = *(_QWORD *)(a1 + 104);
  v14 = (char *)(v13 + v84);
  v15 = v84;
  if ( v84 >> 2 > 0 )
  {
    v16 = *(char **)(a1 + 96);
    while ( *v16 >= 0 )
    {
      if ( v16[1] < 0 )
      {
        ++v16;
        break;
      }
      if ( v16[2] < 0 )
      {
        v16 += 2;
        break;
      }
      if ( v16[3] < 0 )
      {
        v16 += 3;
        break;
      }
      v16 += 4;
      if ( (char *)(v13 + 4 * (v84 >> 2)) == v16 )
      {
        v15 = v14 - v16;
        goto LABEL_116;
      }
    }
LABEL_21:
    if ( v16 != v14 )
      return (_BYTE *)sub_C8D970(v7, v13, v84);
    goto LABEL_22;
  }
  v16 = *(char **)(a1 + 96);
LABEL_116:
  if ( v15 != 2 )
  {
    if ( v15 != 3 )
    {
      if ( v15 != 1 )
        goto LABEL_22;
      goto LABEL_119;
    }
    if ( *v16 < 0 )
      goto LABEL_21;
    ++v16;
  }
  if ( *v16 < 0 )
    goto LABEL_21;
  ++v16;
LABEL_119:
  if ( *v16 < 0 )
    goto LABEL_21;
LABEL_22:
  v91 = v93;
  sub_2240A50(&v91, v84 + 1, 32, v14, v12);
  v17 = *(unsigned int **)(a1 + 136);
  for ( i = *(unsigned int **)(a1 + 128); v17 != i; i += 2 )
  {
    while ( 1 )
    {
      v19 = i[1];
      v20 = *i;
      if ( v19 > v92 )
        v19 = v92;
      v21 = v19 - v20;
      if ( v21 )
        break;
      i += 2;
      if ( v17 == i )
        goto LABEL_29;
    }
    memset((char *)v91 + v20, 126, v21);
  }
LABEL_29:
  v22 = *(_QWORD *)(a1 + 104);
  v23 = *(_QWORD *)(a1 + 8);
  LOBYTE(v96[0]) = 0;
  v24 = *(unsigned int *)(a1 + 160);
  v94 = v96;
  v25 = *(int *)(a1 + 52);
  v95 = 0;
  if ( v24 )
  {
    v26 = v23 - v25;
    v79 = v22;
    v27 = 0;
    v86 = v26 + v22;
    v78 = a1;
    v77 = v7;
    v28 = *(unsigned __int64 **)(a1 + 152);
    v29 = &v28[6 * v24];
    v30 = v26;
    do
    {
      v38 = v28[3];
      v89 = v28[2];
      v90 = v38;
      if ( sub_C934D0(&v89, "\n\r\t", 3, 0) == -1 )
      {
        v39 = *v28;
        v68 = v28[1];
        if ( v86 >= *v28 && v30 <= v68 )
        {
          if ( v30 <= v39 )
          {
            v31 = v39 - v30;
            v32 = (unsigned int)(v39 - v30);
            v33 = (unsigned int)v32;
          }
          else
          {
            v33 = 0;
            v32 = 0;
            v31 = 0;
          }
          v34 = v33;
          if ( v33 < v27 )
          {
            v34 = (unsigned int)(v27 + 1);
            v31 = v27 + 1;
          }
          v35 = v28[3];
          v27 = (unsigned int)(v31 + v35);
          if ( v27 > v95 )
          {
            v70 = v28[1];
            v72 = v34;
            v74 = v32;
            v76 = v33;
            sub_22410F0(&v94, v27, 32);
            v35 = v28[3];
            v68 = v70;
            v34 = v72;
            v32 = v74;
            v33 = v76;
          }
          if ( v35 )
          {
            v71 = v68;
            v73 = v32;
            v75 = v33;
            memmove((char *)v94 + v34, (const void *)v28[2], v35);
            v68 = v71;
            v32 = v73;
            v33 = v75;
          }
          v36 = v68 - v30;
          if ( v86 <= v68 )
            v36 = v79;
          v37 = v36 - v32;
          if ( v37 )
            memset((char *)v91 + v33, 126, v37);
        }
      }
      v28 += 6;
    }
    while ( v29 != v28 );
    a1 = v78;
    v7 = v77;
    v25 = *(int *)(v78 + 52);
  }
  if ( (unsigned int)v25 > (unsigned __int64)v84 )
    v25 = v84;
  *((_BYTE *)v91 + v25) = 94;
  v45 = sub_2241A80(&v91, 32, -1);
  if ( v45 + 1 > v92 )
    sub_222CF80("%s: __pos (which is %zu) > this->size() (which is %zu)", (char)"basic_string::erase");
  v92 = v45 + 1;
  v46 = 0;
  *((_BYTE *)v91 + v45 + 1) = 0;
  sub_C8D970(v7, *(_QWORD *)(a1 + 96), *(_QWORD *)(a1 + 104));
  v89 = v7;
  LODWORD(v90) = v87;
  sub_CA58F0(&v89, 2, 1, 0);
  if ( (_DWORD)v92 )
  {
    v88 = v7;
    v47 = a1;
    v48 = 0;
    v49 = (unsigned int)v92;
    do
    {
      if ( *(_QWORD *)(v47 + 104) > v48 && *(_BYTE *)(*(_QWORD *)(v47 + 96) + v48) == 9 )
      {
        do
        {
          v52 = *((unsigned __int8 *)v91 + v48);
          v53 = *(_BYTE **)(v89 + 32);
          if ( (unsigned __int64)v53 < *(_QWORD *)(v89 + 24) )
          {
            *(_QWORD *)(v89 + 32) = v53 + 1;
            *v53 = v52;
          }
          else
          {
            sub_CB5D20(v89, v52);
          }
          ++v46;
        }
        while ( (v46 & 7) != 0 );
      }
      else
      {
        v50 = *((unsigned __int8 *)v91 + v48);
        v51 = *(_BYTE **)(v89 + 32);
        if ( (unsigned __int64)v51 < *(_QWORD *)(v89 + 24) )
        {
          *(_QWORD *)(v89 + 32) = v51 + 1;
          *v51 = v50;
        }
        else
        {
          sub_CB5D20(v89, v50);
        }
        ++v46;
      }
      ++v48;
    }
    while ( v49 != v48 );
    a1 = v47;
    v7 = v88;
  }
  v54 = *(_BYTE **)(v89 + 32);
  if ( (unsigned __int64)v54 >= *(_QWORD *)(v89 + 24) )
  {
    sub_CB5D20(v89, 10);
  }
  else
  {
    *(_QWORD *)(v89 + 32) = v54 + 1;
    *v54 = 10;
  }
  v55 = 0;
  v56 = 0;
  result = (_BYTE *)sub_CA5960(&v89);
  v57 = v95;
  if ( v95 )
  {
    do
    {
      v58 = v94;
      if ( *(_QWORD *)(a1 + 104) > v56 && *(_BYTE *)(*(_QWORD *)(a1 + 96) + v56) == 9 )
      {
        do
        {
          v61 = *((unsigned __int8 *)v58 + v56);
          v62 = *(_BYTE **)(v7 + 32);
          if ( (unsigned __int64)v62 < *(_QWORD *)(v7 + 24) )
          {
            *(_QWORD *)(v7 + 32) = v62 + 1;
            *v62 = v61;
          }
          else
          {
            sub_CB5D20(v7, v61);
          }
          v58 = v94;
          ++v55;
          v56 += *((_BYTE *)v94 + v56) != 32;
        }
        while ( (v55 & 7) != 0 && v56 != v57 );
      }
      else
      {
        v59 = *((unsigned __int8 *)v94 + v56);
        v60 = *(_BYTE **)(v7 + 32);
        if ( (unsigned __int64)v60 < *(_QWORD *)(v7 + 24) )
        {
          *(_QWORD *)(v7 + 32) = v60 + 1;
          *v60 = v59;
        }
        else
        {
          sub_CB5D20(v7, v59);
        }
        ++v55;
      }
      ++v56;
    }
    while ( v56 < v57 );
    result = *(_BYTE **)(v7 + 32);
    if ( (unsigned __int64)result >= *(_QWORD *)(v7 + 24) )
    {
      result = (_BYTE *)sub_CB5D20(v7, 10);
    }
    else
    {
      *(_QWORD *)(v7 + 32) = result + 1;
      *result = 10;
    }
  }
  if ( v94 != v96 )
    result = (_BYTE *)j_j___libc_free_0(v94, v96[0] + 1LL);
  if ( v91 != v93 )
    return (_BYTE *)j_j___libc_free_0(v91, v93[0] + 1LL);
  return result;
}
