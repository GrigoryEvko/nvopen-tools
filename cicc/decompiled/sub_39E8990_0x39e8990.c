// Function: sub_39E8990
// Address: 0x39e8990
//
void __fastcall sub_39E8990(__int64 a1, _QWORD ***a2, __int64 a3, __int64 a4, int a5, int a6)
{
  _QWORD *v7; // rbx
  size_t v8; // r12
  __int64 v9; // r13
  const void *v10; // r14
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rcx
  int v14; // r8d
  int v15; // r9d
  size_t v16; // r14
  _QWORD *v17; // rax
  _BYTE *v18; // rsi
  _BYTE *v19; // rdx
  void *v20; // rdi
  unsigned __int64 v21; // r12
  unsigned __int64 v22; // r12
  unsigned __int64 v23; // r14
  unsigned __int64 i; // rax
  int v25; // r8d
  int v26; // r9d
  size_t v27; // r10
  int v28; // r9d
  __int64 v29; // rdi
  size_t v30; // r14
  void *v31; // r11
  __int64 v32; // rdx
  unsigned __int64 v33; // rbx
  __int64 v34; // rdi
  __int64 v35; // rax
  const void *v36; // r11
  size_t v37; // r9
  size_t v38; // rax
  char *v39; // rdx
  char *v40; // rsi
  const char *v41; // rdi
  size_t v42; // rdx
  __int64 v43; // r14
  __int64 v44; // rcx
  int v45; // r8d
  int v46; // r9d
  char *v47; // rbx
  size_t v48; // r12
  _QWORD *v49; // rax
  __int64 v50; // rdx
  __int64 v51; // rcx
  int v52; // r8d
  int v53; // r9d
  _BYTE *v54; // rdx
  _BYTE *v55; // rsi
  char *v56; // rbx
  __int64 v57; // r13
  char *v58; // rsi
  size_t v59; // rdx
  void *v60; // rdi
  _QWORD *v61; // rdi
  _QWORD *v62; // rdi
  const void *v63; // [rsp+8h] [rbp-98h]
  void *v64; // [rsp+10h] [rbp-90h]
  int v65; // [rsp+10h] [rbp-90h]
  size_t v66; // [rsp+10h] [rbp-90h]
  const void *v67; // [rsp+20h] [rbp-80h]
  __int64 v68; // [rsp+28h] [rbp-78h]
  size_t v69; // [rsp+38h] [rbp-68h] BYREF
  void *v70; // [rsp+40h] [rbp-60h] BYREF
  size_t v71; // [rsp+48h] [rbp-58h]
  void *src; // [rsp+50h] [rbp-50h] BYREF
  size_t n; // [rsp+58h] [rbp-48h]
  _QWORD v74[8]; // [rsp+60h] [rbp-40h] BYREF

  switch ( *((_BYTE *)a2 + 16) )
  {
    case 0:
    case 2:
    case 6:
      v8 = *((unsigned int *)*a2 + 2);
      v7 = **a2;
      goto LABEL_3;
    case 1:
      v9 = *(_QWORD *)(a1 + 280);
      v70 = 0;
      v71 = 0;
      v41 = *(const char **)(v9 + 40);
      if ( !v41 || !strlen(v41) )
        return;
      v8 = 0;
      v7 = 0;
      goto LABEL_51;
    case 3:
      v7 = *a2;
      v8 = 0;
      if ( *a2 )
        v8 = strlen((const char *)*a2);
      goto LABEL_3;
    case 4:
    case 5:
      v7 = **a2;
      v8 = (size_t)(*a2)[1];
LABEL_3:
      v9 = *(_QWORD *)(a1 + 280);
      v70 = v7;
      v71 = v8;
      v10 = *(const void **)(v9 + 40);
      if ( v10 )
      {
        if ( strlen(*(const char **)(v9 + 40)) == v8 && (!v8 || !memcmp(v7, v10, v8)) )
          return;
      }
      else if ( !v8 )
      {
        return;
      }
      if ( v8 <= 1 )
        goto LABEL_51;
      if ( *(_WORD *)v7 == 12079 )
      {
        v11 = *(unsigned int *)(a1 + 312);
        if ( *(_DWORD *)(a1 + 316) == (_DWORD)v11 )
        {
          sub_16CD150(a1 + 304, (const void *)(a1 + 320), v11 + 1, 1, a5, a6);
          v11 = *(unsigned int *)(a1 + 312);
        }
        *(_BYTE *)(*(_QWORD *)(a1 + 304) + v11) = 9;
        v12 = *(_QWORD *)(a1 + 280);
        ++*(_DWORD *)(a1 + 312);
        sub_39E8530(
          a1 + 304,
          *(_BYTE **)(v12 + 48),
          (_BYTE *)(*(_QWORD *)(v12 + 48) + *(_QWORD *)(v12 + 56)),
          a4,
          a5,
          a6);
        if ( v71 <= 1 )
        {
          if ( !((char *)v70 + v71) )
          {
            v19 = v74;
            n = 0;
            src = v74;
            v18 = v74;
            LOBYTE(v74[0]) = 0;
LABEL_13:
            sub_39E8530(a1 + 304, v18, v19, v13, v14, v15);
            v20 = src;
            if ( src == v74 )
              goto LABEL_15;
            goto LABEL_14;
          }
        }
        else if ( v71 != 2 )
        {
          v16 = v71 - 2;
          src = v74;
          v56 = (char *)v70 + 2;
          v69 = v71 - 2;
          if ( v71 - 2 > 0xF )
          {
            src = (void *)sub_22409D0((__int64)&src, &v69, 0);
            v61 = src;
            v74[0] = v69;
          }
          else
          {
            if ( v71 == 3 )
            {
              LOBYTE(v74[0]) = *((_BYTE *)v70 + 2);
              v17 = v74;
              goto LABEL_12;
            }
            v61 = v74;
          }
          memcpy(v61, v56, v16);
          v16 = v69;
          v17 = src;
          goto LABEL_12;
        }
        v16 = 0;
        src = v74;
        v17 = v74;
LABEL_12:
        n = v16;
        *((_BYTE *)v17 + v16) = 0;
        v18 = src;
        v19 = (char *)src + n;
        goto LABEL_13;
      }
      if ( *(_WORD *)v7 == 10799 )
      {
        v22 = v8 - 2;
        v23 = 2;
        v68 = a1 + 304;
        v67 = (const void *)(a1 + 320);
        for ( i = sub_16D23E0(&v70, "\r\n", 2, 2u); ; i = sub_16D23E0(&v70, "\r\n", 2, v23) )
        {
          v32 = *(unsigned int *)(a1 + 312);
          if ( v22 <= i )
            i = v22;
          v33 = i;
          if ( *(_DWORD *)(a1 + 316) == (_DWORD)v32 )
          {
            sub_16CD150(v68, v67, v32 + 1, 1, v25, v26);
            v32 = *(unsigned int *)(a1 + 312);
          }
          *(_BYTE *)(*(_QWORD *)(a1 + 304) + v32) = 9;
          v34 = (unsigned int)(*(_DWORD *)(a1 + 312) + 1);
          v35 = *(_QWORD *)(a1 + 280);
          *(_DWORD *)(a1 + 312) = v34;
          v36 = *(const void **)(v35 + 48);
          v37 = *(_QWORD *)(v35 + 56);
          if ( v37 > (unsigned __int64)*(unsigned int *)(a1 + 316) - v34 )
          {
            v63 = *(const void **)(v35 + 48);
            v66 = *(_QWORD *)(v35 + 56);
            sub_16CD150(v68, v67, v37 + v34, 1, v25, v37);
            v34 = *(unsigned int *)(a1 + 312);
            v36 = v63;
            v37 = v66;
          }
          if ( v37 )
          {
            v65 = v37;
            memcpy((void *)(*(_QWORD *)(a1 + 304) + v34), v36, v37);
            LODWORD(v34) = *(_DWORD *)(a1 + 312);
            LODWORD(v37) = v65;
          }
          v38 = v71;
          v28 = v34 + v37;
          v39 = (char *)v70;
          v27 = v23;
          *(_DWORD *)(a1 + 312) = v28;
          if ( v23 > v38 )
            v27 = v38;
          v40 = &v39[v27];
          if ( &v39[v27] )
          {
            src = v74;
            if ( v33 >= v27 )
              v27 = v33;
            if ( v27 > v38 )
              v27 = v38;
            sub_39DFBE0((__int64 *)&src, v40, (__int64)&v39[v27]);
            v29 = *(unsigned int *)(a1 + 312);
            v30 = n;
            v31 = src;
            if ( n > (unsigned __int64)*(unsigned int *)(a1 + 316) - v29 )
            {
              v64 = src;
              sub_16CD150(v68, v67, n + v29, 1, v25, v28);
              v29 = *(unsigned int *)(a1 + 312);
              v31 = v64;
            }
            if ( v30 )
            {
              memcpy((void *)(*(_QWORD *)(a1 + 304) + v29), v31, v30);
              LODWORD(v29) = *(_DWORD *)(a1 + 312);
            }
          }
          else
          {
            src = v74;
            LODWORD(v29) = v28;
            LODWORD(v30) = 0;
            n = 0;
            LOBYTE(v74[0]) = 0;
          }
          *(_DWORD *)(a1 + 312) = v30 + v29;
          if ( src != v74 )
            j_j___libc_free_0((unsigned __int64)src);
          if ( v22 > v33 )
          {
            v50 = *(unsigned int *)(a1 + 312);
            if ( *(_DWORD *)(a1 + 316) == (_DWORD)v50 )
            {
              sub_16CD150(v68, v67, v50 + 1, 1, v25, v28);
              v50 = *(unsigned int *)(a1 + 312);
            }
            *(_BYTE *)(*(_QWORD *)(a1 + 304) + v50) = 10;
            ++*(_DWORD *)(a1 + 312);
          }
          v23 = v33 + 1;
          if ( v22 <= v33 + 1 )
            break;
        }
        goto LABEL_15;
      }
LABEL_51:
      v42 = *(_QWORD *)(v9 + 56);
      if ( v42 <= v8 && (!v42 || !memcmp(v7, *(const void **)(v9 + 48), v42)) )
      {
        v43 = a1 + 304;
        sub_39E8530(a1 + 304, "\t", "", a4, a5, a6);
        v47 = (char *)v70;
        if ( v70 )
        {
          v48 = v71;
          src = v74;
          v69 = v71;
          if ( v71 <= 0xF )
          {
            if ( v71 == 1 )
            {
              LOBYTE(v74[0]) = *(_BYTE *)v70;
              v49 = v74;
LABEL_83:
              n = v48;
              *((_BYTE *)v49 + v48) = 0;
              v55 = src;
              v54 = (char *)src + n;
LABEL_72:
              sub_39E8530(v43, v55, v54, v44, v45, v46);
              v20 = src;
              if ( src != v74 )
LABEL_14:
                j_j___libc_free_0((unsigned __int64)v20);
LABEL_15:
              if ( *((char *)v70 + v71 - 1) != 10 )
                return;
              goto LABEL_16;
            }
            if ( !v71 )
            {
LABEL_87:
              v49 = v74;
              goto LABEL_83;
            }
LABEL_95:
            v62 = v74;
            goto LABEL_90;
          }
          goto LABEL_89;
        }
LABEL_71:
        v54 = v74;
        n = 0;
        src = v74;
        v55 = v74;
        LOBYTE(v74[0]) = 0;
        goto LABEL_72;
      }
      if ( *(_BYTE *)v7 == 35 )
      {
        v43 = a1 + 304;
        sub_39E8530(a1 + 304, "\t", "", a4, a5, a6);
        sub_39E8530(
          a1 + 304,
          *(_BYTE **)(*(_QWORD *)(a1 + 280) + 48LL),
          (_BYTE *)(*(_QWORD *)(*(_QWORD *)(a1 + 280) + 48LL) + *(_QWORD *)(*(_QWORD *)(a1 + 280) + 56LL)),
          v51,
          v52,
          v53);
        if ( v71 )
        {
          if ( v71 != 1 )
          {
            v48 = v71 - 1;
            src = v74;
            v47 = (char *)v70 + 1;
            v69 = v71 - 1;
            if ( v71 - 1 <= 0xF )
            {
              if ( v71 == 2 )
              {
                LOBYTE(v74[0]) = *((_BYTE *)v70 + 1);
                goto LABEL_87;
              }
              goto LABEL_95;
            }
LABEL_89:
            src = (void *)sub_22409D0((__int64)&src, &v69, 0);
            v62 = src;
            v74[0] = v69;
LABEL_90:
            memcpy(v62, v47, v48);
            v48 = v69;
            v49 = src;
            goto LABEL_83;
          }
        }
        else if ( !v70 )
        {
          goto LABEL_71;
        }
        v48 = 0;
        src = v74;
        v49 = v74;
        goto LABEL_83;
      }
      if ( *((_BYTE *)v7 + v8 - 1) != 10 )
        return;
LABEL_16:
      v21 = *(unsigned int *)(a1 + 312);
      if ( *(_DWORD *)(a1 + 312) )
      {
        v57 = *(_QWORD *)(a1 + 272);
        v58 = *(char **)(a1 + 304);
        v59 = *(unsigned int *)(a1 + 312);
        v60 = *(void **)(v57 + 24);
        if ( v21 > *(_QWORD *)(v57 + 16) - (_QWORD)v60 )
        {
          sub_16E7EE0(*(_QWORD *)(a1 + 272), v58, v59);
        }
        else
        {
          memcpy(v60, v58, v59);
          *(_QWORD *)(v57 + 24) += v21;
        }
      }
      *(_DWORD *)(a1 + 312) = 0;
      return;
  }
}
