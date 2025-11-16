// Function: sub_2F1B010
// Address: 0x2f1b010
//
__int64 __fastcall sub_2F1B010(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  char v3; // al
  unsigned int v4; // r10d
  __int64 v5; // rbx
  __int64 v6; // rcx
  char *v7; // rsi
  __int8 *v8; // rdx
  char *v9; // r15
  char *v10; // r14
  unsigned __int64 *v11; // rbx
  unsigned __int64 *v12; // r13
  int v14; // ebx
  __int64 v15; // rax
  unsigned __int64 v16; // r15
  __int64 v17; // r14
  __int64 v18; // rbx
  char i; // al
  __int64 v20; // rdx
  __int64 v21; // rsi
  unsigned __int64 v22; // rax
  __int64 v23; // r14
  char **v24; // r13
  __int64 v25; // rax
  unsigned __int64 *v26; // r12
  char v27; // al
  unsigned int v28; // r11d
  __int64 v29; // r8
  __int64 v30; // rcx
  char *v31; // rsi
  unsigned __int64 *v32; // r12
  unsigned __int64 *v33; // r13
  unsigned __int8 (__fastcall *v34)(__int64, char *, _BOOL8); // rbx
  char v35; // al
  _BOOL8 v36; // rdx
  unsigned __int8 (__fastcall *v37)(__int64, const char *, _BOOL8); // rbx
  char v38; // al
  _BOOL8 v39; // rdx
  unsigned __int8 (__fastcall *v40)(__int64, const char *, _BOOL8); // rbx
  char v41; // al
  _BOOL8 v42; // rdx
  unsigned __int8 (__fastcall *v43)(__int64, const char *, _BOOL8); // rbx
  char v44; // al
  _BOOL8 v45; // rdx
  unsigned __int8 (__fastcall *v46)(__int64, const char *, _BOOL8); // rbx
  char v47; // al
  _BOOL8 v48; // rdx
  unsigned __int8 (__fastcall *v49)(__int64, char *, _BOOL8); // rbx
  char v50; // al
  _BOOL8 v51; // rdx
  unsigned __int8 (__fastcall *v52)(__int64, const char *, _BOOL8); // rbx
  char v53; // al
  _BOOL8 v54; // rdx
  __int64 v55; // r13
  __int64 v56; // r12
  unsigned __int64 *v57; // rbx
  unsigned __int64 *v58; // r14
  char *v59; // r8
  __int64 v60; // r14
  __int64 v61; // rdx
  unsigned __int64 *v62; // rbx
  unsigned __int64 *v63; // r13
  __int64 v64; // r14
  __int64 v65; // r12
  __int64 v66; // rcx
  char **v67; // r8
  size_t v68; // rdx
  int v69; // eax
  __int64 v70; // r9
  __int64 v71; // r15
  __int64 v72; // rbx
  size_t v73; // rdx
  int v74; // eax
  __int64 v75; // [rsp+8h] [rbp-D8h]
  __int64 v76; // [rsp+10h] [rbp-D0h]
  char **v77; // [rsp+10h] [rbp-D0h]
  __int64 v78; // [rsp+18h] [rbp-C8h]
  __int64 v79; // [rsp+18h] [rbp-C8h]
  unsigned __int64 *v80; // [rsp+20h] [rbp-C0h]
  char *v81; // [rsp+30h] [rbp-B0h]
  __int64 v82; // [rsp+38h] [rbp-A8h]
  __int64 v83; // [rsp+38h] [rbp-A8h]
  __int64 v85; // [rsp+48h] [rbp-98h]
  __int64 v86; // [rsp+48h] [rbp-98h]
  char v87; // [rsp+56h] [rbp-8Ah] BYREF
  char v88; // [rsp+57h] [rbp-89h] BYREF
  char *v89; // [rsp+58h] [rbp-88h] BYREF
  __int64 v90; // [rsp+60h] [rbp-80h] BYREF
  char *v91; // [rsp+68h] [rbp-78h] BYREF
  char *v92; // [rsp+70h] [rbp-70h] BYREF
  char *v93; // [rsp+78h] [rbp-68h]
  __int64 v94; // [rsp+80h] [rbp-60h]
  unsigned __int64 *v95; // [rsp+90h] [rbp-50h] BYREF
  unsigned __int64 *v96; // [rsp+98h] [rbp-48h]
  __int64 v97; // [rsp+A0h] [rbp-40h]

  v2 = a1;
  (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 104LL))(a1);
  if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, __int64, _QWORD, char **, unsigned __int64 **))(*(_QWORD *)a1 + 120LL))(
         a1,
         "kind",
         1,
         0,
         &v92,
         &v95) )
  {
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 160LL))(a1);
    v34 = *(unsigned __int8 (__fastcall **)(__int64, char *, _BOOL8))(*(_QWORD *)a1 + 168LL);
    v35 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    v36 = 0;
    if ( v35 )
      v36 = *(_DWORD *)a2 == 0;
    if ( v34(a1, "block-address", v36) )
      *(_DWORD *)a2 = 0;
    v37 = *(unsigned __int8 (__fastcall **)(__int64, const char *, _BOOL8))(*(_QWORD *)a1 + 168LL);
    v38 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    v39 = 0;
    if ( v38 )
      v39 = *(_DWORD *)a2 == 1;
    if ( v37(a1, "gp-rel64-block-address", v39) )
      *(_DWORD *)a2 = 1;
    v40 = *(unsigned __int8 (__fastcall **)(__int64, const char *, _BOOL8))(*(_QWORD *)a1 + 168LL);
    v41 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    v42 = 0;
    if ( v41 )
      v42 = *(_DWORD *)a2 == 2;
    if ( v40(a1, "gp-rel32-block-address", v42) )
      *(_DWORD *)a2 = 2;
    v43 = *(unsigned __int8 (__fastcall **)(__int64, const char *, _BOOL8))(*(_QWORD *)a1 + 168LL);
    v44 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    v45 = 0;
    if ( v44 )
      v45 = *(_DWORD *)a2 == 3;
    if ( v43(a1, "label-difference32", v45) )
      *(_DWORD *)a2 = 3;
    v46 = *(unsigned __int8 (__fastcall **)(__int64, const char *, _BOOL8))(*(_QWORD *)a1 + 168LL);
    v47 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    v48 = 0;
    if ( v47 )
      v48 = *(_DWORD *)a2 == 4;
    if ( v46(a1, "label-difference64", v48) )
      *(_DWORD *)a2 = 4;
    v49 = *(unsigned __int8 (__fastcall **)(__int64, char *, _BOOL8))(*(_QWORD *)a1 + 168LL);
    v50 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    v51 = 0;
    if ( v50 )
      v51 = *(_DWORD *)a2 == 5;
    if ( v49(a1, "inline", v51) )
      *(_DWORD *)a2 = 5;
    v52 = *(unsigned __int8 (__fastcall **)(__int64, const char *, _BOOL8))(*(_QWORD *)a1 + 168LL);
    v53 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    v54 = 0;
    if ( v53 )
      v54 = *(_DWORD *)a2 == 6;
    if ( v52(a1, "custom32", v54) )
      *(_DWORD *)a2 = 6;
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 184LL))(a1);
    (*(void (__fastcall **)(__int64, unsigned __int64 *))(*(_QWORD *)a1 + 128LL))(a1, v95);
  }
  v92 = 0;
  v93 = 0;
  v94 = 0;
  v80 = (unsigned __int64 *)(a2 + 8);
  v3 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v4 = 0;
  if ( v3 )
  {
    v5 = *(_QWORD *)(a2 + 8);
    v6 = *(_QWORD *)(a2 + 16);
    if ( v6 - v5 == v93 - v92 )
    {
      if ( v5 == v6 )
      {
LABEL_108:
        v4 = 1;
      }
      else
      {
        v59 = v92;
        v60 = *(_QWORD *)(a2 + 8);
        while ( *(_DWORD *)v60 == *(_DWORD *)v59 )
        {
          v70 = *(_QWORD *)(v60 + 32);
          v71 = *(_QWORD *)(v60 + 24);
          v72 = *((_QWORD *)v59 + 3);
          if ( v70 - v71 != *((_QWORD *)v59 + 4) - v72 )
            break;
          for ( ; v70 != v71; v72 += 48 )
          {
            v73 = *(_QWORD *)(v71 + 8);
            if ( v73 != *(_QWORD *)(v72 + 8) )
              goto LABEL_91;
            if ( v73 )
            {
              v81 = v59;
              v83 = v6;
              v86 = v70;
              v74 = memcmp(*(const void **)v71, *(const void **)v72, v73);
              v70 = v86;
              v6 = v83;
              v59 = v81;
              if ( v74 )
                goto LABEL_91;
            }
            v71 += 48;
          }
          v60 += 48;
          v59 += 48;
          if ( v6 == v60 )
            goto LABEL_108;
        }
LABEL_91:
        v4 = 0;
      }
    }
  }
  v7 = "entries";
  if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, _QWORD, _QWORD, char *, char **))(*(_QWORD *)a1 + 120LL))(
         a1,
         "entries",
         0,
         v4,
         &v87,
         &v89) )
  {
    v14 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 24LL))(a1);
    if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
      v14 = -1431655765 * ((__int64)(*(_QWORD *)(a2 + 16) - *(_QWORD *)(a2 + 8)) >> 4);
    if ( v14 )
    {
      v15 = (unsigned int)(v14 - 1);
      v16 = 1;
      v17 = 0;
      v18 = a1;
      v82 = v15 + 2;
      for ( i = (*(__int64 (__fastcall **)(__int64, _QWORD, __int64 *))(*(_QWORD *)v18 + 32LL))(v18, 0, &v90);
            ;
            i = (*(__int64 (__fastcall **)(__int64, _QWORD, __int64 *))(*(_QWORD *)v18 + 32LL))(
                  v18,
                  (unsigned int)(v16 - 1),
                  &v90) )
      {
        v85 = v17 + 48;
        if ( i )
        {
          v20 = *(_QWORD *)(a2 + 8);
          v21 = *(_QWORD *)(a2 + 16);
          v22 = 0xAAAAAAAAAAAAAAABLL * ((v21 - v20) >> 4);
          if ( v22 <= v16 - 1 )
          {
            if ( v22 < v16 )
            {
              sub_2F1AD30((__int64)v80, v16 - v22);
              v20 = *(_QWORD *)(a2 + 8);
            }
            else if ( v22 > v16 )
            {
              v55 = *(_QWORD *)(a2 + 16);
              v56 = v20 + v17 + 48;
              v78 = v56;
              if ( v21 != v56 )
              {
                v76 = v17;
                v75 = v18;
                do
                {
                  v57 = *(unsigned __int64 **)(v56 + 32);
                  v58 = *(unsigned __int64 **)(v56 + 24);
                  if ( v57 != v58 )
                  {
                    do
                    {
                      if ( (unsigned __int64 *)*v58 != v58 + 2 )
                        j_j___libc_free_0(*v58);
                      v58 += 6;
                    }
                    while ( v57 != v58 );
                    v58 = *(unsigned __int64 **)(v56 + 24);
                  }
                  if ( v58 )
                    j_j___libc_free_0((unsigned __int64)v58);
                  v56 += 48;
                }
                while ( v55 != v56 );
                v17 = v76;
                v18 = v75;
                *(_QWORD *)(a2 + 16) = v78;
                v20 = *(_QWORD *)(a2 + 8);
              }
            }
          }
          v23 = v20 + v17;
          v24 = &v91;
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v18 + 104LL))(v18);
          if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, __int64, _QWORD, char **, unsigned __int64 **))(*(_QWORD *)v18 + 120LL))(
                 v18,
                 "id",
                 1,
                 0,
                 &v91,
                 &v95) )
          {
            sub_2F08170(v18, v23);
            (*(void (__fastcall **)(__int64, unsigned __int64 *))(*(_QWORD *)v18 + 128LL))(v18, v95);
          }
          v25 = *(_QWORD *)v18;
          v26 = (unsigned __int64 *)(v23 + 24);
          v95 = 0;
          v96 = 0;
          v97 = 0;
          v27 = (*(__int64 (__fastcall **)(__int64))(v25 + 16))(v18);
          v28 = 0;
          if ( v27 )
          {
            v29 = *(_QWORD *)(v23 + 32);
            v30 = *(_QWORD *)(v23 + 24);
            if ( v29 - v30 == (char *)v96 - (char *)v95 )
            {
              if ( v30 == v29 )
              {
                v28 = 1;
              }
              else
              {
                v61 = v18;
                v62 = v95;
                v63 = (unsigned __int64 *)(v23 + 24);
                v64 = *(_QWORD *)(v23 + 24);
                v65 = v29;
                v66 = v61;
                v67 = &v91;
                do
                {
                  v68 = *(_QWORD *)(v64 + 8);
                  if ( v68 != v62[1]
                    || v68
                    && (v77 = v67,
                        v79 = v66,
                        v69 = memcmp(*(const void **)v64, (const void *)*v62, v68),
                        v66 = v79,
                        v67 = v77,
                        v69) )
                  {
                    v26 = v63;
                    v18 = v66;
                    v24 = v67;
                    v28 = 0;
                    goto LABEL_32;
                  }
                  v64 += 48;
                  v62 += 6;
                }
                while ( v65 != v64 );
                v26 = v63;
                v18 = v66;
                v24 = v67;
                v28 = 1;
              }
            }
          }
LABEL_32:
          v31 = "blocks";
          if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, _QWORD, _QWORD, char *, char **))(*(_QWORD *)v18 + 120LL))(
                 v18,
                 "blocks",
                 0,
                 v28,
                 &v88,
                 v24) )
          {
            sub_2F1A910(v18, v26);
            v31 = v91;
            (*(void (__fastcall **)(__int64, char *))(*(_QWORD *)v18 + 128LL))(v18, v91);
          }
          else if ( v88 )
          {
            v31 = (char *)&v95;
            sub_2F08860((__int64)v26, &v95);
          }
          v32 = v96;
          v33 = v95;
          if ( v96 != v95 )
          {
            do
            {
              if ( (unsigned __int64 *)*v33 != v33 + 2 )
              {
                v31 = (char *)(v33[2] + 1);
                j_j___libc_free_0(*v33);
              }
              v33 += 6;
            }
            while ( v32 != v33 );
            v33 = v95;
          }
          if ( v33 )
          {
            v31 = (char *)(v97 - (_QWORD)v33);
            j_j___libc_free_0((unsigned __int64)v33);
          }
          (*(void (__fastcall **)(__int64, char *))(*(_QWORD *)v18 + 112LL))(v18, v31);
          (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v18 + 40LL))(v18, v90);
        }
        v17 = v85;
        if ( v82 == ++v16 )
          break;
      }
      v2 = v18;
    }
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v2 + 48LL))(v2);
    v7 = v89;
    (*(void (__fastcall **)(__int64, char *))(*(_QWORD *)v2 + 128LL))(v2, v89);
  }
  else if ( v87 )
  {
    v7 = (char *)&v92;
    sub_2F08B30(v80, (unsigned __int64)&v92, v8);
  }
  v9 = v93;
  v10 = v92;
  if ( v93 != v92 )
  {
    do
    {
      v11 = (unsigned __int64 *)*((_QWORD *)v10 + 4);
      v12 = (unsigned __int64 *)*((_QWORD *)v10 + 3);
      if ( v11 != v12 )
      {
        do
        {
          if ( (unsigned __int64 *)*v12 != v12 + 2 )
          {
            v7 = (char *)(v12[2] + 1);
            j_j___libc_free_0(*v12);
          }
          v12 += 6;
        }
        while ( v11 != v12 );
        v12 = (unsigned __int64 *)*((_QWORD *)v10 + 3);
      }
      if ( v12 )
      {
        v7 = (char *)(*((_QWORD *)v10 + 5) - (_QWORD)v12);
        j_j___libc_free_0((unsigned __int64)v12);
      }
      v10 += 48;
    }
    while ( v9 != v10 );
    v10 = v92;
  }
  if ( v10 )
  {
    v7 = (char *)(v94 - (_QWORD)v10);
    j_j___libc_free_0((unsigned __int64)v10);
  }
  return (*(__int64 (__fastcall **)(__int64, char *))(*(_QWORD *)v2 + 112LL))(v2, v7);
}
