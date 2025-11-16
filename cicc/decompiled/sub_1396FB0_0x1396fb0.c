// Function: sub_1396FB0
// Address: 0x1396fb0
//
void **__fastcall sub_1396FB0(__int64 a1, void ***a2, __int64 a3)
{
  void **result; // rax
  void ***v5; // r13
  void **v6; // rax
  void **v7; // rdx
  void *v8; // r15
  void *v9; // rdi
  bool v10; // si
  __int64 v11; // rax
  size_t v12; // rdx
  size_t v13; // rbx
  const void *v14; // rsi
  size_t v15; // rdx
  const void *v16; // rdi
  void *v17; // rcx
  int v18; // eax
  void *v19; // r15
  __int64 v20; // rdi
  void *v21; // rdx
  const void *v22; // rax
  size_t v23; // rdx
  void *v24; // rcx
  size_t v25; // rbx
  int v26; // eax
  __int64 v27; // rdi
  __int64 v28; // rax
  size_t v29; // rdx
  size_t v30; // rbx
  const void *v31; // rsi
  size_t v32; // rdx
  const void *v33; // rdi
  void *v34; // rcx
  int v35; // eax
  void **v36; // rdx
  void **v37; // rsi
  void ***v38; // rbx
  void ***v39; // r15
  const void *v40; // r13
  size_t v41; // rdx
  size_t v42; // r14
  size_t v43; // rdx
  const void *v44; // rdi
  void *v45; // r8
  int v46; // eax
  void *v47; // r14
  void **v48; // rax
  void *v49; // rdi
  __int64 v50; // rax
  size_t v51; // rdx
  size_t v52; // r13
  const void *v53; // rsi
  size_t v54; // rdx
  const void *v55; // rdi
  size_t v56; // r14
  int v57; // eax
  __int64 v58; // r14
  void **v59; // rax
  __int64 v60; // rbx
  __int64 v61; // r14
  void **v62; // rcx
  void **v63; // rax
  void **v64; // rdx
  void **v65; // rax
  __int64 *v66; // rdx
  void *v67; // r15
  __int64 v68; // rdi
  __int64 v69; // rax
  size_t v70; // rdx
  size_t v71; // rbx
  const void *v72; // rsi
  size_t v73; // rdx
  const void *v74; // rdi
  void *v75; // rcx
  int v76; // eax
  void *v77; // r13
  __int64 v78; // rdi
  __int64 v79; // rax
  size_t v80; // rdx
  size_t v81; // rbx
  const void *v82; // rsi
  size_t v83; // rdx
  const void *v84; // rdi
  size_t v85; // r13
  int v86; // eax
  void **v87; // rax
  void ***v88; // [rsp+8h] [rbp-58h]
  __int64 v89; // [rsp+10h] [rbp-50h]
  void ***v90; // [rsp+18h] [rbp-48h]
  void *v91; // [rsp+20h] [rbp-40h]
  void ***v92; // [rsp+20h] [rbp-40h]
  void *s2b; // [rsp+28h] [rbp-38h]
  void *s2; // [rsp+28h] [rbp-38h]
  void *s2c; // [rsp+28h] [rbp-38h]
  void *s2d; // [rsp+28h] [rbp-38h]
  void *s2a; // [rsp+28h] [rbp-38h]
  void *s2e; // [rsp+28h] [rbp-38h]
  void *s2f; // [rsp+28h] [rbp-38h]
  void *s2g; // [rsp+28h] [rbp-38h]
  void *s2h; // [rsp+28h] [rbp-38h]
  void *s2i; // [rsp+28h] [rbp-38h]

  result = (void **)((char *)a2 - a1);
  v90 = a2;
  v89 = a3;
  if ( (__int64)a2 - a1 > 128 )
  {
    if ( a3 )
    {
      v88 = (void ***)(a1 + 8);
      while ( 1 )
      {
        --v89;
        v5 = (void ***)(a1 + 8 * ((__int64)result >> 4));
        v6 = *(void ***)(a1 + 8);
        v7 = *v5;
        v8 = *v6;
        v9 = **v5;
        v10 = *v6 != 0;
        if ( !v9 )
          goto LABEL_24;
        if ( !*v6 )
          goto LABEL_12;
        v11 = sub_1649960(v9);
        v13 = v12;
        v14 = (const void *)v11;
        v16 = (const void *)sub_1649960(v8);
        v17 = (void *)v15;
        if ( v15 > v13 )
        {
          if ( v13 )
          {
            s2c = (void *)v15;
            v18 = memcmp(v16, v14, v13);
            v17 = s2c;
            if ( v18 )
              goto LABEL_22;
LABEL_10:
            if ( (unsigned __int64)v17 < v13 )
              goto LABEL_11;
          }
        }
        else
        {
          if ( v15 )
          {
            s2b = (void *)v15;
            v18 = memcmp(v16, v14, v15);
            v17 = s2b;
            if ( v18 )
            {
LABEL_22:
              if ( v18 < 0 )
              {
LABEL_11:
                v7 = *v5;
LABEL_12:
                v19 = *v7;
                v20 = (__int64)**(v90 - 1);
                if ( !*v7 )
                {
                  if ( v20 )
                    goto LABEL_77;
                  goto LABEL_31;
                }
                if ( !v20 )
                  goto LABEL_31;
                v91 = (void *)sub_1649960(v20);
                s2 = v21;
                v22 = (const void *)sub_1649960(v19);
                v24 = s2;
                v25 = v23;
                if ( v23 > (unsigned __int64)s2 )
                {
                  if ( !s2 )
                    goto LABEL_92;
                  v26 = memcmp(v22, v91, (size_t)s2);
                  v24 = s2;
                  if ( !v26 )
                    goto LABEL_18;
                }
                else if ( !v23 || (v26 = memcmp(v22, v91, v23), v24 = s2, !v26) )
                {
                  if ( (void *)v25 != v24 )
                  {
LABEL_18:
                    if ( v25 < (unsigned __int64)v24 )
                      goto LABEL_19;
                  }
LABEL_92:
                  v66 = (__int64 *)*(v90 - 1);
                  v6 = *(void ***)(a1 + 8);
                  v77 = *v6;
                  v78 = *v66;
                  if ( !*v6 )
                  {
                    if ( v78 )
                      goto LABEL_100;
                    goto LABEL_32;
                  }
                  if ( v78 )
                  {
                    v79 = sub_1649960(v78);
                    v81 = v80;
                    v82 = (const void *)v79;
                    v84 = (const void *)sub_1649960(v77);
                    v85 = v83;
                    if ( v83 > v81 )
                    {
                      if ( !v81 )
                        goto LABEL_31;
                      v86 = memcmp(v84, v82, v81);
                      if ( !v86 )
                        goto LABEL_98;
                    }
                    else if ( !v83 || (v86 = memcmp(v84, v82, v83)) == 0 )
                    {
                      if ( v85 != v81 )
                      {
LABEL_98:
                        if ( v85 < v81 )
                          goto LABEL_99;
                      }
LABEL_31:
                      v6 = *(void ***)(a1 + 8);
                      goto LABEL_32;
                    }
                    if ( v86 < 0 )
                      goto LABEL_99;
                    goto LABEL_31;
                  }
                  goto LABEL_32;
                }
                if ( v26 < 0 )
                {
LABEL_19:
                  v7 = *v5;
LABEL_77:
                  v65 = *(void ***)a1;
                  *(_QWORD *)a1 = v7;
                  *v5 = v65;
                  v37 = *(void ***)a1;
                  v36 = *(void ***)(a1 + 8);
                  v6 = *(void ***)a1;
                  goto LABEL_33;
                }
                goto LABEL_92;
              }
              goto LABEL_23;
            }
          }
          if ( v17 != (void *)v13 )
            goto LABEL_10;
        }
LABEL_23:
        v6 = *(void ***)(a1 + 8);
        v8 = *v6;
        v10 = *v6 != 0;
LABEL_24:
        v27 = (__int64)**(v90 - 1);
        if ( !v27 )
          goto LABEL_74;
        if ( v10 )
        {
          v28 = sub_1649960(v27);
          v30 = v29;
          v31 = (const void *)v28;
          v33 = (const void *)sub_1649960(v8);
          v34 = (void *)v32;
          if ( v30 < v32 )
          {
            if ( !v30 )
              goto LABEL_81;
            s2g = (void *)v32;
            v35 = memcmp(v33, v31, v30);
            v34 = s2g;
            if ( !v35 )
              goto LABEL_30;
          }
          else if ( !v32 || (s2d = (void *)v32, v35 = memcmp(v33, v31, v32), v34 = s2d, !v35) )
          {
            if ( (void *)v30 == v34 )
              goto LABEL_81;
LABEL_30:
            if ( v30 <= (unsigned __int64)v34 )
              goto LABEL_81;
            goto LABEL_31;
          }
          if ( v35 >= 0 )
          {
LABEL_81:
            v66 = (__int64 *)*(v90 - 1);
            v63 = *v5;
            v67 = **v5;
            v68 = *v66;
            if ( !v67 )
            {
              if ( v68 )
                goto LABEL_100;
              goto LABEL_75;
            }
            if ( v68 )
            {
              v69 = sub_1649960(v68);
              v71 = v70;
              v72 = (const void *)v69;
              v74 = (const void *)sub_1649960(v67);
              v75 = (void *)v73;
              if ( v73 > v71 )
              {
                if ( !v71 )
                  goto LABEL_74;
                s2i = (void *)v73;
                v76 = memcmp(v74, v72, v71);
                v75 = s2i;
                if ( !v76 )
                  goto LABEL_87;
              }
              else if ( !v73 || (s2h = (void *)v73, v76 = memcmp(v74, v72, v73), v75 = s2h, !v76) )
              {
                if ( v75 != (void *)v71 )
                {
LABEL_87:
                  if ( (unsigned __int64)v75 < v71 )
                    goto LABEL_99;
                }
LABEL_74:
                v63 = *v5;
                goto LABEL_75;
              }
              if ( v76 < 0 )
              {
LABEL_99:
                v66 = (__int64 *)*(v90 - 1);
LABEL_100:
                v87 = *(void ***)a1;
                *(_QWORD *)a1 = v66;
                *(v90 - 1) = v87;
                v37 = *(void ***)a1;
                v36 = *(void ***)(a1 + 8);
                v6 = *(void ***)a1;
                goto LABEL_33;
              }
              goto LABEL_74;
            }
LABEL_75:
            v64 = *(void ***)a1;
            *(_QWORD *)a1 = v63;
            *v5 = v64;
            v37 = *(void ***)a1;
            v36 = *(void ***)(a1 + 8);
            v6 = *(void ***)a1;
            goto LABEL_33;
          }
          goto LABEL_31;
        }
LABEL_32:
        v36 = *(void ***)a1;
        *(_QWORD *)a1 = v6;
        v37 = v6;
        *(_QWORD *)(a1 + 8) = v36;
LABEL_33:
        v38 = v88;
        v39 = v90;
        while ( 1 )
        {
          v47 = *v6;
          v92 = v38;
          if ( !*v6 )
            goto LABEL_43;
          s2a = *v36;
          if ( !*v36 )
            break;
          v40 = (const void *)sub_1649960(v47);
          v42 = v41;
          v44 = (const void *)sub_1649960(s2a);
          v45 = (void *)v43;
          if ( v43 > v42 )
          {
            if ( !v42 )
              goto LABEL_50;
            s2f = (void *)v43;
            v46 = memcmp(v44, v40, v42);
            v45 = s2f;
            if ( !v46 )
              goto LABEL_39;
          }
          else if ( !v43 || (s2e = (void *)v43, v46 = memcmp(v44, v40, v43), v45 = s2e, !v46) )
          {
            if ( v45 == (void *)v42 )
              goto LABEL_50;
LABEL_39:
            if ( (unsigned __int64)v45 >= v42 )
              goto LABEL_50;
            goto LABEL_40;
          }
          if ( v46 < 0 )
            goto LABEL_40;
          do
          {
            while ( 1 )
            {
LABEL_50:
              --v39;
              v58 = **(_QWORD **)a1;
              v49 = **v39;
              if ( !v58 )
                goto LABEL_51;
              if ( !v49 )
                goto LABEL_52;
              v50 = sub_1649960(v49);
              v52 = v51;
              v53 = (const void *)v50;
              v55 = (const void *)sub_1649960(v58);
              v56 = v54;
              if ( v52 >= v54 )
                break;
              if ( !v52 )
                goto LABEL_52;
              v57 = memcmp(v55, v53, v52);
              if ( v57 )
                goto LABEL_62;
LABEL_49:
              if ( v52 <= v56 )
                goto LABEL_52;
            }
            if ( !v54 || (v57 = memcmp(v55, v53, v54)) == 0 )
            {
              if ( v52 == v56 )
                goto LABEL_52;
              goto LABEL_49;
            }
LABEL_62:
            ;
          }
          while ( v57 < 0 );
          if ( v38 >= v39 )
            goto LABEL_64;
LABEL_53:
          v59 = *v38;
          *v38 = *v39;
          *v39 = v59;
LABEL_40:
          v37 = *(void ***)a1;
LABEL_41:
          v36 = v38[1];
          v6 = v37;
          ++v38;
        }
        if ( v47 )
          goto LABEL_41;
        do
        {
LABEL_43:
          v48 = *--v39;
          v49 = *v48;
LABEL_51:
          ;
        }
        while ( v49 );
LABEL_52:
        if ( v38 < v39 )
          goto LABEL_53;
LABEL_64:
        sub_1396FB0(v38, v90, v89);
        result = (void **)((char *)v38 - a1);
        if ( (__int64)v38 - a1 <= 128 )
          return result;
        if ( !v89 )
          goto LABEL_56;
        v90 = v38;
      }
    }
    v92 = a2;
LABEL_56:
    v60 = (__int64)result >> 3;
    v61 = (((__int64)result >> 3) - 2) >> 1;
    sub_1396CF0(a1, v61, (__int64)result >> 3, *(void ***)(a1 + 8 * v61));
    do
    {
      --v61;
      sub_1396CF0(a1, v61, v60, *(void ***)(a1 + 8 * v61));
    }
    while ( v61 );
    do
    {
      v62 = *--v92;
      *v92 = *(void ***)a1;
      result = sub_1396CF0(a1, 0, ((__int64)v92 - a1) >> 3, v62);
    }
    while ( (__int64)v92 - a1 > 8 );
  }
  return result;
}
