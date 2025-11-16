// Function: sub_D0F410
// Address: 0xd0f410
//
__int64 __fastcall sub_D0F410(char *a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 result; // rax
  __int64 *v7; // rcx
  char *v8; // r12
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // r14
  __int64 v12; // rdi
  bool v13; // si
  const char *v14; // rax
  size_t v15; // rdx
  size_t v16; // rbx
  const char *v17; // r15
  const char *v18; // rax
  size_t v19; // rdx
  size_t v20; // r14
  bool v21; // cc
  size_t v22; // rdx
  int v23; // eax
  __int64 v24; // rbx
  __int64 v25; // rdi
  const char *v26; // rax
  size_t v27; // rdx
  size_t v28; // r15
  const char *v29; // r14
  const char *v30; // rax
  size_t v31; // rdx
  size_t v32; // rbx
  size_t v33; // rdx
  int v34; // eax
  __int64 v35; // rdi
  const char *v36; // rax
  size_t v37; // rdx
  size_t v38; // r15
  const char *v39; // rax
  size_t v40; // rdx
  size_t v41; // rbx
  size_t v42; // rdx
  int v43; // eax
  __int64 v44; // rdx
  __int64 v45; // rax
  __int64 v46; // rdi
  __int64 v47; // r14
  const char *v48; // rax
  size_t v49; // rdx
  size_t v50; // rbx
  const char *v51; // r15
  const char *v52; // rax
  size_t v53; // rdx
  size_t v54; // r14
  size_t v55; // rdx
  int v56; // eax
  __int64 v57; // rdx
  __int64 v58; // rsi
  __int64 v59; // rdx
  __int64 v60; // rbx
  __int64 v61; // r12
  __int64 v62; // rcx
  __int64 *v63; // r12
  __int64 *v64; // r15
  const char *v65; // rax
  size_t v66; // rdx
  size_t v67; // rbx
  const char *v68; // rax
  __int64 v69; // r9
  size_t v70; // rdx
  size_t v71; // r14
  size_t v72; // rdx
  int v73; // eax
  __int64 v74; // r14
  __int64 v75; // rax
  __int64 v76; // rdi
  const char *v77; // rax
  size_t v78; // rdx
  size_t v79; // rbx
  const char *v80; // rax
  size_t v81; // rdx
  size_t v82; // r14
  int v83; // eax
  __int64 v84; // rax
  __int64 i; // r14
  __int64 v86; // rax
  __int64 v87; // rax
  __int64 v88; // rdi
  __int64 v89; // r12
  const char *v90; // rax
  size_t v91; // rdx
  size_t v92; // rbx
  const char *v93; // r14
  const char *v94; // rax
  size_t v95; // rdx
  size_t v96; // r12
  size_t v97; // rdx
  int v98; // eax
  __int64 v99; // rax
  __int64 *v100; // [rsp+0h] [rbp-60h]
  __int64 v101; // [rsp+8h] [rbp-58h]
  __int64 *v102; // [rsp+10h] [rbp-50h]
  const char *v103; // [rsp+18h] [rbp-48h]
  __int64 *v104; // [rsp+20h] [rbp-40h]
  const char *s2; // [rsp+28h] [rbp-38h]
  void *s2a; // [rsp+28h] [rbp-38h]
  const char *s2b; // [rsp+28h] [rbp-38h]

  result = (char *)a2 - a1;
  v102 = a2;
  v101 = a3;
  if ( (char *)a2 - a1 > 128 )
  {
    if ( a3 )
    {
      v7 = (__int64 *)(a1 + 8);
      v100 = (__int64 *)(a1 + 8);
      while ( 1 )
      {
        --v101;
        v8 = &a1[8 * (result >> 4)];
        v9 = *((_QWORD *)a1 + 1);
        v10 = *(_QWORD *)v8;
        v11 = *(_QWORD *)(v9 + 8);
        v12 = *(_QWORD *)(*(_QWORD *)v8 + 8LL);
        LOBYTE(v7) = v11 != 0;
        v13 = v11 != 0;
        if ( v12 )
        {
          if ( !v11 )
            goto LABEL_13;
          v14 = sub_BD5D20(v12);
          v16 = v15;
          v17 = v14;
          v18 = sub_BD5D20(v11);
          v20 = v19;
          v21 = v19 <= v16;
          v22 = v16;
          if ( v21 )
            v22 = v20;
          if ( v22 && (v23 = memcmp(v18, v17, v22)) != 0 )
          {
            if ( v23 < 0 )
              goto LABEL_12;
          }
          else if ( v20 != v16 && v20 < v16 )
          {
LABEL_12:
            v10 = *(_QWORD *)v8;
LABEL_13:
            v24 = *(_QWORD *)(v10 + 8);
            v25 = *(_QWORD *)(*(v102 - 1) + 8);
            if ( v25 )
            {
              if ( !v24 )
                goto LABEL_81;
              v26 = sub_BD5D20(v25);
              v28 = v27;
              v29 = v26;
              v30 = sub_BD5D20(v24);
              v32 = v31;
              v21 = v31 <= v28;
              v33 = v28;
              if ( v21 )
                v33 = v32;
              if ( v33 && (v34 = memcmp(v30, v29, v33)) != 0 )
              {
                if ( v34 < 0 )
                  goto LABEL_21;
              }
              else if ( v32 != v28 && v32 < v28 )
              {
LABEL_21:
                v10 = *(_QWORD *)v8;
LABEL_81:
                v87 = *(_QWORD *)a1;
                *(_QWORD *)a1 = v10;
                *(_QWORD *)v8 = v87;
                v58 = *(_QWORD *)a1;
                v59 = *((_QWORD *)a1 + 1);
                v9 = *(_QWORD *)a1;
                goto LABEL_49;
              }
              v44 = *(v102 - 1);
              v9 = *((_QWORD *)a1 + 1);
              v88 = *(_QWORD *)(v44 + 8);
              v89 = *(_QWORD *)(v9 + 8);
              if ( v88 )
              {
                if ( !v89 )
                  goto LABEL_94;
                v90 = sub_BD5D20(v88);
                v92 = v91;
                v93 = v90;
                v94 = sub_BD5D20(v89);
                v96 = v95;
                v21 = v95 <= v92;
                v97 = v92;
                if ( v21 )
                  v97 = v96;
                if ( v97 && (v98 = memcmp(v94, v93, v97)) != 0 )
                {
                  if ( v98 < 0 )
                    goto LABEL_93;
                }
                else if ( v96 != v92 && v96 < v92 )
                {
                  goto LABEL_93;
                }
                goto LABEL_83;
              }
            }
            else
            {
LABEL_83:
              v9 = *((_QWORD *)a1 + 1);
            }
LABEL_48:
            v59 = *(_QWORD *)a1;
            *(_QWORD *)a1 = v9;
            v58 = v9;
            *((_QWORD *)a1 + 1) = v59;
            goto LABEL_49;
          }
          v9 = *((_QWORD *)a1 + 1);
          v11 = *(_QWORD *)(v9 + 8);
          v13 = v11 != 0;
        }
        v7 = v102;
        v35 = *(_QWORD *)(*(v102 - 1) + 8);
        if ( !v35 )
          goto LABEL_40;
        if ( !v13 )
          goto LABEL_48;
        v36 = sub_BD5D20(v35);
        v38 = v37;
        s2 = v36;
        v39 = sub_BD5D20(v11);
        v41 = v40;
        v21 = v40 <= v38;
        v42 = v38;
        if ( v21 )
          v42 = v41;
        if ( v42 && (v43 = memcmp(v39, s2, v42)) != 0 )
        {
          if ( v43 < 0 )
            goto LABEL_83;
        }
        else if ( v41 != v38 && v41 < v38 )
        {
          goto LABEL_83;
        }
        v44 = *(v102 - 1);
        v45 = *(_QWORD *)v8;
        v46 = *(_QWORD *)(v44 + 8);
        v47 = *(_QWORD *)(*(_QWORD *)v8 + 8LL);
        if ( v46 )
          break;
LABEL_41:
        v57 = *(_QWORD *)a1;
        *(_QWORD *)a1 = v45;
        *(_QWORD *)v8 = v57;
        v58 = *(_QWORD *)a1;
        v59 = *((_QWORD *)a1 + 1);
        v9 = *(_QWORD *)a1;
LABEL_49:
        v63 = v100;
        v64 = v102;
        while ( 1 )
        {
          v74 = *(_QWORD *)(v9 + 8);
          v69 = *(_QWORD *)(v59 + 8);
          v104 = v63;
          if ( !v74 )
            goto LABEL_60;
          s2a = *(void **)(v59 + 8);
          if ( v69 )
            break;
LABEL_58:
          v59 = v63[1];
          v9 = v58;
          ++v63;
        }
        v65 = sub_BD5D20(v74);
        v67 = v66;
        v103 = v65;
        v68 = sub_BD5D20((__int64)s2a);
        v71 = v70;
        v21 = v70 <= v67;
        v72 = v67;
        if ( v21 )
          v72 = v71;
        if ( !v72 || (v73 = memcmp(v68, v103, v72)) == 0 )
        {
          if ( v71 != v67 && v71 < v67 )
            goto LABEL_57;
          goto LABEL_71;
        }
        if ( v73 < 0 )
          goto LABEL_57;
        for ( i = *(_QWORD *)(*(_QWORD *)a1 + 8LL); ; i = *(_QWORD *)(*(_QWORD *)a1 + 8LL) )
        {
          v86 = *--v64;
          v76 = *(_QWORD *)(v86 + 8);
          if ( !i )
            break;
          if ( !v76 )
            goto LABEL_68;
          v77 = sub_BD5D20(v76);
          v79 = v78;
          s2b = v77;
          v80 = sub_BD5D20(i);
          v82 = v81;
          if ( v79 <= v81 )
            v81 = v79;
          if ( v81 && (v83 = memcmp(v80, s2b, v81)) != 0 )
          {
            if ( v83 >= 0 )
              goto LABEL_68;
          }
          else if ( v79 == v82 || v79 <= v82 )
          {
LABEL_68:
            if ( v63 >= v64 )
              goto LABEL_75;
LABEL_69:
            v84 = *v63;
            *v63 = *v64;
            *v64 = v84;
LABEL_57:
            v58 = *(_QWORD *)a1;
            goto LABEL_58;
          }
LABEL_71:
          ;
        }
        while ( v76 )
        {
LABEL_60:
          v75 = *--v64;
          v76 = *(_QWORD *)(v75 + 8);
        }
        if ( v63 < v64 )
          goto LABEL_69;
LABEL_75:
        sub_D0F410(v63, v102, v101, v7, a5, v69);
        result = (char *)v63 - a1;
        if ( (char *)v63 - a1 <= 128 )
          return result;
        if ( !v101 )
          goto LABEL_44;
        v102 = v63;
      }
      if ( !v47 )
        goto LABEL_94;
      v48 = sub_BD5D20(v46);
      v50 = v49;
      v51 = v48;
      v52 = sub_BD5D20(v47);
      v54 = v53;
      v21 = v53 <= v50;
      v55 = v50;
      if ( v21 )
        v55 = v54;
      if ( v55 && (v56 = memcmp(v52, v51, v55)) != 0 )
      {
        if ( v56 < 0 )
          goto LABEL_93;
      }
      else if ( v54 != v50 && v54 < v50 )
      {
LABEL_93:
        v44 = *(v102 - 1);
LABEL_94:
        v99 = *(_QWORD *)a1;
        v7 = v102;
        *(_QWORD *)a1 = v44;
        *(v102 - 1) = v99;
        v58 = *(_QWORD *)a1;
        v59 = *((_QWORD *)a1 + 1);
        v9 = *(_QWORD *)a1;
        goto LABEL_49;
      }
LABEL_40:
      v45 = *(_QWORD *)v8;
      goto LABEL_41;
    }
    v104 = a2;
LABEL_44:
    v60 = result >> 3;
    v61 = ((result >> 3) - 2) >> 1;
    sub_D0EFF0((__int64)a1, v61, result >> 3, *(_QWORD *)&a1[8 * v61]);
    do
    {
      --v61;
      sub_D0EFF0((__int64)a1, v61, v60, *(_QWORD *)&a1[8 * v61]);
    }
    while ( v61 );
    do
    {
      v62 = *--v104;
      *v104 = *(_QWORD *)a1;
      result = sub_D0EFF0((__int64)a1, 0, ((char *)v104 - a1) >> 3, v62);
    }
    while ( (char *)v104 - a1 > 8 );
  }
  return result;
}
