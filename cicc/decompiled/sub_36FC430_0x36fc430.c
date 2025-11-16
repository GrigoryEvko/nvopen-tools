// Function: sub_36FC430
// Address: 0x36fc430
//
char __fastcall sub_36FC430(__int64 a1, char *a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v4; // r11
  char *v5; // r10
  char *v6; // rbx
  size_t *v7; // r15
  size_t **v8; // r9
  size_t *v9; // r8
  int v10; // r13d
  size_t v11; // r11
  bool v12; // zf
  bool v13; // sf
  bool v14; // of
  size_t *v15; // r14
  size_t v16; // rcx
  int v17; // r12d
  size_t *v18; // rax
  size_t *v19; // r13
  size_t *v20; // r12
  size_t v21; // r9
  size_t *v22; // rcx
  size_t **v23; // r14
  size_t **v24; // r11
  int v25; // r13d
  size_t v26; // rax
  int v27; // esi
  size_t **v28; // rbx
  size_t v29; // rax
  bool v30; // zf
  bool v31; // sf
  bool v32; // of
  int v33; // eax
  size_t v34; // r10
  size_t v35; // r8
  size_t v36; // rdx
  int v37; // eax
  int v38; // eax
  int v39; // esi
  size_t v40; // r8
  size_t v41; // rbx
  size_t v42; // rdx
  int v43; // eax
  int v44; // esi
  size_t v45; // r12
  size_t v46; // r14
  size_t v47; // rdx
  int v48; // eax
  bool v49; // cc
  bool v50; // zf
  bool v51; // sf
  bool v52; // of
  __int64 v53; // r13
  __int64 v54; // r12
  __int64 i; // rbx
  char *v56; // r12
  char *v57; // rbx
  size_t *v58; // rcx
  __int64 v59; // r13
  int v60; // esi
  size_t v61; // rax
  size_t v62; // r13
  int v63; // eax
  bool v64; // cc
  bool v65; // zf
  bool v66; // sf
  bool v67; // of
  int v68; // eax
  size_t v69; // r15
  size_t v70; // r12
  size_t v71; // rdx
  size_t v72; // rax
  size_t v73; // r11
  int v74; // eax
  size_t v75; // r13
  size_t v76; // r12
  size_t v77; // rdx
  int v78; // eax
  int v79; // eax
  char *v81; // [rsp+10h] [rbp-80h]
  __int64 v82; // [rsp+18h] [rbp-78h]
  size_t **v83; // [rsp+20h] [rbp-70h]
  size_t **v84; // [rsp+28h] [rbp-68h]
  char *v85; // [rsp+28h] [rbp-68h]
  char *v86; // [rsp+28h] [rbp-68h]
  size_t *v87; // [rsp+30h] [rbp-60h]
  char *v88; // [rsp+30h] [rbp-60h]
  size_t v89; // [rsp+30h] [rbp-60h]
  size_t v90; // [rsp+30h] [rbp-60h]
  size_t v91; // [rsp+38h] [rbp-58h]
  size_t **v92; // [rsp+38h] [rbp-58h]
  size_t v93; // [rsp+38h] [rbp-58h]
  size_t *v94; // [rsp+38h] [rbp-58h]
  size_t v95; // [rsp+38h] [rbp-58h]
  size_t v96; // [rsp+40h] [rbp-50h]
  size_t v97; // [rsp+40h] [rbp-50h]
  size_t **v98; // [rsp+40h] [rbp-50h]
  size_t v99; // [rsp+40h] [rbp-50h]
  size_t **v100; // [rsp+40h] [rbp-50h]
  size_t v101; // [rsp+48h] [rbp-48h]
  size_t v102; // [rsp+48h] [rbp-48h]
  size_t *v103; // [rsp+48h] [rbp-48h]
  size_t **v104; // [rsp+48h] [rbp-48h]
  size_t *v105; // [rsp+48h] [rbp-48h]
  char *v106; // [rsp+48h] [rbp-48h]
  size_t v107; // [rsp+50h] [rbp-40h]
  size_t *v108; // [rsp+50h] [rbp-40h]
  size_t v109; // [rsp+50h] [rbp-40h]
  size_t **v110; // [rsp+50h] [rbp-40h]
  int v111; // [rsp+58h] [rbp-38h]
  char *v112; // [rsp+58h] [rbp-38h]
  size_t v113; // [rsp+58h] [rbp-38h]
  char *v114; // [rsp+58h] [rbp-38h]
  size_t *v115; // [rsp+58h] [rbp-38h]

  v3 = (__int64)&a2[-a1];
  v82 = a3;
  if ( (__int64)&a2[-a1] <= 128 )
    return v3;
  v4 = a1;
  if ( !a3 )
  {
    v112 = a2;
    goto LABEL_61;
  }
  v5 = a2;
  v6 = (char *)a1;
  while ( 2 )
  {
    v7 = (size_t *)*((_QWORD *)v6 + 1);
    --v82;
    v8 = (size_t **)&v6[8 * ((__int64)(((v5 - v6) >> 3) + ((unsigned __int64)(v5 - v6) >> 63)) >> 1)];
    v9 = *v8;
    v107 = v7[1];
    v10 = *(_DWORD *)(v107 + 80);
    v11 = (*v8)[1];
    v111 = *(_DWORD *)(v11 + 80);
    v14 = __OFSUB__(v10, v111);
    v12 = v10 == v111;
    v13 = v10 - v111 < 0;
    if ( v10 == v111
      && (v44 = *(_DWORD *)(v11 + 84),
          v14 = __OFSUB__(*(_DWORD *)(v107 + 84), v44),
          v12 = *(_DWORD *)(v107 + 84) == v44,
          v13 = *(_DWORD *)(v107 + 84) - v44 < 0,
          *(_DWORD *)(v107 + 84) == v44) )
    {
      v45 = *v9;
      v46 = *v7;
      v47 = *v7;
      if ( *v9 <= *v7 )
        v47 = *v9;
      if ( !v47 )
        goto LABEL_53;
      v88 = v5;
      v93 = (*v8)[1];
      v98 = (size_t **)&v6[8 * ((__int64)(((v5 - v6) >> 3) + ((unsigned __int64)(v5 - v6) >> 63)) >> 1)];
      v103 = *v8;
      v48 = memcmp(v7 + 2, v9 + 2, v47);
      v9 = v103;
      v8 = v98;
      v11 = v93;
      v5 = v88;
      if ( !v48 )
      {
LABEL_53:
        v12 = v45 == v46;
        v49 = v45 <= v46;
        v15 = (size_t *)*((_QWORD *)v5 - 1);
        v16 = v15[1];
        v17 = *(_DWORD *)(v16 + 80);
        if ( v12 || v49 )
          goto LABEL_55;
LABEL_6:
        if ( v17 == v111 )
        {
          v38 = *(_DWORD *)(v16 + 84);
          if ( *(_DWORD *)(v11 + 84) == v38 )
          {
            v72 = *v9;
            v73 = *v15;
            v113 = *v9;
            if ( *v15 <= *v9 )
              v72 = *v15;
            if ( !v72 )
              goto LABEL_89;
            v86 = v5;
            v90 = *v15;
            v95 = v16;
            v100 = v8;
            v105 = v9;
            v74 = memcmp(v9 + 2, v15 + 2, v72);
            v9 = v105;
            v8 = v100;
            v16 = v95;
            v73 = v90;
            v5 = v86;
            if ( v74 )
            {
              if ( v74 < 0 )
                goto LABEL_8;
            }
            else
            {
LABEL_89:
              if ( v73 != v113 && v73 > v113 )
                goto LABEL_8;
            }
          }
          else if ( *(_DWORD *)(v11 + 84) > v38 )
          {
            goto LABEL_8;
          }
        }
        else if ( v17 < v111 )
        {
LABEL_8:
          v18 = *(size_t **)v6;
          *(_QWORD *)v6 = v9;
          *v8 = v18;
          v19 = (size_t *)*((_QWORD *)v6 + 1);
          v7 = *(size_t **)v6;
          v20 = (size_t *)*((_QWORD *)v5 - 1);
          goto LABEL_9;
        }
        if ( v17 == v10 )
        {
          v39 = *(_DWORD *)(v16 + 84);
          if ( *(_DWORD *)(v107 + 84) == v39 )
          {
            v75 = *v15;
            v76 = *v7;
            v77 = *v7;
            if ( *v15 <= *v7 )
              v77 = *v15;
            if ( v77 && (v114 = v5, v78 = memcmp(v7 + 2, v15 + 2, v77), v5 = v114, v78) )
            {
              if ( v78 < 0 )
                goto LABEL_31;
            }
            else if ( v75 != v76 && v75 > v76 )
            {
              goto LABEL_31;
            }
          }
          else if ( *(_DWORD *)(v107 + 84) > v39 )
          {
            goto LABEL_31;
          }
        }
        else if ( v17 < v10 )
        {
LABEL_31:
          v20 = *(size_t **)v6;
          *(_QWORD *)v6 = v15;
          *((_QWORD *)v5 - 1) = v20;
          v19 = (size_t *)*((_QWORD *)v6 + 1);
          v7 = *(size_t **)v6;
          goto LABEL_9;
        }
        v19 = *(size_t **)v6;
        *(_QWORD *)v6 = v7;
        *((_QWORD *)v6 + 1) = v19;
        v20 = (size_t *)*((_QWORD *)v5 - 1);
        goto LABEL_9;
      }
      v15 = (size_t *)*((_QWORD *)v88 - 1);
      v16 = v15[1];
      v17 = *(_DWORD *)(v16 + 80);
      if ( v48 < 0 )
        goto LABEL_6;
    }
    else
    {
      v15 = (size_t *)*((_QWORD *)v5 - 1);
      v16 = v15[1];
      v17 = *(_DWORD *)(v16 + 80);
      if ( !(v13 ^ v14 | v12) )
        goto LABEL_6;
    }
LABEL_55:
    v52 = __OFSUB__(v10, v17);
    v50 = v10 == v17;
    v51 = v10 - v17 < 0;
    if ( v10 == v17
      && (v60 = *(_DWORD *)(v16 + 84),
          v52 = __OFSUB__(*(_DWORD *)(v107 + 84), v60),
          v50 = *(_DWORD *)(v107 + 84) == v60,
          v51 = *(_DWORD *)(v107 + 84) - v60 < 0,
          *(_DWORD *)(v107 + 84) == v60) )
    {
      v61 = *v7;
      v62 = *v15;
      v109 = *v7;
      if ( *v15 <= *v7 )
        v61 = *v15;
      if ( !v61 )
        goto LABEL_72;
      v85 = v5;
      v89 = v11;
      v94 = v9;
      v99 = v16;
      v104 = v8;
      v63 = memcmp(v7 + 2, v15 + 2, v61);
      v8 = v104;
      v16 = v99;
      v9 = v94;
      v11 = v89;
      v5 = v85;
      if ( v63 )
      {
        v19 = *(size_t **)v6;
        if ( v63 >= 0 )
          goto LABEL_74;
      }
      else
      {
LABEL_72:
        v12 = v62 == v109;
        v64 = v62 <= v109;
        v19 = *(size_t **)v6;
        if ( v12 || v64 )
        {
LABEL_74:
          v67 = __OFSUB__(v111, v17);
          v65 = v111 == v17;
          v66 = v111 - v17 < 0;
          if ( v111 == v17
            && (v68 = *(_DWORD *)(v16 + 84),
                v67 = __OFSUB__(*(_DWORD *)(v11 + 84), v68),
                v65 = *(_DWORD *)(v11 + 84) == v68,
                v66 = *(_DWORD *)(v11 + 84) - v68 < 0,
                *(_DWORD *)(v11 + 84) == v68) )
          {
            v69 = *v15;
            v70 = *v9;
            v71 = *v9;
            if ( *v15 <= *v9 )
              v71 = *v15;
            if ( v71
              && (v106 = v5,
                  v110 = v8,
                  v115 = v9,
                  v79 = memcmp(v9 + 2, v15 + 2, v71),
                  v9 = v115,
                  v8 = v110,
                  v5 = v106,
                  v79) )
            {
              if ( v79 < 0 )
                goto LABEL_76;
            }
            else if ( v69 != v70 && v69 > v70 )
            {
              goto LABEL_76;
            }
          }
          else if ( !(v66 ^ v67 | v65) )
          {
LABEL_76:
            *(_QWORD *)v6 = v15;
            v20 = v19;
            *((_QWORD *)v5 - 1) = v19;
            v7 = *(size_t **)v6;
            v19 = (size_t *)*((_QWORD *)v6 + 1);
            goto LABEL_9;
          }
          *(_QWORD *)v6 = v9;
          *v8 = v19;
          v19 = (size_t *)*((_QWORD *)v6 + 1);
          v7 = *(size_t **)v6;
          v20 = (size_t *)*((_QWORD *)v5 - 1);
          goto LABEL_9;
        }
      }
    }
    else
    {
      v19 = *(size_t **)v6;
      if ( v51 ^ v52 | v50 )
        goto LABEL_74;
    }
    *(_QWORD *)v6 = v7;
    *((_QWORD *)v6 + 1) = v19;
    v20 = (size_t *)*((_QWORD *)v5 - 1);
LABEL_9:
    v21 = v7[1];
    v22 = v19;
    v81 = v5;
    v23 = (size_t **)v5;
    v83 = (size_t **)v6;
    v24 = (size_t **)(a1 + 16);
    v25 = *(_DWORD *)(v21 + 80);
    while ( 1 )
    {
      v112 = (char *)(v24 - 1);
      v26 = v22[1];
      if ( v25 != *(_DWORD *)(v26 + 80) )
      {
        if ( v25 >= *(_DWORD *)(v26 + 80) )
          goto LABEL_15;
        goto LABEL_11;
      }
      v27 = *(_DWORD *)(v21 + 84);
      if ( *(_DWORD *)(v26 + 84) != v27 )
      {
        if ( *(_DWORD *)(v26 + 84) <= v27 )
          goto LABEL_15;
        goto LABEL_11;
      }
      v40 = *v7;
      v41 = *v22;
      v42 = *v22;
      if ( *v7 <= *v22 )
        v42 = *v7;
      if ( !v42 )
        break;
      v92 = v24;
      v97 = v21;
      v102 = *v7;
      v108 = v22;
      v43 = memcmp(v22 + 2, v7 + 2, v42);
      v22 = v108;
      v40 = v102;
      v21 = v97;
      v24 = v92;
      if ( !v43 )
        break;
      if ( v43 >= 0 )
        goto LABEL_15;
LABEL_11:
      v22 = *v24++;
    }
    if ( v40 != v41 && v40 > v41 )
      goto LABEL_11;
LABEL_15:
    v28 = v23 - 1;
    while ( 2 )
    {
      v29 = v20[1];
      v23 = v28;
      v32 = __OFSUB__(v25, *(_DWORD *)(v29 + 80));
      v30 = v25 == *(_DWORD *)(v29 + 80);
      v31 = v25 - *(_DWORD *)(v29 + 80) < 0;
      if ( v25 != *(_DWORD *)(v29 + 80)
        || (v33 = *(_DWORD *)(v29 + 84),
            v32 = __OFSUB__(*(_DWORD *)(v21 + 84), v33),
            v30 = *(_DWORD *)(v21 + 84) == v33,
            v31 = *(_DWORD *)(v21 + 84) - v33 < 0,
            *(_DWORD *)(v21 + 84) != v33) )
      {
        --v28;
        if ( v31 ^ v32 | v30 )
          goto LABEL_36;
        goto LABEL_17;
      }
      v34 = *v20;
      v35 = *v7;
      v36 = *v7;
      if ( *v20 <= *v7 )
        v36 = *v20;
      if ( v36 )
      {
        v84 = v24;
        v87 = v22;
        v91 = v21;
        v96 = *v7;
        v101 = *v20;
        v37 = memcmp(v7 + 2, v20 + 2, v36);
        v34 = v101;
        v35 = v96;
        v21 = v91;
        v22 = v87;
        v24 = v84;
        if ( v37 )
        {
          if ( v37 >= 0 )
            goto LABEL_36;
LABEL_26:
          --v28;
LABEL_17:
          v20 = *v28;
          continue;
        }
      }
      break;
    }
    if ( v34 != v35 && v34 > v35 )
      goto LABEL_26;
LABEL_36:
    if ( v23 > (size_t **)v112 )
    {
      *(v24 - 1) = v20;
      *v23 = v22;
      v20 = *(v23 - 1);
      v7 = *v83;
      v21 = (*v83)[1];
      v25 = *(_DWORD *)(v21 + 80);
      goto LABEL_11;
    }
    v6 = (char *)v83;
    sub_36FC430(v112, v81, v82);
    v3 = v112 - (char *)v83;
    if ( v112 - (char *)v83 > 128 )
    {
      if ( v82 )
      {
        v5 = v112;
        continue;
      }
      v4 = (__int64)v83;
LABEL_61:
      v53 = v4;
      v54 = v3 >> 3;
      for ( i = ((v3 >> 3) - 2) >> 1; ; --i )
      {
        sub_36FBFE0(v53, i, v54, *(size_t **)(v53 + 8 * i));
        if ( !i )
          break;
      }
      v56 = (char *)v53;
      v57 = v112 - 8;
      do
      {
        v58 = *(size_t **)v57;
        v59 = v57 - v56;
        v57 -= 8;
        *((_QWORD *)v57 + 1) = *(_QWORD *)v56;
        LOBYTE(v3) = sub_36FBFE0((__int64)v56, 0, v59 >> 3, v58);
      }
      while ( v59 > 8 );
    }
    return v3;
  }
}
