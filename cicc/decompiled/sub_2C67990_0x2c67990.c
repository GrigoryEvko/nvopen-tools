// Function: sub_2C67990
// Address: 0x2c67990
//
__int64 __fastcall sub_2C67990(
        unsigned int *a1,
        unsigned int *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        char *a6,
        __int64 a7,
        __int64 **a8,
        _BYTE **a9)
{
  unsigned int *v9; // r14
  unsigned int *v10; // r13
  unsigned int *v11; // r12
  _QWORD *v13; // r10
  __int64 result; // rax
  __int64 v15; // r15
  __int64 v16; // r8
  __int64 v17; // rcx
  char *v18; // rdx
  unsigned int *v19; // rax
  unsigned int v20; // esi
  char *v21; // r15
  unsigned int *v22; // r8
  char *v23; // r15
  _BYTE *v24; // rcx
  __int64 v25; // rdx
  __int64 v26; // r12
  __int64 *v27; // r11
  unsigned __int8 *v28; // rax
  __int64 v29; // r9
  char v30; // di
  __int64 v31; // rsi
  __int64 v32; // r9
  __int64 v33; // r12
  unsigned int *i; // r10
  __int64 v35; // r14
  unsigned int *v36; // rax
  char *v37; // r10
  unsigned int *v38; // r13
  __int64 v39; // r11
  char *v40; // rax
  int v41; // ecx
  __int64 v42; // rdx
  __int64 v43; // r8
  __int64 v44; // rcx
  char *v45; // rdx
  unsigned int *v46; // rax
  unsigned int v47; // esi
  char *v48; // r8
  unsigned int *v49; // r9
  unsigned int *v50; // r14
  unsigned int v51; // eax
  _BYTE *v52; // rdx
  __int64 v53; // r12
  __int64 v54; // r15
  unsigned __int8 *v55; // rax
  __int64 v56; // rsi
  char v57; // di
  __int64 v58; // rcx
  __int64 v59; // rsi
  unsigned int v60; // eax
  __int64 v61; // r8
  unsigned int v62; // edx
  __int64 v63; // rdi
  _QWORD *v64; // rax
  _QWORD *v65; // rdx
  __int64 v66; // rcx
  _QWORD *v67; // r11
  _QWORD *v68; // rcx
  __int64 v69; // rdi
  _QWORD *v70; // rax
  _QWORD *v71; // rcx
  __int64 v72; // rsi
  _QWORD *v73; // r13
  _QWORD *v74; // rsi
  _QWORD *v75; // rax
  _QWORD *v76; // rsi
  unsigned int *v77; // rax
  __int64 *v78; // rax
  __int64 *v79; // rax
  __int64 *v80; // rax
  __int64 *v81; // rax
  unsigned int *v82; // r15
  unsigned int *v83; // r14
  unsigned int *v84; // r14
  char *j; // r15
  _QWORD *v86; // [rsp+8h] [rbp-68h]
  _QWORD *v87; // [rsp+8h] [rbp-68h]
  unsigned int *v89; // [rsp+10h] [rbp-60h]
  unsigned int *v90; // [rsp+10h] [rbp-60h]
  unsigned int *v91; // [rsp+10h] [rbp-60h]
  unsigned int *v92; // [rsp+10h] [rbp-60h]
  unsigned int *v93; // [rsp+18h] [rbp-58h]
  int v94; // [rsp+18h] [rbp-58h]
  unsigned int *v95; // [rsp+18h] [rbp-58h]
  char *v96; // [rsp+18h] [rbp-58h]
  char *v97; // [rsp+18h] [rbp-58h]
  int v98; // [rsp+18h] [rbp-58h]
  unsigned int v99; // [rsp+18h] [rbp-58h]
  unsigned int *v100; // [rsp+20h] [rbp-50h]
  _QWORD *v101; // [rsp+20h] [rbp-50h]
  _QWORD *v102; // [rsp+20h] [rbp-50h]
  _BYTE *v103; // [rsp+20h] [rbp-50h]
  _BYTE *v104; // [rsp+20h] [rbp-50h]
  unsigned int *v105; // [rsp+28h] [rbp-48h]
  _QWORD *v106; // [rsp+28h] [rbp-48h]
  _BYTE *v107; // [rsp+28h] [rbp-48h]
  _BYTE *v108; // [rsp+28h] [rbp-48h]
  __int64 v109; // [rsp+28h] [rbp-48h]
  __int64 v110; // [rsp+28h] [rbp-48h]
  unsigned int *v111; // [rsp+30h] [rbp-40h]
  __int64 *v112; // [rsp+30h] [rbp-40h]

  v9 = (unsigned int *)a3;
  v10 = a1;
  v11 = a2;
  v13 = a9;
  result = a5;
  if ( a7 <= a5 )
    result = a7;
  if ( a4 <= result )
    goto LABEL_33;
  v15 = a5;
  if ( a7 >= a5 )
    goto LABEL_5;
  v105 = a1;
  v33 = a4;
  for ( i = a2; ; i = v100 )
  {
    v93 = i;
    if ( v15 < v33 )
    {
      v38 = &v105[2 * (v33 / 2)];
      v77 = sub_2C4E5D0(i, a3, v38, a8, a9);
      v37 = (char *)v93;
      v39 = v33 / 2;
      v100 = v77;
      v35 = ((char *)v77 - (char *)v93) >> 3;
    }
    else
    {
      v35 = v15 / 2;
      v100 = &i[2 * (v15 / 2)];
      v36 = sub_2C4E390(v105, (__int64)i, v100, a8, a9);
      v37 = (char *)v93;
      v38 = v36;
      v39 = ((char *)v36 - (char *)v105) >> 3;
    }
    v33 -= v39;
    v94 = v39;
    v15 -= v35;
    v40 = sub_2C67750((char *)v38, v37, (char *)v100, v33, v35, a6, a7);
    v41 = v94;
    v95 = (unsigned int *)v40;
    sub_2C67990((_DWORD)v105, (_DWORD)v38, (_DWORD)v40, v41, v35, (_DWORD)a6, a7, (__int64)a8, (__int64)a9);
    v42 = v15;
    if ( v15 > a7 )
      v42 = a7;
    result = (__int64)v95;
    if ( v33 <= v42 )
    {
      v13 = a9;
      v9 = (unsigned int *)a3;
      v10 = v95;
      v11 = v100;
LABEL_33:
      v43 = (char *)v11 - (char *)v10;
      v44 = ((char *)v11 - (char *)v10) >> 3;
      if ( (char *)v11 - (char *)v10 <= 0 )
        return result;
      v45 = a6;
      v46 = v10;
      do
      {
        v47 = *v46;
        v45 += 8;
        v46 += 2;
        *((_DWORD *)v45 - 2) = v47;
        *((_DWORD *)v45 - 1) = *(v46 - 1);
        --v44;
      }
      while ( v44 );
      result = 8;
      if ( v43 <= 0 )
        v43 = 8;
      v48 = &a6[v43];
      if ( v9 == v11 )
      {
LABEL_53:
        if ( a6 != v48 )
        {
          v61 = v48 - a6;
          result = v61 >> 3;
          if ( v61 > 0 )
          {
            do
            {
              v62 = *(_DWORD *)a6;
              v10 += 2;
              a6 += 8;
              *(v10 - 2) = v62;
              *(v10 - 1) = *((_DWORD *)a6 - 1);
              --result;
            }
            while ( result );
          }
        }
        return result;
      }
      if ( a6 == v48 )
        return result;
      v49 = v9;
      v50 = v11;
      while ( 1 )
      {
        v52 = (_BYTE *)*v13;
        v53 = *v50;
        v54 = *(unsigned int *)a6;
        if ( *(_BYTE *)*v13 != 92 )
          goto LABEL_50;
        v112 = *a8;
        v55 = (unsigned __int8 *)*((_QWORD *)v52 - 4);
        if ( (unsigned int)*v55 - 12 <= 1 )
        {
          v56 = *((_QWORD *)v52 - 8);
          v57 = *(_BYTE *)v56;
          if ( *(_BYTE *)v56 == 92 )
          {
            v66 = *v112;
            if ( !*(_BYTE *)(*v112 + 28) )
            {
              v89 = v49;
              v96 = v48;
              v101 = v13;
              v107 = (_BYTE *)*v13;
              v78 = sub_C8CA60(*v112, v56);
              v52 = v107;
              v13 = v101;
              v48 = v96;
              v49 = v89;
              v57 = *v107;
              v112 = *a8;
              if ( v78 )
                goto LABEL_85;
LABEL_96:
              v58 = *((_QWORD *)v52 + 9);
              LODWORD(v53) = *(_DWORD *)(v58 + 4 * v53);
LABEL_86:
              if ( v57 != 92 )
                goto LABEL_50;
              v55 = (unsigned __int8 *)*((_QWORD *)v52 - 4);
              goto LABEL_47;
            }
            v67 = *(_QWORD **)(v66 + 8);
            v106 = &v67[*(unsigned int *)(v66 + 20)];
            if ( v67 != v106 )
            {
              v68 = *(_QWORD **)(v66 + 8);
              while ( v56 != *v68 )
              {
                if ( v106 == ++v68 )
                  goto LABEL_96;
              }
LABEL_85:
              v58 = *((_QWORD *)v52 + 9);
              LODWORD(v53) = *(_DWORD *)(*(_QWORD *)(v56 + 72) + 4LL * *(unsigned int *)(v58 + 4 * v53));
              goto LABEL_86;
            }
          }
        }
        v58 = *((_QWORD *)v52 + 9);
        LODWORD(v53) = *(_DWORD *)(v58 + 4 * v53);
LABEL_47:
        if ( (unsigned int)*v55 - 12 <= 1 )
        {
          v59 = *((_QWORD *)v52 - 8);
          if ( *(_BYTE *)v59 == 92 )
          {
            v63 = *v112;
            if ( *(_BYTE *)(*v112 + 28) )
            {
              v64 = *(_QWORD **)(v63 + 8);
              v65 = &v64[*(unsigned int *)(v63 + 20)];
              if ( v64 != v65 )
              {
                while ( v59 != *v64 )
                {
                  if ( v65 == ++v64 )
                    goto LABEL_49;
                }
LABEL_89:
                LODWORD(v54) = *(_DWORD *)(*(_QWORD *)(v59 + 72) + 4LL * *(unsigned int *)(v58 + 4 * v54));
                goto LABEL_50;
              }
            }
            else
            {
              v90 = v49;
              v97 = v48;
              v102 = v13;
              v108 = v52;
              v79 = sub_C8CA60(v63, v59);
              v13 = v102;
              v48 = v97;
              v49 = v90;
              v58 = *((_QWORD *)v108 + 9);
              if ( v79 )
                goto LABEL_89;
            }
          }
        }
LABEL_49:
        LODWORD(v54) = *(_DWORD *)(v58 + 4 * v54);
LABEL_50:
        if ( (int)v53 < (int)v54 )
        {
          v51 = *v50;
          v10 += 2;
          v50 += 2;
          *(v10 - 2) = v51;
          result = *(v50 - 1);
          *(v10 - 1) = result;
          if ( a6 == v48 )
            return result;
        }
        else
        {
          v60 = *(_DWORD *)a6;
          a6 += 8;
          v10 += 2;
          *(v10 - 2) = v60;
          result = *((unsigned int *)a6 - 1);
          *(v10 - 1) = result;
          if ( a6 == v48 )
            return result;
        }
        if ( v49 == v50 )
          goto LABEL_53;
      }
    }
    if ( v15 <= a7 )
      break;
    v105 = v95;
  }
  v13 = a9;
  v9 = (unsigned int *)a3;
  v10 = v95;
  v11 = v100;
LABEL_5:
  v16 = (char *)v9 - (char *)v11;
  v17 = ((char *)v9 - (char *)v11) >> 3;
  if ( (char *)v9 - (char *)v11 <= 0 )
    return result;
  v18 = a6;
  v19 = v11;
  do
  {
    v20 = *v19;
    v18 += 8;
    v19 += 2;
    *((_DWORD *)v18 - 2) = v20;
    *((_DWORD *)v18 - 1) = *(v19 - 1);
    --v17;
  }
  while ( v17 );
  result = 8;
  if ( v16 <= 0 )
    v16 = 8;
  v21 = &a6[v16];
  if ( v10 == v11 )
  {
    result = v16 >> 3;
    v84 = &v9[-2 * (v16 >> 3)];
    for ( j = &v21[-8 * (v16 >> 3)]; ; v20 = *(_DWORD *)&j[8 * result - 8] )
    {
      v84[2 * result - 2] = v20;
      v84[2 * result - 1] = *(_DWORD *)&j[8 * result - 4];
      if ( !--result )
        break;
    }
    return result;
  }
  if ( a6 == v21 )
    return result;
  v111 = v10;
  v22 = v11 - 2;
  v23 = v21 - 8;
  while ( 2 )
  {
    v24 = (_BYTE *)*v13;
    v25 = *(unsigned int *)v23;
    v26 = *v22;
    if ( *(_BYTE *)*v13 != 92 )
      goto LABEL_20;
LABEL_14:
    v27 = *a8;
    v28 = (unsigned __int8 *)*((_QWORD *)v24 - 4);
    if ( (unsigned int)*v28 - 12 <= 1 )
    {
      v29 = *((_QWORD *)v24 - 8);
      v30 = *(_BYTE *)v29;
      if ( *(_BYTE *)v29 == 92 )
      {
        v72 = *v27;
        if ( *(_BYTE *)(*v27 + 28) )
        {
          v73 = *(_QWORD **)(v72 + 8);
          v74 = &v73[*(unsigned int *)(v72 + 20)];
          if ( v73 == v74 )
            goto LABEL_16;
          v75 = v74;
          v76 = v73;
          while ( v29 != *v76 )
          {
            if ( v75 == ++v76 )
              goto LABEL_93;
          }
        }
        else
        {
          v87 = v13;
          v92 = v22;
          v99 = v25;
          v104 = v24;
          v110 = *((_QWORD *)v24 - 8);
          v81 = sub_C8CA60(*v27, v110);
          v24 = v104;
          v29 = v110;
          v25 = v99;
          v22 = v92;
          v13 = v87;
          v27 = *a8;
          v30 = *v104;
          if ( !v81 )
          {
LABEL_93:
            v31 = *((_QWORD *)v24 + 9);
            LODWORD(v25) = *(_DWORD *)(v31 + 4 * v25);
LABEL_81:
            if ( v30 == 92 )
            {
              v28 = (unsigned __int8 *)*((_QWORD *)v24 - 4);
              goto LABEL_17;
            }
            goto LABEL_20;
          }
        }
        v31 = *((_QWORD *)v24 + 9);
        LODWORD(v25) = *(_DWORD *)(*(_QWORD *)(v29 + 72) + 4LL * *(unsigned int *)(v31 + 4 * v25));
        goto LABEL_81;
      }
    }
LABEL_16:
    v31 = *((_QWORD *)v24 + 9);
    LODWORD(v25) = *(_DWORD *)(v31 + 4 * v25);
LABEL_17:
    if ( (unsigned int)*v28 - 12 <= 1 && (v32 = *((_QWORD *)v24 - 8), *(_BYTE *)v32 == 92) )
    {
      v69 = *v27;
      if ( *(_BYTE *)(*v27 + 28) )
      {
        v70 = *(_QWORD **)(v69 + 8);
        v71 = &v70[*(unsigned int *)(v69 + 20)];
        if ( v70 == v71 )
          goto LABEL_19;
        while ( v32 != *v70 )
        {
          if ( v71 == ++v70 )
            goto LABEL_19;
        }
      }
      else
      {
        v86 = v13;
        v91 = v22;
        v98 = v25;
        v103 = v24;
        v109 = *((_QWORD *)v24 - 8);
        v80 = sub_C8CA60(v69, v109);
        v32 = v109;
        LODWORD(v25) = v98;
        v22 = v91;
        v13 = v86;
        v31 = *((_QWORD *)v103 + 9);
        if ( !v80 )
          goto LABEL_19;
      }
      LODWORD(v26) = *(_DWORD *)(*(_QWORD *)(v32 + 72) + 4LL * *(unsigned int *)(v31 + 4 * v26));
    }
    else
    {
LABEL_19:
      LODWORD(v26) = *(_DWORD *)(v31 + 4 * v26);
    }
LABEL_20:
    v9 -= 2;
    if ( (int)v26 <= (int)v25 )
    {
LABEL_24:
      *v9 = *(_DWORD *)v23;
      result = *((unsigned int *)v23 + 1);
      v9[1] = result;
      if ( a6 == v23 )
        return result;
      v23 -= 8;
      continue;
    }
    break;
  }
  while ( 1 )
  {
    *v9 = *v22;
    v9[1] = v22[1];
    if ( v111 == v22 )
      break;
    v24 = (_BYTE *)*v13;
    v22 -= 2;
    v25 = *(unsigned int *)v23;
    v26 = *v22;
    if ( *(_BYTE *)*v13 == 92 )
      goto LABEL_14;
    v9 -= 2;
    if ( (int)v26 <= (int)v25 )
      goto LABEL_24;
  }
  result = (v23 + 8 - a6) >> 3;
  if ( v23 + 8 - a6 > 0 )
  {
    v82 = (unsigned int *)&v23[-8 * result];
    v83 = &v9[-2 * result];
    do
    {
      v83[2 * result - 2] = v82[2 * result];
      v83[2 * result - 1] = v82[2 * result + 1];
      --result;
    }
    while ( result );
  }
  return result;
}
