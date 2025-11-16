// Function: sub_25FF3B0
// Address: 0x25ff3b0
//
__int64 __fastcall sub_25FF3B0(char *a1, unsigned int *a2, __int64 a3, __int64 a4, __int64 a5, _DWORD *a6, __int64 a7)
{
  __int64 v7; // rax
  unsigned int *v8; // r13
  __int64 v9; // r12
  __int64 v10; // rbx
  unsigned int *v11; // r14
  __int64 v12; // r10
  __int64 *v13; // r10
  int v14; // ecx
  __int64 result; // rax
  __int64 v16; // r15
  unsigned __int64 v17; // r13
  __int64 v18; // r12
  unsigned int *v19; // rbx
  __int64 v20; // rsi
  __int64 v21; // rdi
  __int64 v22; // rdx
  _DWORD *v23; // r13
  char *v24; // r14
  char *v25; // r14
  unsigned __int64 v26; // r15
  __int64 v27; // rsi
  __int64 v28; // rdi
  __int64 v29; // rcx
  char *v30; // rbx
  char *v31; // r15
  __int64 v32; // rsi
  unsigned __int64 v33; // r14
  unsigned int *v34; // r12
  __int64 v35; // rsi
  __int64 v36; // rdi
  __int64 v37; // rax
  __int64 v38; // r14
  unsigned int *v39; // rbx
  unsigned __int64 v40; // r13
  unsigned int *v41; // r12
  __int64 v42; // rsi
  __int64 v43; // rdi
  __int64 v44; // rbx
  signed __int64 v45; // r13
  __int64 v46; // r10
  unsigned __int64 v47; // r14
  __int64 v48; // r12
  unsigned int *v49; // rbx
  __int64 v50; // rsi
  __int64 v51; // rdi
  __int64 v52; // rax
  __int64 v53; // rbx
  unsigned __int64 v54; // r14
  __int64 v55; // r13
  unsigned int *v56; // rbx
  signed __int64 v57; // r14
  __int64 v58; // rsi
  __int64 v59; // rdi
  __int64 v60; // r10
  __int64 v61; // r15
  __int64 v62; // r14
  unsigned __int64 v63; // r12
  __int64 v64; // rsi
  __int64 v65; // rdi
  __int64 v66; // r13
  unsigned __int64 v67; // rbx
  unsigned __int64 v68; // rbx
  __int64 v69; // [rsp+0h] [rbp-90h]
  __int64 v70; // [rsp+0h] [rbp-90h]
  __int64 v71; // [rsp+0h] [rbp-90h]
  unsigned __int64 v72; // [rsp+8h] [rbp-88h]
  unsigned __int64 v73; // [rsp+8h] [rbp-88h]
  int v74; // [rsp+10h] [rbp-80h]
  __int64 v75; // [rsp+10h] [rbp-80h]
  int v76; // [rsp+10h] [rbp-80h]
  __int64 v77; // [rsp+10h] [rbp-80h]
  __int64 v78; // [rsp+18h] [rbp-78h]
  int v79; // [rsp+18h] [rbp-78h]
  __int64 v80; // [rsp+18h] [rbp-78h]
  __int64 v81; // [rsp+18h] [rbp-78h]
  __int64 v82; // [rsp+18h] [rbp-78h]
  __int64 v83; // [rsp+18h] [rbp-78h]
  __int64 v84; // [rsp+20h] [rbp-70h]
  signed __int64 v85; // [rsp+20h] [rbp-70h]
  __int64 v86; // [rsp+20h] [rbp-70h]
  signed __int64 v87; // [rsp+20h] [rbp-70h]
  __int64 v88; // [rsp+28h] [rbp-68h]
  __int64 v89; // [rsp+30h] [rbp-60h]
  __int64 v90; // [rsp+38h] [rbp-58h]
  __int64 *v91; // [rsp+38h] [rbp-58h]
  unsigned int *v92; // [rsp+40h] [rbp-50h]
  unsigned int *v93; // [rsp+48h] [rbp-48h]
  _DWORD *v94; // [rsp+48h] [rbp-48h]
  char *v95; // [rsp+48h] [rbp-48h]
  unsigned int *v96; // [rsp+50h] [rbp-40h]
  char *v97; // [rsp+50h] [rbp-40h]
  __int64 v98; // [rsp+58h] [rbp-38h]
  __int64 v99; // [rsp+58h] [rbp-38h]
  __int64 v100; // [rsp+58h] [rbp-38h]

  while ( 1 )
  {
    v7 = a5;
    v8 = a2;
    v9 = (__int64)a1;
    v10 = (__int64)a6;
    v96 = (unsigned int *)a3;
    v98 = a5;
    if ( a5 > a7 )
      v7 = a7;
    if ( v7 >= a4 )
      break;
    v11 = a2;
    if ( a5 <= a7 )
    {
      result = 0x86BCA1AF286BCA1BLL;
      v16 = a3;
      v99 = a3 - (_QWORD)a2;
      v17 = 0x86BCA1AF286BCA1BLL * ((a3 - (__int64)a2) >> 3);
      if ( a3 - (__int64)a2 > 0 )
      {
        v97 = a1;
        v18 = (__int64)a6;
        v19 = a2;
        v94 = a6;
        do
        {
          v20 = (__int64)v19;
          v21 = v18;
          v19 += 38;
          v18 += 152;
          sub_25F6310(v21, v20);
          --v17;
        }
        while ( v17 );
        result = 152;
        v22 = v99;
        if ( v99 <= 0 )
          v22 = 152;
        v23 = (_DWORD *)((char *)v94 + v22);
        if ( v97 == (char *)v11 )
        {
          result = 0x86BCA1AF286BCA1BLL;
          v68 = 0x86BCA1AF286BCA1BLL * (v22 >> 3);
          if ( v22 > 0 )
          {
            do
            {
              v23 -= 38;
              v16 -= 152;
              result = sub_25F6310(v16, (__int64)v23);
              --v68;
            }
            while ( v68 );
          }
        }
        else if ( v94 != v23 )
        {
          v24 = (char *)(v11 - 38);
          while ( 1 )
          {
            v23 -= 38;
            v16 -= 152;
            if ( *v23 < *(_DWORD *)v24 )
              break;
LABEL_22:
            result = sub_25F6310(v16, (__int64)v23);
            if ( v94 == v23 )
              return result;
          }
          while ( 1 )
          {
            sub_25F6310(v16, (__int64)v24);
            if ( v24 == v97 )
              break;
            v24 -= 152;
            v16 -= 152;
            if ( *v23 >= *(_DWORD *)v24 )
              goto LABEL_22;
          }
          v66 = (__int64)(v23 + 38);
          result = v66 - (_QWORD)v94;
          v67 = 0x86BCA1AF286BCA1BLL * ((v66 - (__int64)v94) >> 3);
          if ( v66 - (__int64)v94 > 0 )
          {
            do
            {
              v66 -= 152;
              v16 -= 152;
              result = sub_25F6310(v16, v66);
              --v67;
            }
            while ( v67 );
          }
        }
      }
      return result;
    }
    if ( a5 >= a4 )
    {
      v89 = a5 / 2;
      v92 = &a2[38 * (a5 / 2)];
      v93 = sub_25F6810(a1, (__int64)a2, v92);
      v90 = 0x86BCA1AF286BCA1BLL * (((char *)v93 - a1) >> 3);
    }
    else
    {
      v90 = a4 / 2;
      v93 = (unsigned int *)&a1[152 * (a4 / 2)];
      v92 = sub_25F67A0(a2, a3, v93);
      v89 = 0x86BCA1AF286BCA1BLL * (((char *)v92 - (char *)a2) >> 3);
    }
    v88 = v12 - v90;
    if ( v12 - v90 > v89 && a7 >= v89 )
    {
      v13 = (__int64 *)v93;
      if ( !v89 )
        goto LABEL_10;
      v86 = (char *)v92 - (char *)a2;
      v81 = (char *)a2 - (char *)v93;
      v73 = 0x86BCA1AF286BCA1BLL * (((char *)a2 - (char *)v93) >> 3);
      if ( (char *)v92 - (char *)a2 <= 0 )
      {
        if ( v81 <= 0 )
          goto LABEL_10;
        v87 = 0;
        v77 = 0;
LABEL_60:
        v82 = v10;
        v53 = (__int64)v92;
        v54 = v73;
        do
        {
          v8 -= 38;
          v53 -= 152;
          sub_25F6310(v53, (__int64)v8);
          --v54;
        }
        while ( v54 );
        v10 = v82;
      }
      else
      {
        v76 = (int)a1;
        v47 = 0x86BCA1AF286BCA1BLL * (((char *)v92 - (char *)a2) >> 3);
        v48 = v10;
        v71 = v10;
        v49 = a2;
        do
        {
          v50 = (__int64)v49;
          v51 = v48;
          v49 += 38;
          v48 += 152;
          sub_25F6310(v51, v50);
          --v47;
        }
        while ( v47 );
        v52 = 152;
        LODWORD(v9) = v76;
        v10 = v71;
        if ( v86 > 0 )
          v52 = v86;
        v77 = v52;
        v87 = 0x86BCA1AF286BCA1BLL * (v52 >> 3);
        if ( v81 > 0 )
          goto LABEL_60;
      }
      if ( v77 <= 0 )
      {
        v13 = (__int64 *)v93;
      }
      else
      {
        v83 = v10;
        v55 = v10;
        v56 = v93;
        v57 = v87;
        do
        {
          v58 = v55;
          v59 = (__int64)v56;
          v55 += 152;
          v56 += 38;
          sub_25F6310(v59, v58);
          --v57;
        }
        while ( v57 );
        v10 = v83;
        v60 = 38 * v87;
        if ( v87 <= 0 )
          v60 = 38;
        v13 = (__int64 *)&v93[v60];
      }
      goto LABEL_10;
    }
    if ( a7 < v88 )
    {
      v13 = sub_25FED10((__int64 *)v93, (__int64 *)a2, (__int64 *)v92);
      goto LABEL_10;
    }
    v13 = (__int64 *)v92;
    if ( !v88 )
      goto LABEL_10;
    v84 = (char *)a2 - (char *)v93;
    v78 = (char *)v92 - (char *)a2;
    v72 = 0x86BCA1AF286BCA1BLL * (((char *)v92 - (char *)a2) >> 3);
    if ( (char *)a2 - (char *)v93 <= 0 )
    {
      if ( v78 <= 0 )
        goto LABEL_10;
      v85 = 0;
      v38 = v10;
      v75 = 0;
    }
    else
    {
      v69 = v10;
      v33 = 0x86BCA1AF286BCA1BLL * (((char *)a2 - (char *)v93) >> 3);
      v74 = (int)a1;
      v34 = v93;
      do
      {
        v35 = (__int64)v34;
        v36 = v10;
        v34 += 38;
        v10 += 152;
        sub_25F6310(v36, v35);
        --v33;
      }
      while ( v33 );
      v37 = 152;
      v10 = v69;
      LODWORD(v9) = v74;
      if ( v84 > 0 )
        v37 = v84;
      v75 = v37;
      v38 = v69 + v37;
      v85 = 0x86BCA1AF286BCA1BLL * (v37 >> 3);
      if ( v78 <= 0 )
        goto LABEL_47;
    }
    v70 = v10;
    v39 = v8;
    v40 = v72;
    v79 = v9;
    v41 = v93;
    do
    {
      v42 = (__int64)v39;
      v43 = (__int64)v41;
      v39 += 38;
      v41 += 38;
      sub_25F6310(v43, v42);
      --v40;
    }
    while ( v40 );
    LODWORD(v9) = v79;
    v10 = v70;
LABEL_47:
    if ( v75 <= 0 )
    {
      v13 = (__int64 *)v92;
    }
    else
    {
      v80 = v10;
      v44 = (__int64)v92;
      v45 = v85;
      do
      {
        v38 -= 152;
        v44 -= 152;
        sub_25F6310(v44, v38);
        --v45;
      }
      while ( v45 );
      v10 = v80;
      v46 = -38 * v85;
      if ( v85 <= 0 )
        v46 = 0x3FFFFFFFFFFFFFDALL;
      v13 = (__int64 *)&v92[v46];
    }
LABEL_10:
    v14 = v90;
    v91 = v13;
    sub_25FF3B0(v9, (_DWORD)v93, (_DWORD)v13, v14, v89, v10, a7);
    a4 = v88;
    a6 = (_DWORD *)v10;
    a2 = v92;
    a3 = (__int64)v96;
    a5 = v98 - v89;
    a1 = (char *)v91;
  }
  v100 = (char *)a2 - a1;
  result = 0x86BCA1AF286BCA1BLL * (((char *)a2 - a1) >> 3);
  v25 = a1;
  if ( (char *)a2 - a1 > 0 )
  {
    v95 = (char *)a6;
    v26 = 0x86BCA1AF286BCA1BLL * (((char *)a2 - a1) >> 3);
    do
    {
      v27 = (__int64)v25;
      v28 = v10;
      v25 += 152;
      v10 += 152;
      sub_25F6310(v28, v27);
      --v26;
    }
    while ( v26 );
    v29 = v100;
    result = 152;
    v30 = v95;
    if ( v100 <= 0 )
      v29 = 152;
    v31 = &v95[v29];
    if ( v95 != &v95[v29] )
    {
      while ( v8 != v96 )
      {
        if ( *v8 < *(_DWORD *)v30 )
        {
          v32 = (__int64)v8;
          v8 += 38;
        }
        else
        {
          v32 = (__int64)v30;
          v30 += 152;
        }
        result = sub_25F6310(v9, v32);
        v9 += 152;
        if ( v30 == v31 )
          return result;
      }
      result = 0x86BCA1AF286BCA1BLL;
      v61 = v31 - v30;
      v62 = v9;
      v63 = 0x86BCA1AF286BCA1BLL * (v61 >> 3);
      if ( v61 > 0 )
      {
        do
        {
          v64 = (__int64)v30;
          v65 = v62;
          v30 += 152;
          v62 += 152;
          result = sub_25F6310(v65, v64);
          --v63;
        }
        while ( v63 );
      }
    }
  }
  return result;
}
