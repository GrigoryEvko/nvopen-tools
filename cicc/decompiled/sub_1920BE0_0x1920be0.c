// Function: sub_1920BE0
// Address: 0x1920be0
//
unsigned __int64 __fastcall sub_1920BE0(char *a1, char *a2, __int64 a3, __int64 a4, __int64 a5, char *a6, __int64 a7)
{
  __int64 v7; // rax
  char *v8; // r10
  char *v10; // r12
  __int64 v11; // rbx
  __int64 v12; // r14
  __int64 v13; // r15
  char *v14; // r8
  char *v15; // r10
  char *v16; // rax
  __int64 v17; // r10
  char *v18; // r11
  char *v19; // r8
  __int64 v20; // rcx
  __int64 v21; // rbx
  unsigned __int64 v22; // rax
  __int64 v23; // rdx
  char *v24; // rax
  unsigned __int64 v25; // rdx
  unsigned __int64 v26; // rsi
  char *v27; // r9
  char *v28; // rax
  int v29; // edx
  __int64 v30; // r9
  char *v31; // rax
  signed __int64 v32; // rdi
  char *v33; // rsi
  int v34; // eax
  char *v35; // rdx
  signed __int64 v36; // rsi
  int v37; // r8d
  __int64 v38; // rax
  unsigned __int64 result; // rax
  unsigned __int64 v40; // rcx
  char *v41; // rdx
  int v42; // esi
  __int64 v43; // rcx
  char *v44; // rcx
  int v45; // eax
  int v46; // eax
  int v47; // edx
  int v48; // eax
  __int64 v49; // rcx
  int v50; // edx
  __int64 v51; // r8
  unsigned __int64 v52; // rcx
  char *v53; // rdx
  char *v54; // rax
  int v55; // esi
  char *v56; // r10
  __int64 i; // rbx
  int v58; // ecx
  int v59; // edx
  __int64 v60; // rcx
  __int64 v61; // rdi
  unsigned __int64 v62; // rdx
  unsigned __int64 v63; // rsi
  char *v64; // r9
  char *v65; // rax
  int v66; // edx
  signed __int64 v67; // rsi
  char *v68; // rax
  int v69; // r9d
  char *v70; // rdx
  char *v71; // rax
  signed __int64 v72; // rdi
  int v73; // r8d
  __int64 v74; // rax
  unsigned __int64 v75; // rdx
  int v76; // esi
  unsigned __int64 v77; // rdx
  int v78; // [rsp+8h] [rbp-58h]
  int v79; // [rsp+10h] [rbp-50h]
  char *v81; // [rsp+20h] [rbp-40h]
  char *v82; // [rsp+20h] [rbp-40h]
  char *v83; // [rsp+20h] [rbp-40h]
  __int64 v84; // [rsp+20h] [rbp-40h]
  char *v85; // [rsp+20h] [rbp-40h]
  __int64 v86; // [rsp+20h] [rbp-40h]
  int v87; // [rsp+20h] [rbp-40h]
  char *v88; // [rsp+28h] [rbp-38h]

  v7 = a5;
  v8 = a2;
  v10 = a1;
  v11 = a3;
  if ( a7 <= a5 )
    v7 = a7;
  if ( a4 > v7 )
  {
    v12 = a5;
    if ( a7 >= a5 )
    {
LABEL_51:
      result = 0xAAAAAAAAAAAAAAABLL;
      v51 = v11 - (_QWORD)v8;
      v52 = 0xAAAAAAAAAAAAAAABLL * ((v11 - (__int64)v8) >> 3);
      if ( v11 - (__int64)v8 <= 0 )
        return result;
      v53 = a6;
      v54 = v8;
      do
      {
        v55 = *(_DWORD *)v54;
        v53 += 24;
        v54 += 24;
        *((_DWORD *)v53 - 6) = v55;
        *((_DWORD *)v53 - 5) = *((_DWORD *)v54 - 5);
        *((_QWORD *)v53 - 2) = *((_QWORD *)v54 - 2);
        *((_QWORD *)v53 - 1) = *((_QWORD *)v54 - 1);
        --v52;
      }
      while ( v52 );
      if ( v51 <= 0 )
        v51 = 24;
      result = (unsigned __int64)&a6[v51];
      if ( v10 == v8 )
      {
        v77 = 0xAAAAAAAAAAAAAAABLL * (v51 >> 3);
        while ( 1 )
        {
          result -= 24LL;
          *(_DWORD *)(v11 - 24) = v55;
          v11 -= 24;
          *(_DWORD *)(v11 + 4) = *(_DWORD *)(result + 4);
          *(_QWORD *)(v11 + 8) = *(_QWORD *)(result + 8);
          *(_QWORD *)(v11 + 16) = *(_QWORD *)(result + 16);
          if ( !--v77 )
            break;
          v55 = *(_DWORD *)(result - 24);
        }
        return result;
      }
      if ( a6 == (char *)result )
        return result;
      v56 = v8 - 24;
      result -= 24LL;
      for ( i = v11 - 24; ; i -= 24 )
      {
        v58 = *(_DWORD *)result;
        v59 = *(_DWORD *)v56;
        if ( *(_DWORD *)result < *(_DWORD *)v56 || v58 == v59 && *(_DWORD *)(result + 4) < *((_DWORD *)v56 + 1) )
        {
          *(_DWORD *)i = v59;
          v60 = i;
          *(_DWORD *)(i + 4) = *((_DWORD *)v56 + 1);
          *(_QWORD *)(i + 8) = *((_QWORD *)v56 + 1);
          *(_QWORD *)(i + 16) = *((_QWORD *)v56 + 2);
          if ( v56 == v10 )
          {
            result += 24LL;
            v75 = 0xAAAAAAAAAAAAAAABLL * ((__int64)(result - (_QWORD)a6) >> 3);
            if ( (__int64)(result - (_QWORD)a6) > 0 )
            {
              do
              {
                v76 = *(_DWORD *)(result - 24);
                result -= 24LL;
                v60 -= 24;
                *(_DWORD *)v60 = v76;
                *(_DWORD *)(v60 + 4) = *(_DWORD *)(result + 4);
                *(_QWORD *)(v60 + 8) = *(_QWORD *)(result + 8);
                *(_QWORD *)(v60 + 16) = *(_QWORD *)(result + 16);
                --v75;
              }
              while ( v75 );
            }
            return result;
          }
          v56 -= 24;
        }
        else
        {
          *(_DWORD *)i = v58;
          *(_DWORD *)(i + 4) = *(_DWORD *)(result + 4);
          *(_QWORD *)(i + 8) = *(_QWORD *)(result + 8);
          *(_QWORD *)(i + 16) = *(_QWORD *)(result + 16);
          if ( a6 == (char *)result )
            return result;
          result -= 24LL;
        }
      }
    }
    v13 = a4;
    v14 = a2;
    v15 = a1;
    if ( v12 >= a4 )
      goto LABEL_14;
LABEL_6:
    v81 = v14;
    v16 = (char *)sub_1920010(
                    v14,
                    a3,
                    (unsigned int *)&v15[8 * (v13 / 2)
                                       + 8 * ((v13 + ((unsigned __int64)v13 >> 63)) & 0xFFFFFFFFFFFFFFFELL)]);
    v19 = v81;
    v20 = v13 / 2;
    v88 = v16;
    v13 -= v13 / 2;
    v21 = 0xAAAAAAAAAAAAAAABLL * ((v16 - v81) >> 3);
    if ( v13 <= v21 )
      goto LABEL_15;
LABEL_7:
    if ( a7 >= v21 )
    {
      v22 = (unsigned __int64)v18;
      if ( !v21 )
        goto LABEL_9;
      v61 = v88 - v19;
      v86 = v19 - v18;
      v62 = 0xAAAAAAAAAAAAAAABLL * ((v19 - v18) >> 3);
      v63 = 0xAAAAAAAAAAAAAAABLL * ((v88 - v19) >> 3);
      if ( v88 - v19 <= 0 )
      {
        if ( v86 <= 0 )
          goto LABEL_9;
        v67 = 0;
        v61 = 0;
      }
      else
      {
        v64 = a6;
        v65 = v19;
        do
        {
          v66 = *(_DWORD *)v65;
          v64 += 24;
          v65 += 24;
          *((_DWORD *)v64 - 6) = v66;
          *((_DWORD *)v64 - 5) = *((_DWORD *)v65 - 5);
          *((_QWORD *)v64 - 2) = *((_QWORD *)v65 - 2);
          *((_QWORD *)v64 - 1) = *((_QWORD *)v65 - 1);
          --v63;
        }
        while ( v63 );
        v62 = 0xAAAAAAAAAAAAAAABLL * ((v19 - v18) >> 3);
        if ( v61 <= 0 )
          v61 = 24;
        v67 = 0xAAAAAAAAAAAAAAABLL * (v61 >> 3);
        if ( v86 <= 0 )
          goto LABEL_75;
      }
      v68 = v88;
      do
      {
        v69 = *((_DWORD *)v19 - 6);
        v19 -= 24;
        v68 -= 24;
        *(_DWORD *)v68 = v69;
        *((_DWORD *)v68 + 1) = *((_DWORD *)v19 + 1);
        *((_QWORD *)v68 + 1) = *((_QWORD *)v19 + 1);
        *((_QWORD *)v68 + 2) = *((_QWORD *)v19 + 2);
        --v62;
      }
      while ( v62 );
LABEL_75:
      if ( v61 <= 0 )
      {
        v22 = (unsigned __int64)v18;
      }
      else
      {
        v70 = v18;
        v71 = a6;
        v72 = v67;
        do
        {
          v73 = *(_DWORD *)v71;
          v70 += 24;
          v71 += 24;
          *((_DWORD *)v70 - 6) = v73;
          *((_DWORD *)v70 - 5) = *((_DWORD *)v71 - 5);
          *((_QWORD *)v70 - 2) = *((_QWORD *)v71 - 2);
          *((_QWORD *)v70 - 1) = *((_QWORD *)v71 - 1);
          --v72;
        }
        while ( v72 );
        v74 = 24 * v67;
        if ( v67 <= 0 )
          v74 = 24;
        v22 = (unsigned __int64)&v18[v74];
      }
      goto LABEL_9;
    }
    while ( 1 )
    {
LABEL_15:
      if ( a7 < v13 )
      {
        v78 = v17;
        v79 = v20;
        v87 = (int)v18;
        v22 = sub_191FBD0(v18, v19, v88);
        LODWORD(v18) = v87;
        LODWORD(v20) = v79;
        LODWORD(v17) = v78;
        goto LABEL_9;
      }
      v22 = (unsigned __int64)v88;
      if ( !v13 )
        goto LABEL_9;
      v84 = v88 - v19;
      v25 = 0xAAAAAAAAAAAAAAABLL * ((v88 - v19) >> 3);
      v26 = 0xAAAAAAAAAAAAAAABLL * ((v19 - v18) >> 3);
      if ( v19 - v18 <= 0 )
      {
        if ( v84 <= 0 )
          goto LABEL_9;
        v31 = a6;
        v32 = 0;
        v30 = 0;
      }
      else
      {
        v27 = a6;
        v28 = v18;
        do
        {
          v29 = *(_DWORD *)v28;
          v27 += 24;
          v28 += 24;
          *((_DWORD *)v27 - 6) = v29;
          *((_DWORD *)v27 - 5) = *((_DWORD *)v28 - 5);
          *((_QWORD *)v27 - 2) = *((_QWORD *)v28 - 2);
          *((_QWORD *)v27 - 1) = *((_QWORD *)v28 - 1);
          --v26;
        }
        while ( v26 );
        v30 = 24;
        v25 = 0xAAAAAAAAAAAAAAABLL * ((v88 - v19) >> 3);
        if ( v19 - v18 > 0 )
          v30 = v19 - v18;
        v31 = &a6[v30];
        v32 = 0xAAAAAAAAAAAAAAABLL * (v30 >> 3);
        if ( v84 <= 0 )
          goto LABEL_26;
      }
      v85 = v31;
      v33 = v18;
      do
      {
        v34 = *(_DWORD *)v19;
        v33 += 24;
        v19 += 24;
        *((_DWORD *)v33 - 6) = v34;
        *((_DWORD *)v33 - 5) = *((_DWORD *)v19 - 5);
        *((_QWORD *)v33 - 2) = *((_QWORD *)v19 - 2);
        *((_QWORD *)v33 - 1) = *((_QWORD *)v19 - 1);
        --v25;
      }
      while ( v25 );
      v31 = v85;
LABEL_26:
      if ( v30 <= 0 )
      {
        v22 = (unsigned __int64)v88;
      }
      else
      {
        v35 = v88;
        v36 = v32;
        do
        {
          v37 = *((_DWORD *)v31 - 6);
          v31 -= 24;
          v35 -= 24;
          *(_DWORD *)v35 = v37;
          *((_DWORD *)v35 + 1) = *((_DWORD *)v31 + 1);
          *((_QWORD *)v35 + 1) = *((_QWORD *)v31 + 1);
          *((_QWORD *)v35 + 2) = *((_QWORD *)v31 + 2);
          --v36;
        }
        while ( v36 );
        v38 = -24 * v32;
        if ( v32 <= 0 )
          v38 = -24;
        v22 = (unsigned __int64)&v88[v38];
      }
LABEL_9:
      v12 -= v21;
      v82 = (char *)v22;
      sub_1920BE0(v17, (_DWORD)v18, v22, v20, v21, (_DWORD)a6, a7);
      v23 = v12;
      if ( a7 <= v12 )
        v23 = a7;
      if ( v23 >= v13 )
      {
        v11 = a3;
        v8 = v88;
        v10 = v82;
        break;
      }
      if ( a7 >= v12 )
      {
        v11 = a3;
        v8 = v88;
        v10 = v82;
        goto LABEL_51;
      }
      v14 = v88;
      v15 = v82;
      if ( v12 < v13 )
        goto LABEL_6;
LABEL_14:
      v83 = v14;
      v21 = v12 / 2;
      v88 = &v14[8 * (v12 / 2) + 8 * ((v12 + ((unsigned __int64)v12 >> 63)) & 0xFFFFFFFFFFFFFFFELL)];
      v24 = (char *)sub_1920080(v15, (__int64)v14, (unsigned int *)v88);
      v19 = v83;
      v18 = v24;
      v20 = 0xAAAAAAAAAAAAAAABLL * ((__int64)&v24[-v17] >> 3);
      v13 -= v20;
      if ( v13 > v12 / 2 )
        goto LABEL_7;
    }
  }
  result = 0xAAAAAAAAAAAAAAABLL;
  v40 = 0xAAAAAAAAAAAAAAABLL * ((v8 - v10) >> 3);
  if ( v8 - v10 <= 0 )
    return result;
  v41 = a6;
  result = (unsigned __int64)v10;
  do
  {
    v42 = *(_DWORD *)result;
    v41 += 24;
    result += 24LL;
    *((_DWORD *)v41 - 6) = v42;
    *((_DWORD *)v41 - 5) = *(_DWORD *)(result - 20);
    *((_QWORD *)v41 - 2) = *(_QWORD *)(result - 16);
    *((_QWORD *)v41 - 1) = *(_QWORD *)(result - 8);
    --v40;
  }
  while ( v40 );
  v43 = 24;
  if ( v8 - v10 > 0 )
    v43 = v8 - v10;
  v44 = &a6[v43];
  if ( a6 != v44 )
  {
    while ( (char *)v11 != v8 )
    {
      v46 = *(_DWORD *)v8;
      v47 = *(_DWORD *)a6;
      if ( *(_DWORD *)v8 < *(_DWORD *)a6 || v46 == v47 && *((_DWORD *)v8 + 1) < *((_DWORD *)a6 + 1) )
      {
        *(_DWORD *)v10 = v46;
        v48 = *((_DWORD *)v8 + 1);
        v10 += 24;
        v8 += 24;
        *((_DWORD *)v10 - 5) = v48;
        *((_QWORD *)v10 - 2) = *((_QWORD *)v8 - 2);
        result = *((_QWORD *)v8 - 1);
        *((_QWORD *)v10 - 1) = result;
        if ( a6 == v44 )
          break;
      }
      else
      {
        *(_DWORD *)v10 = v47;
        v45 = *((_DWORD *)a6 + 1);
        a6 += 24;
        v10 += 24;
        *((_DWORD *)v10 - 5) = v45;
        *((_QWORD *)v10 - 2) = *((_QWORD *)a6 - 2);
        result = *((_QWORD *)a6 - 1);
        *((_QWORD *)v10 - 1) = result;
        if ( a6 == v44 )
          break;
      }
    }
  }
  if ( v44 != a6 )
  {
    v49 = v44 - a6;
    result = 0xAAAAAAAAAAAAAAABLL * (v49 >> 3);
    if ( v49 > 0 )
    {
      do
      {
        v50 = *(_DWORD *)a6;
        v10 += 24;
        a6 += 24;
        *((_DWORD *)v10 - 6) = v50;
        *((_DWORD *)v10 - 5) = *((_DWORD *)a6 - 5);
        *((_QWORD *)v10 - 2) = *((_QWORD *)a6 - 2);
        *((_QWORD *)v10 - 1) = *((_QWORD *)a6 - 1);
        --result;
      }
      while ( result );
    }
  }
  return result;
}
