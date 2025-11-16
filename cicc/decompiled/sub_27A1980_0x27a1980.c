// Function: sub_27A1980
// Address: 0x27a1980
//
__int64 __fastcall sub_27A1980(char *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, char *a6, __int64 a7)
{
  __int64 result; // rax
  char *v8; // r10
  char *v9; // r13
  char *v10; // r12
  char *v11; // rbx
  __int64 v12; // r15
  __int64 v13; // r13
  __int64 v14; // r11
  __int64 v15; // r10
  unsigned int *v16; // r12
  __int64 v17; // rax
  __int64 v18; // r10
  __int64 v19; // r11
  __int64 v20; // rcx
  __int64 v21; // rbx
  char *v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rdi
  __int64 v25; // r9
  __int64 v26; // rdx
  __int64 v27; // rsi
  char *v28; // rdi
  unsigned int *v29; // rax
  unsigned int v30; // edx
  char *v31; // rax
  __int64 v32; // rdi
  unsigned int *v33; // rsi
  unsigned int v34; // r8d
  char *v35; // rdx
  __int64 v36; // rsi
  int v37; // r8d
  __int64 v38; // rax
  __int64 v39; // rcx
  char *v40; // rdx
  int v41; // edi
  __int64 v42; // rdx
  char *v43; // rdx
  __int64 v44; // rax
  int v45; // eax
  int v46; // ecx
  __int64 v47; // rax
  __int64 v48; // rdx
  int v49; // edx
  __int64 v50; // r8
  __int64 v51; // rcx
  char *v52; // rdx
  char *v53; // rax
  int v54; // esi
  char *v55; // r10
  char *i; // rbx
  int v57; // ecx
  int v58; // edx
  char *v59; // rcx
  __int64 v60; // r8
  __int64 v61; // rdi
  __int64 v62; // rdx
  __int64 v63; // rsi
  char *v64; // r9
  _DWORD *v65; // rax
  int v66; // edx
  __int64 v67; // rsi
  char *v68; // rax
  int v69; // r8d
  unsigned int *v70; // rax
  unsigned int *v71; // rdx
  __int64 v72; // rdi
  unsigned int v73; // r8d
  __int64 v74; // rax
  __int64 v75; // rdx
  int v76; // esi
  __int64 v77; // rdx
  int v78; // [rsp+8h] [rbp-58h]
  char *v80; // [rsp+18h] [rbp-48h]
  __int64 v81; // [rsp+18h] [rbp-48h]
  int v82; // [rsp+18h] [rbp-48h]
  char *v84; // [rsp+28h] [rbp-38h]

  result = a5;
  v8 = (char *)a2;
  v9 = a6;
  v10 = a1;
  v11 = (char *)a3;
  if ( a7 <= a5 )
    result = a7;
  if ( a4 > result )
  {
    if ( a7 >= a5 )
    {
LABEL_50:
      v50 = v11 - v8;
      v51 = (v11 - v8) >> 5;
      if ( v11 - v8 <= 0 )
        return result;
      v52 = v9;
      v53 = v8;
      do
      {
        v54 = *(_DWORD *)v53;
        v52 += 32;
        v53 += 32;
        *((_DWORD *)v52 - 8) = v54;
        *((_QWORD *)v52 - 3) = *((_QWORD *)v53 - 3);
        *((_QWORD *)v52 - 2) = *((_QWORD *)v53 - 2);
        *((_QWORD *)v52 - 1) = *((_QWORD *)v53 - 1);
        --v51;
      }
      while ( v51 );
      if ( v50 <= 0 )
        v50 = 32;
      result = (__int64)&v9[v50];
      if ( v10 == v8 )
      {
        v77 = v50 >> 5;
        while ( 1 )
        {
          result -= 32;
          *((_DWORD *)v11 - 8) = v54;
          v11 -= 32;
          *((_QWORD *)v11 + 1) = *(_QWORD *)(result + 8);
          *((_QWORD *)v11 + 2) = *(_QWORD *)(result + 16);
          *((_QWORD *)v11 + 3) = *(_QWORD *)(result + 24);
          if ( !--v77 )
            break;
          v54 = *(_DWORD *)(result - 32);
        }
        return result;
      }
      if ( v9 == (char *)result )
        return result;
      v55 = v8 - 32;
      result -= 32;
      for ( i = v11 - 32; ; i -= 32 )
      {
        v57 = *(_DWORD *)result;
        v58 = *(_DWORD *)v55;
        if ( *(_DWORD *)result < *(_DWORD *)v55 || v57 == v58 && *(_QWORD *)(result + 8) < *((_QWORD *)v55 + 1) )
        {
          *(_DWORD *)i = v58;
          v59 = i;
          *((_QWORD *)i + 1) = *((_QWORD *)v55 + 1);
          *((_QWORD *)i + 2) = *((_QWORD *)v55 + 2);
          *((_QWORD *)i + 3) = *((_QWORD *)v55 + 3);
          if ( v55 == v10 )
          {
            result += 32;
            v75 = (result - (__int64)v9) >> 5;
            if ( result - (__int64)v9 > 0 )
            {
              do
              {
                v76 = *(_DWORD *)(result - 32);
                result -= 32;
                v59 -= 32;
                *(_DWORD *)v59 = v76;
                *((_QWORD *)v59 + 1) = *(_QWORD *)(result + 8);
                *((_QWORD *)v59 + 2) = *(_QWORD *)(result + 16);
                *((_QWORD *)v59 + 3) = *(_QWORD *)(result + 24);
                --v75;
              }
              while ( v75 );
            }
            return result;
          }
          v55 -= 32;
        }
        else
        {
          *(_DWORD *)i = v57;
          *((_QWORD *)i + 1) = *(_QWORD *)(result + 8);
          *((_QWORD *)i + 2) = *(_QWORD *)(result + 16);
          *((_QWORD *)i + 3) = *(_QWORD *)(result + 24);
          if ( v9 == (char *)result )
            return result;
          result -= 32;
        }
      }
    }
    v12 = a4;
    v13 = a5;
    v14 = a2;
    v15 = (__int64)a1;
    if ( a5 >= a4 )
      goto LABEL_14;
LABEL_6:
    v16 = (unsigned int *)(v15 + 32 * (v12 / 2));
    v17 = sub_27A11C0(v14, a3, v16);
    v20 = v12 / 2;
    v84 = (char *)v17;
    v12 -= v12 / 2;
    v21 = (v17 - v19) >> 5;
    if ( v12 <= v21 )
      goto LABEL_15;
LABEL_7:
    if ( a7 >= v21 )
    {
      v22 = (char *)v16;
      if ( !v21 )
        goto LABEL_9;
      v60 = v19 - (_QWORD)v16;
      v61 = (__int64)&v84[-v19];
      v62 = (v19 - (__int64)v16) >> 5;
      v63 = (__int64)&v84[-v19] >> 5;
      if ( (__int64)&v84[-v19] <= 0 )
      {
        if ( v60 <= 0 )
          goto LABEL_9;
        v67 = 0;
        v61 = 0;
      }
      else
      {
        v64 = a6;
        v65 = (_DWORD *)v19;
        do
        {
          v66 = *v65;
          v64 += 32;
          v65 += 8;
          *((_DWORD *)v64 - 8) = v66;
          *((_QWORD *)v64 - 3) = *((_QWORD *)v65 - 3);
          *((_QWORD *)v64 - 2) = *((_QWORD *)v65 - 2);
          *((_QWORD *)v64 - 1) = *((_QWORD *)v65 - 1);
          --v63;
        }
        while ( v63 );
        v62 = (v19 - (__int64)v16) >> 5;
        if ( v61 <= 0 )
          v61 = 32;
        v67 = v61 >> 5;
        if ( v60 <= 0 )
          goto LABEL_74;
      }
      v68 = v84;
      do
      {
        v69 = *(_DWORD *)(v19 - 32);
        v19 -= 32;
        v68 -= 32;
        *(_DWORD *)v68 = v69;
        *((_QWORD *)v68 + 1) = *(_QWORD *)(v19 + 8);
        *((_QWORD *)v68 + 2) = *(_QWORD *)(v19 + 16);
        *((_QWORD *)v68 + 3) = *(_QWORD *)(v19 + 24);
        --v62;
      }
      while ( v62 );
LABEL_74:
      if ( v61 <= 0 )
      {
        v22 = (char *)v16;
      }
      else
      {
        v70 = (unsigned int *)a6;
        v71 = v16;
        v72 = v67;
        do
        {
          v73 = *v70;
          v71 += 8;
          v70 += 8;
          *(v71 - 8) = v73;
          *((_QWORD *)v71 - 3) = *((_QWORD *)v70 - 3);
          *((_QWORD *)v71 - 2) = *((_QWORD *)v70 - 2);
          *((_QWORD *)v71 - 1) = *((_QWORD *)v70 - 1);
          --v72;
        }
        while ( v72 );
        v74 = 8 * v67;
        if ( v67 <= 0 )
          v74 = 8;
        v22 = (char *)&v16[v74];
      }
      goto LABEL_9;
    }
    while ( 1 )
    {
LABEL_15:
      if ( a7 < v12 )
      {
        v78 = v18;
        v82 = v20;
        v22 = sub_27A0CC0((char *)v16, (char *)v19, v84);
        LODWORD(v20) = v82;
        LODWORD(v18) = v78;
        goto LABEL_9;
      }
      v22 = v84;
      if ( !v12 )
        goto LABEL_9;
      v24 = (__int64)&v84[-v19];
      v25 = v19 - (_QWORD)v16;
      v26 = (__int64)&v84[-v19] >> 5;
      v27 = (v19 - (__int64)v16) >> 5;
      if ( v19 - (__int64)v16 <= 0 )
      {
        if ( v24 <= 0 )
          goto LABEL_9;
        v31 = a6;
        v32 = 0;
        v25 = 0;
      }
      else
      {
        v81 = v24 >> 5;
        v28 = a6;
        v29 = v16;
        do
        {
          v30 = *v29;
          v28 += 32;
          v29 += 8;
          *((_DWORD *)v28 - 8) = v30;
          *((_QWORD *)v28 - 3) = *((_QWORD *)v29 - 3);
          *((_QWORD *)v28 - 2) = *((_QWORD *)v29 - 2);
          *((_QWORD *)v28 - 1) = *((_QWORD *)v29 - 1);
          --v27;
        }
        while ( v27 );
        v26 = v81;
        if ( v25 <= 0 )
          v25 = 32;
        v31 = &a6[v25];
        v32 = v25 >> 5;
        if ( (__int64)&v84[-v19] <= 0 )
          goto LABEL_25;
      }
      v33 = v16;
      do
      {
        v34 = *(_DWORD *)v19;
        v33 += 8;
        v19 += 32;
        *(v33 - 8) = v34;
        *((_QWORD *)v33 - 3) = *(_QWORD *)(v19 - 24);
        *((_QWORD *)v33 - 2) = *(_QWORD *)(v19 - 16);
        *((_QWORD *)v33 - 1) = *(_QWORD *)(v19 - 8);
        --v26;
      }
      while ( v26 );
LABEL_25:
      if ( v25 <= 0 )
      {
        v22 = v84;
      }
      else
      {
        v35 = v84;
        v36 = v32;
        do
        {
          v37 = *((_DWORD *)v31 - 8);
          v31 -= 32;
          v35 -= 32;
          *(_DWORD *)v35 = v37;
          *((_QWORD *)v35 + 1) = *((_QWORD *)v31 + 1);
          *((_QWORD *)v35 + 2) = *((_QWORD *)v31 + 2);
          *((_QWORD *)v35 + 3) = *((_QWORD *)v31 + 3);
          --v36;
        }
        while ( v36 );
        v38 = -32 * v32;
        if ( v32 <= 0 )
          v38 = -32;
        v22 = &v84[v38];
      }
LABEL_9:
      v13 -= v21;
      v80 = v22;
      sub_27A1980(v18, (_DWORD)v16, (_DWORD)v22, v20, v21, (_DWORD)a6, a7);
      v23 = v13;
      result = (__int64)v80;
      if ( a7 <= v13 )
        v23 = a7;
      if ( v23 >= v12 )
      {
        v11 = (char *)a3;
        v9 = a6;
        v10 = v80;
        v8 = v84;
        break;
      }
      if ( a7 >= v13 )
      {
        v11 = (char *)a3;
        v9 = a6;
        v10 = v80;
        v8 = v84;
        goto LABEL_50;
      }
      v14 = (__int64)v84;
      v15 = (__int64)v80;
      if ( v13 < v12 )
        goto LABEL_6;
LABEL_14:
      v21 = v13 / 2;
      v84 = (char *)(v14 + 32 * (v13 / 2));
      v16 = (unsigned int *)sub_27A1160(v15, v14, (unsigned int *)v84);
      v20 = ((__int64)v16 - v18) >> 5;
      v12 -= v20;
      if ( v12 > v13 / 2 )
        goto LABEL_7;
    }
  }
  v39 = (v8 - v10) >> 5;
  if ( v8 - v10 <= 0 )
    return result;
  v40 = v9;
  result = (__int64)v10;
  do
  {
    v41 = *(_DWORD *)result;
    v40 += 32;
    result += 32;
    *((_DWORD *)v40 - 8) = v41;
    *((_QWORD *)v40 - 3) = *(_QWORD *)(result - 24);
    *((_QWORD *)v40 - 2) = *(_QWORD *)(result - 16);
    *((_QWORD *)v40 - 1) = *(_QWORD *)(result - 8);
    --v39;
  }
  while ( v39 );
  v42 = 32;
  if ( v8 - v10 > 0 )
    v42 = v8 - v10;
  v43 = &v9[v42];
  if ( v9 != v43 )
  {
    while ( v11 != v8 )
    {
      v45 = *(_DWORD *)v8;
      v46 = *(_DWORD *)v9;
      if ( *(_DWORD *)v8 < *(_DWORD *)v9 || v45 == v46 && *((_QWORD *)v8 + 1) < *((_QWORD *)v9 + 1) )
      {
        *(_DWORD *)v10 = v45;
        v47 = *((_QWORD *)v8 + 1);
        v10 += 32;
        v8 += 32;
        *((_QWORD *)v10 - 3) = v47;
        *((_QWORD *)v10 - 2) = *((_QWORD *)v8 - 2);
        result = *((_QWORD *)v8 - 1);
        *((_QWORD *)v10 - 1) = result;
        if ( v9 == v43 )
          break;
      }
      else
      {
        *(_DWORD *)v10 = v46;
        v44 = *((_QWORD *)v9 + 1);
        v9 += 32;
        v10 += 32;
        *((_QWORD *)v10 - 3) = v44;
        *((_QWORD *)v10 - 2) = *((_QWORD *)v9 - 2);
        result = *((_QWORD *)v9 - 1);
        *((_QWORD *)v10 - 1) = result;
        if ( v9 == v43 )
          break;
      }
    }
  }
  if ( v43 != v9 )
  {
    v48 = v43 - v9;
    result = v48 >> 5;
    if ( v48 > 0 )
    {
      do
      {
        v49 = *(_DWORD *)v9;
        v10 += 32;
        v9 += 32;
        *((_DWORD *)v10 - 8) = v49;
        *((_QWORD *)v10 - 3) = *((_QWORD *)v9 - 3);
        *((_QWORD *)v10 - 2) = *((_QWORD *)v9 - 2);
        *((_QWORD *)v10 - 1) = *((_QWORD *)v9 - 1);
        --result;
      }
      while ( result );
    }
  }
  return result;
}
