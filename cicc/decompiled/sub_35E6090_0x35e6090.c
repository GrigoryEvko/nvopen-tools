// Function: sub_35E6090
// Address: 0x35e6090
//
__int64 __fastcall sub_35E6090(char *a1, char *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 v7; // rax
  char *v8; // r10
  __int64 v9; // r13
  char *v10; // r12
  __int64 v12; // r14
  __int64 v13; // r15
  char *v14; // r9
  char *v15; // r10
  char *v16; // rax
  __int64 v17; // r9
  __int64 v18; // r10
  _DWORD *v19; // r11
  __int64 v20; // rcx
  __int64 v21; // r13
  unsigned __int64 v22; // rax
  __int64 v23; // rdx
  unsigned __int64 v24; // rdx
  unsigned __int64 v25; // rsi
  __int64 v26; // r8
  _DWORD *v27; // rax
  __int64 v28; // rdx
  __int64 v29; // r8
  _DWORD *v30; // rax
  signed __int64 v31; // rdi
  _DWORD *v32; // rsi
  __int64 v33; // rax
  char *v34; // rdx
  signed __int64 v35; // rsi
  __int64 v36; // r8
  __int64 v37; // rax
  __int64 result; // rax
  unsigned __int64 v39; // rcx
  __int64 v40; // rdx
  char *v41; // rax
  __int64 v42; // rsi
  __int64 v43; // rax
  __int64 v44; // rdx
  __int64 v45; // rdx
  unsigned __int64 v46; // rdx
  __int64 v47; // rax
  __int64 v48; // r8
  unsigned __int64 v49; // rcx
  __int64 v50; // rdx
  char *v51; // rax
  __int64 v52; // rsi
  char *v53; // r10
  __int64 v54; // rdi
  unsigned __int64 v55; // rdx
  unsigned __int64 v56; // rsi
  __int64 v57; // r8
  __int64 v58; // rax
  __int64 v59; // rdx
  signed __int64 v60; // rsi
  char *v61; // rax
  __int64 v62; // r8
  _DWORD *v63; // rdx
  __int64 v64; // rax
  signed __int64 v65; // rdi
  __int64 v66; // r8
  __int64 v67; // rax
  unsigned __int64 v68; // rdx
  __int64 v69; // rcx
  unsigned __int64 v70; // rdx
  int v71; // [rsp+8h] [rbp-58h]
  int v72; // [rsp+10h] [rbp-50h]
  char *v74; // [rsp+20h] [rbp-40h]
  __int64 v75; // [rsp+20h] [rbp-40h]
  _DWORD *v76; // [rsp+20h] [rbp-40h]
  __int64 v77; // [rsp+20h] [rbp-40h]
  int v78; // [rsp+20h] [rbp-40h]
  char *v79; // [rsp+28h] [rbp-38h]

  v7 = a5;
  v8 = a2;
  v9 = a3;
  v10 = a1;
  if ( a7 <= a5 )
    v7 = a7;
  if ( v7 >= a4 )
  {
LABEL_33:
    result = 0xAAAAAAAAAAAAAAABLL;
    v39 = 0xAAAAAAAAAAAAAAABLL * ((v8 - v10) >> 3);
    if ( v8 - v10 > 0 )
    {
      v40 = a6;
      v41 = v10;
      do
      {
        v42 = *((_QWORD *)v41 + 2);
        v40 += 24;
        v41 += 24;
        *(_QWORD *)(v40 - 8) = v42;
        *(_DWORD *)(v40 - 16) = *((_DWORD *)v41 - 4);
        *(_DWORD *)(v40 - 20) = *((_DWORD *)v41 - 5);
        *(_DWORD *)(v40 - 24) = *((_DWORD *)v41 - 6);
        --v39;
      }
      while ( v39 );
      v43 = 24;
      if ( v8 - v10 > 0 )
        v43 = v8 - v10;
      result = a6 + v43;
      if ( a6 != result )
      {
        while ( (char *)v9 != v8 )
        {
          if ( *(_DWORD *)v8 < *(_DWORD *)a6 )
          {
            v44 = *((_QWORD *)v8 + 2);
            v10 += 24;
            v8 += 24;
            *((_QWORD *)v10 - 1) = v44;
            *((_DWORD *)v10 - 4) = *((_DWORD *)v8 - 4);
            *((_DWORD *)v10 - 5) = *((_DWORD *)v8 - 5);
            *((_DWORD *)v10 - 6) = *((_DWORD *)v8 - 6);
            if ( result == a6 )
              break;
          }
          else
          {
            v45 = *(_QWORD *)(a6 + 16);
            a6 += 24;
            v10 += 24;
            *((_QWORD *)v10 - 1) = v45;
            *((_DWORD *)v10 - 4) = *(_DWORD *)(a6 - 16);
            *((_DWORD *)v10 - 5) = *(_DWORD *)(a6 - 20);
            *((_DWORD *)v10 - 6) = *(_DWORD *)(a6 - 24);
            if ( result == a6 )
              break;
          }
        }
        if ( a6 != result )
        {
          result -= a6;
          v46 = 0xAAAAAAAAAAAAAAABLL * (result >> 3);
          if ( result > 0 )
          {
            do
            {
              v47 = *(_QWORD *)(a6 + 16);
              v10 += 24;
              a6 += 24;
              *((_QWORD *)v10 - 1) = v47;
              *((_DWORD *)v10 - 4) = *(_DWORD *)(a6 - 16);
              *((_DWORD *)v10 - 5) = *(_DWORD *)(a6 - 20);
              result = *(unsigned int *)(a6 - 24);
              *((_DWORD *)v10 - 6) = result;
              --v46;
            }
            while ( v46 );
          }
        }
      }
    }
  }
  else
  {
    v12 = a5;
    if ( a7 < a5 )
    {
      v13 = a4;
      v14 = a2;
      v15 = a1;
      if ( a5 >= a4 )
        goto LABEL_14;
LABEL_6:
      v16 = (char *)sub_35E5080(
                      v14,
                      a3,
                      &v15[8 * (v13 / 2) + 8 * ((v13 + ((unsigned __int64)v13 >> 63)) & 0xFFFFFFFFFFFFFFFELL)]);
      v20 = v13 / 2;
      v79 = v16;
      v13 -= v13 / 2;
      v21 = 0xAAAAAAAAAAAAAAABLL * ((__int64)&v16[-v17] >> 3);
      if ( v13 <= v21 )
        goto LABEL_15;
LABEL_7:
      if ( a7 >= v21 )
      {
        v22 = (unsigned __int64)v19;
        if ( !v21 )
          goto LABEL_9;
        v54 = (__int64)&v79[-v17];
        v77 = v17 - (_QWORD)v19;
        v55 = 0xAAAAAAAAAAAAAAABLL * ((v17 - (__int64)v19) >> 3);
        v56 = 0xAAAAAAAAAAAAAAABLL * ((__int64)&v79[-v17] >> 3);
        if ( (__int64)&v79[-v17] <= 0 )
        {
          if ( v77 <= 0 )
            goto LABEL_9;
          v60 = 0;
          v54 = 0;
        }
        else
        {
          v57 = a6;
          v58 = v17;
          do
          {
            v59 = *(_QWORD *)(v58 + 16);
            v57 += 24;
            v58 += 24;
            *(_QWORD *)(v57 - 8) = v59;
            *(_DWORD *)(v57 - 16) = *(_DWORD *)(v58 - 16);
            *(_DWORD *)(v57 - 20) = *(_DWORD *)(v58 - 20);
            *(_DWORD *)(v57 - 24) = *(_DWORD *)(v58 - 24);
            --v56;
          }
          while ( v56 );
          v55 = 0xAAAAAAAAAAAAAAABLL * ((v17 - (__int64)v19) >> 3);
          if ( v54 <= 0 )
            v54 = 24;
          v60 = 0xAAAAAAAAAAAAAAABLL * (v54 >> 3);
          if ( v77 <= 0 )
            goto LABEL_70;
        }
        v61 = v79;
        do
        {
          v62 = *(_QWORD *)(v17 - 8);
          v17 -= 24;
          v61 -= 24;
          *((_QWORD *)v61 + 2) = v62;
          *((_DWORD *)v61 + 2) = *(_DWORD *)(v17 + 8);
          *((_DWORD *)v61 + 1) = *(_DWORD *)(v17 + 4);
          *(_DWORD *)v61 = *(_DWORD *)v17;
          --v55;
        }
        while ( v55 );
LABEL_70:
        if ( v54 <= 0 )
        {
          v22 = (unsigned __int64)v19;
        }
        else
        {
          v63 = v19;
          v64 = a6;
          v65 = v60;
          do
          {
            v66 = *(_QWORD *)(v64 + 16);
            v63 += 6;
            v64 += 24;
            *((_QWORD *)v63 - 1) = v66;
            *(v63 - 4) = *(_DWORD *)(v64 - 16);
            *(v63 - 5) = *(_DWORD *)(v64 - 20);
            *(v63 - 6) = *(_DWORD *)(v64 - 24);
            --v65;
          }
          while ( v65 );
          v67 = 6 * v60;
          if ( v60 <= 0 )
            v67 = 6;
          v22 = (unsigned __int64)&v19[v67];
        }
        goto LABEL_9;
      }
      while ( 1 )
      {
LABEL_15:
        if ( a7 < v13 )
        {
          v71 = v18;
          v72 = v20;
          v78 = (int)v19;
          v22 = sub_35E4E60((__int64)v19, v17, (__int64)v79);
          LODWORD(v19) = v78;
          LODWORD(v20) = v72;
          LODWORD(v18) = v71;
          goto LABEL_9;
        }
        v22 = (unsigned __int64)v79;
        if ( !v13 )
          goto LABEL_9;
        v75 = (__int64)&v79[-v17];
        v24 = 0xAAAAAAAAAAAAAAABLL * ((__int64)&v79[-v17] >> 3);
        v25 = 0xAAAAAAAAAAAAAAABLL * ((v17 - (__int64)v19) >> 3);
        if ( v17 - (__int64)v19 <= 0 )
        {
          if ( v75 <= 0 )
            goto LABEL_9;
          v30 = (_DWORD *)a6;
          v31 = 0;
          v29 = 0;
        }
        else
        {
          v26 = a6;
          v27 = v19;
          do
          {
            v28 = *((_QWORD *)v27 + 2);
            v26 += 24;
            v27 += 6;
            *(_QWORD *)(v26 - 8) = v28;
            *(_DWORD *)(v26 - 16) = *(v27 - 4);
            *(_DWORD *)(v26 - 20) = *(v27 - 5);
            *(_DWORD *)(v26 - 24) = *(v27 - 6);
            --v25;
          }
          while ( v25 );
          v29 = 24;
          v24 = 0xAAAAAAAAAAAAAAABLL * ((__int64)&v79[-v17] >> 3);
          if ( v17 - (__int64)v19 > 0 )
            v29 = v17 - (_QWORD)v19;
          v30 = (_DWORD *)(a6 + v29);
          v31 = 0xAAAAAAAAAAAAAAABLL * (v29 >> 3);
          if ( v75 <= 0 )
            goto LABEL_26;
        }
        v76 = v30;
        v32 = v19;
        do
        {
          v33 = *(_QWORD *)(v17 + 16);
          v32 += 6;
          v17 += 24;
          *((_QWORD *)v32 - 1) = v33;
          *(v32 - 4) = *(_DWORD *)(v17 - 16);
          *(v32 - 5) = *(_DWORD *)(v17 - 20);
          *(v32 - 6) = *(_DWORD *)(v17 - 24);
          --v24;
        }
        while ( v24 );
        v30 = v76;
LABEL_26:
        if ( v29 <= 0 )
        {
          v22 = (unsigned __int64)v79;
        }
        else
        {
          v34 = v79;
          v35 = v31;
          do
          {
            v36 = *((_QWORD *)v30 - 1);
            v30 -= 6;
            v34 -= 24;
            *((_QWORD *)v34 + 2) = v36;
            *((_DWORD *)v34 + 2) = v30[2];
            *((_DWORD *)v34 + 1) = v30[1];
            *(_DWORD *)v34 = *v30;
            --v35;
          }
          while ( v35 );
          v37 = -24 * v31;
          if ( v31 <= 0 )
            v37 = -24;
          v22 = (unsigned __int64)&v79[v37];
        }
LABEL_9:
        v12 -= v21;
        v74 = (char *)v22;
        sub_35E6090(v18, (_DWORD)v19, v22, v20, v21, a6, a7);
        v23 = v12;
        if ( a7 <= v12 )
          v23 = a7;
        if ( v23 >= v13 )
        {
          v9 = a3;
          v8 = v79;
          v10 = v74;
          goto LABEL_33;
        }
        if ( a7 >= v12 )
        {
          v9 = a3;
          v8 = v79;
          v10 = v74;
          break;
        }
        v14 = v79;
        v15 = v74;
        if ( v12 < v13 )
          goto LABEL_6;
LABEL_14:
        v21 = v12 / 2;
        v79 = &v14[8 * (v12 / 2) + 8 * ((v12 + ((unsigned __int64)v12 >> 63)) & 0xFFFFFFFFFFFFFFFELL)];
        v19 = sub_35E50E0(v15, (__int64)v14, v79);
        v20 = 0xAAAAAAAAAAAAAAABLL * (((__int64)v19 - v18) >> 3);
        v13 -= v20;
        if ( v13 > v12 / 2 )
          goto LABEL_7;
      }
    }
    result = 0xAAAAAAAAAAAAAAABLL;
    v48 = v9 - (_QWORD)v8;
    v49 = 0xAAAAAAAAAAAAAAABLL * ((v9 - (__int64)v8) >> 3);
    if ( v9 - (__int64)v8 > 0 )
    {
      v50 = a6;
      v51 = v8;
      do
      {
        v52 = *((_QWORD *)v51 + 2);
        v50 += 24;
        v51 += 24;
        *(_QWORD *)(v50 - 8) = v52;
        *(_DWORD *)(v50 - 16) = *((_DWORD *)v51 - 4);
        *(_DWORD *)(v50 - 20) = *((_DWORD *)v51 - 5);
        *(_DWORD *)(v50 - 24) = *((_DWORD *)v51 - 6);
        --v49;
      }
      while ( v49 );
      if ( v48 <= 0 )
        v48 = 24;
      result = a6 + v48;
      if ( v8 == v10 )
      {
        v70 = 0xAAAAAAAAAAAAAAABLL * (v48 >> 3);
        while ( 1 )
        {
          result -= 24;
          *(_QWORD *)(v9 - 8) = v52;
          v9 -= 24;
          *(_DWORD *)(v9 + 8) = *(_DWORD *)(result + 8);
          *(_DWORD *)(v9 + 4) = *(_DWORD *)(result + 4);
          *(_DWORD *)v9 = *(_DWORD *)result;
          if ( !--v70 )
            break;
          v52 = *(_QWORD *)(result - 8);
        }
      }
      else if ( a6 != result )
      {
        v53 = v8 - 24;
        while ( 1 )
        {
          result -= 24;
          v9 -= 24;
          if ( *(_DWORD *)result < *(_DWORD *)v53 )
            break;
LABEL_60:
          *(_QWORD *)(v9 + 16) = *(_QWORD *)(result + 16);
          *(_DWORD *)(v9 + 8) = *(_DWORD *)(result + 8);
          *(_DWORD *)(v9 + 4) = *(_DWORD *)(result + 4);
          *(_DWORD *)v9 = *(_DWORD *)result;
          if ( a6 == result )
            return result;
        }
        while ( 1 )
        {
          *(_QWORD *)(v9 + 16) = *((_QWORD *)v53 + 2);
          *(_DWORD *)(v9 + 8) = *((_DWORD *)v53 + 2);
          *(_DWORD *)(v9 + 4) = *((_DWORD *)v53 + 1);
          *(_DWORD *)v9 = *(_DWORD *)v53;
          if ( v53 == v10 )
            break;
          v53 -= 24;
          v9 -= 24;
          if ( *(_DWORD *)result >= *(_DWORD *)v53 )
            goto LABEL_60;
        }
        result += 24;
        v68 = 0xAAAAAAAAAAAAAAABLL * ((result - a6) >> 3);
        if ( result - a6 > 0 )
        {
          do
          {
            v69 = *(_QWORD *)(result - 8);
            result -= 24;
            v9 -= 24;
            *(_QWORD *)(v9 + 16) = v69;
            *(_DWORD *)(v9 + 8) = *(_DWORD *)(result + 8);
            *(_DWORD *)(v9 + 4) = *(_DWORD *)(result + 4);
            *(_DWORD *)v9 = *(_DWORD *)result;
            --v68;
          }
          while ( v68 );
        }
      }
    }
  }
  return result;
}
