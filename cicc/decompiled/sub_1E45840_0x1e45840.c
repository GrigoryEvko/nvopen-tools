// Function: sub_1E45840
// Address: 0x1e45840
//
__int64 __fastcall sub_1E45840(char *a1, char *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 v7; // r15
  _DWORD *v8; // r14
  char *v9; // r13
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // r12
  char *v13; // r14
  char *v14; // r15
  __int64 v15; // rbx
  __int64 v16; // rax
  int v17; // ecx
  __int64 v18; // rdx
  char *v19; // rbx
  __int64 result; // rax
  __int64 v21; // r12
  __int64 v22; // rsi
  __int64 v23; // rdi
  unsigned int v24; // edx
  unsigned int v25; // eax
  bool v26; // cl
  unsigned int v27; // eax
  unsigned int v28; // edx
  int v29; // ecx
  int v30; // edx
  bool v31; // al
  __int64 v32; // rsi
  __int64 v33; // rdi
  __int64 v34; // rdx
  __int64 v35; // rsi
  char *v36; // r12
  _DWORD *i; // rbx
  unsigned int v38; // edx
  unsigned int v39; // eax
  bool v40; // cl
  unsigned int v41; // edx
  unsigned int v42; // eax
  unsigned int v43; // eax
  unsigned int v44; // edx
  int v45; // edx
  int v46; // eax
  char *v47; // r13
  char *v48; // rbx
  unsigned __int64 v49; // r14
  __int64 v50; // rdi
  __int64 v51; // rcx
  __int64 v52; // rdx
  __int64 v53; // rdi
  __int64 v54; // rsi
  __int64 v55; // [rsp+8h] [rbp-78h]
  __int64 v56; // [rsp+10h] [rbp-70h]
  __int64 v57; // [rsp+10h] [rbp-70h]
  char *v59; // [rsp+20h] [rbp-60h]
  __int64 v61; // [rsp+30h] [rbp-50h]
  char *v62; // [rsp+30h] [rbp-50h]
  __int64 v63; // [rsp+40h] [rbp-40h]

  v7 = a3;
  v8 = (_DWORD *)a6;
  v9 = a2;
  v63 = a4;
  v10 = a7;
  if ( a7 > a5 )
    v10 = a5;
  if ( v10 >= a4 )
  {
LABEL_17:
    v19 = a1;
    result = sub_1E42C80((__int64)a1, (__int64)v9, (__int64)v8);
    v21 = result;
    if ( v8 == (_DWORD *)result || (char *)v7 == v9 )
    {
LABEL_51:
      if ( (_DWORD *)v21 != v8 )
        return sub_1E42C80((__int64)v8, v21, (__int64)a1);
      return result;
    }
    while ( 1 )
    {
      v24 = *((_DWORD *)v9 + 15);
      v25 = v8[15];
      v26 = v24 > v25;
      if ( v24 != v25 || (v27 = *((_DWORD *)v9 + 18)) != 0 && (v28 = v8[18], v27 != v28) && (v26 = v27 < v28, v28) )
      {
        if ( !v26 )
          goto LABEL_30;
      }
      else
      {
        v29 = *((_DWORD *)v9 + 16);
        v30 = v8[16];
        v31 = v29 < v30;
        if ( v29 == v30 )
          v31 = *((_DWORD *)v9 + 17) > v8[17];
        if ( !v31 )
        {
LABEL_30:
          v32 = (__int64)v8;
          v33 = (__int64)v19;
          v8 += 24;
          v19 += 96;
          result = sub_1E41D50(v33, v32);
          if ( (_DWORD *)v21 == v8 )
            return result;
          goto LABEL_22;
        }
      }
      v22 = (__int64)v9;
      v23 = (__int64)v19;
      v9 += 96;
      v19 += 96;
      result = sub_1E41D50(v23, v22);
      if ( (_DWORD *)v21 == v8 )
        return result;
LABEL_22:
      if ( (char *)v7 == v9 )
      {
        a1 = v19;
        goto LABEL_51;
      }
    }
  }
  if ( a7 >= a5 )
    goto LABEL_33;
  v11 = a4;
  v12 = a5;
  v13 = a2;
  if ( a5 >= a4 )
    goto LABEL_15;
LABEL_6:
  v61 = v11 / 2;
  v14 = &a1[32 * (v11 / 2) + 32 * ((v11 + ((unsigned __int64)v11 >> 63)) & 0xFFFFFFFFFFFFFFFELL)];
  v59 = (char *)sub_1E426C0(v13, a3, v14);
  v15 = 0xAAAAAAAAAAAAAAABLL * ((v59 - v13) >> 5);
  while ( 1 )
  {
    v63 -= v61;
    if ( v63 <= v15 || a7 < v15 )
    {
      if ( a7 < v63 )
      {
        v16 = sub_1E435D0((__int64)v14, (__int64)v13, (__int64)v59);
      }
      else
      {
        v16 = (__int64)v59;
        if ( v63 )
        {
          v56 = sub_1E42C80((__int64)v14, (__int64)v13, a6);
          sub_1E42C80((__int64)v13, (__int64)v59, (__int64)v14);
          v16 = sub_1E42AF0(a6, v56, (__int64)v59);
        }
      }
    }
    else
    {
      v16 = (__int64)v14;
      if ( v15 )
      {
        v47 = v59;
        v57 = sub_1E42C80((__int64)v13, (__int64)v59, a6);
        if ( v13 - v14 > 0 )
        {
          v55 = v15;
          v48 = v13;
          v49 = 0xAAAAAAAAAAAAAAABLL * ((v13 - v14) >> 5);
          do
          {
            v50 = *((_QWORD *)v47 - 11);
            v47 -= 96;
            v48 -= 96;
            j___libc_free_0(v50);
            *((_DWORD *)v47 + 6) = 0;
            *((_QWORD *)v47 + 1) = 0;
            *((_DWORD *)v47 + 4) = 0;
            *((_DWORD *)v47 + 5) = 0;
            ++*(_QWORD *)v47;
            v51 = *((_QWORD *)v48 + 1);
            ++*(_QWORD *)v48;
            v52 = *((_QWORD *)v47 + 1);
            *((_QWORD *)v47 + 1) = v51;
            LODWORD(v51) = *((_DWORD *)v48 + 4);
            *((_QWORD *)v48 + 1) = v52;
            LODWORD(v52) = *((_DWORD *)v47 + 4);
            *((_DWORD *)v47 + 4) = v51;
            LODWORD(v51) = *((_DWORD *)v48 + 5);
            *((_DWORD *)v48 + 4) = v52;
            LODWORD(v52) = *((_DWORD *)v47 + 5);
            *((_DWORD *)v47 + 5) = v51;
            LODWORD(v51) = *((_DWORD *)v48 + 6);
            *((_DWORD *)v48 + 5) = v52;
            LODWORD(v52) = *((_DWORD *)v47 + 6);
            *((_DWORD *)v47 + 6) = v51;
            *((_DWORD *)v48 + 6) = v52;
            v53 = *((_QWORD *)v47 + 4);
            v54 = *((_QWORD *)v47 + 6);
            *((_QWORD *)v47 + 4) = *((_QWORD *)v48 + 4);
            *((_QWORD *)v47 + 5) = *((_QWORD *)v48 + 5);
            *((_QWORD *)v47 + 6) = *((_QWORD *)v48 + 6);
            *((_QWORD *)v48 + 4) = 0;
            *((_QWORD *)v48 + 5) = 0;
            *((_QWORD *)v48 + 6) = 0;
            if ( v53 )
              j_j___libc_free_0(v53, v54 - v53);
            v47[56] = v48[56];
            *((_DWORD *)v47 + 15) = *((_DWORD *)v48 + 15);
            *((_DWORD *)v47 + 16) = *((_DWORD *)v48 + 16);
            *((_DWORD *)v47 + 17) = *((_DWORD *)v48 + 17);
            *((_DWORD *)v47 + 18) = *((_DWORD *)v48 + 18);
            *((_QWORD *)v47 + 10) = *((_QWORD *)v48 + 10);
            *((_DWORD *)v47 + 22) = *((_DWORD *)v48 + 22);
            --v49;
          }
          while ( v49 );
          v15 = v55;
        }
        v16 = sub_1E42C80(a6, v57, (__int64)v14);
      }
    }
    v17 = v61;
    v12 -= v15;
    v62 = (char *)v16;
    sub_1E45840((_DWORD)a1, (_DWORD)v14, v16, v17, v15, a6, a7);
    v18 = a7;
    if ( a7 > v12 )
      v18 = v12;
    if ( v18 >= v63 )
    {
      v7 = a3;
      v8 = (_DWORD *)a6;
      a1 = v62;
      v9 = v59;
      goto LABEL_17;
    }
    if ( a7 >= v12 )
      break;
    a1 = v62;
    v11 = v63;
    v13 = v59;
    if ( v12 < v63 )
      goto LABEL_6;
LABEL_15:
    v15 = v12 / 2;
    v59 = &v13[32 * (v12 / 2) + 32 * ((v12 + ((unsigned __int64)v12 >> 63)) & 0xFFFFFFFFFFFFFFFELL)];
    v14 = (char *)sub_1E42610(a1, (__int64)v13, v59);
    v61 = 0xAAAAAAAAAAAAAAABLL * ((v14 - a1) >> 5);
  }
  v7 = a3;
  v8 = (_DWORD *)a6;
  a1 = v62;
  v9 = v59;
LABEL_33:
  result = sub_1E42C80((__int64)v9, v7, (__int64)v8);
  v34 = v7;
  v35 = result;
  if ( v9 == a1 )
    return sub_1E42AF0((__int64)v8, v35, v34);
  if ( v8 != (_DWORD *)result )
  {
    v36 = v9 - 96;
    for ( i = (_DWORD *)(result - 96); ; i -= 24 )
    {
      v38 = i[15];
      v39 = *((_DWORD *)v36 + 15);
      v40 = v38 > v39;
      if ( v38 == v39 )
        goto LABEL_40;
      while ( 1 )
      {
        v7 -= 96;
        if ( !v40 )
          break;
        sub_1E41D50(v7, (__int64)v36);
        if ( v36 == a1 )
        {
          v35 = (__int64)(i + 24);
          v34 = v7;
          return sub_1E42AF0((__int64)v8, v35, v34);
        }
        v41 = i[15];
        v42 = *((_DWORD *)v36 - 9);
        v36 -= 96;
        v40 = v41 > v42;
        if ( v41 == v42 )
        {
LABEL_40:
          v43 = i[18];
          if ( !v43 || (v44 = *((_DWORD *)v36 + 18)) == 0 || (v40 = v43 < v44, v43 == v44) )
          {
            v45 = i[16];
            v46 = *((_DWORD *)v36 + 16);
            v40 = v45 < v46;
            if ( v45 == v46 )
              v40 = i[17] > *((_DWORD *)v36 + 17);
          }
        }
      }
      result = sub_1E41D50(v7, (__int64)i);
      if ( v8 == i )
        return result;
    }
  }
  return result;
}
