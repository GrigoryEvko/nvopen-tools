// Function: sub_BA9C10
// Address: 0xba9c10
//
unsigned __int64 *__fastcall sub_BA9C10(_QWORD **a1, __int64 a2, __int64 a3, __int64 a4)
{
  _QWORD *v5; // r15
  _QWORD *v6; // r13
  unsigned __int64 *v7; // rcx
  unsigned __int64 v8; // rdx
  __int64 v9; // rdx
  __int64 v10; // rcx
  _QWORD *v11; // r15
  _QWORD **v12; // r13
  _QWORD *v13; // r14
  unsigned __int64 *v14; // rcx
  unsigned __int64 v15; // rdx
  _QWORD *v16; // rcx
  _QWORD **v17; // r14
  _QWORD *v18; // r15
  unsigned __int64 *v19; // rsi
  unsigned __int64 v20; // rdx
  _QWORD *v21; // r8
  _QWORD **v22; // r15
  __int64 v23; // r12
  __int64 *v24; // rsi
  unsigned __int64 v25; // rcx
  __int64 v26; // rdx
  __int64 v27; // rsi
  __int64 v28; // rcx
  _QWORD *v29; // r8
  __int64 v30; // rcx
  __int64 v31; // rax
  _QWORD *v32; // rdi
  __int64 v33; // rcx
  _QWORD *v34; // r8
  __int64 v35; // rcx
  __int64 v36; // rax
  _QWORD *v37; // rdi
  _QWORD **v38; // rdi
  _QWORD **v39; // rdi
  _QWORD **v40; // rdi
  _QWORD *v41; // rdi
  _QWORD *v42; // rdi
  _QWORD *v43; // rdi
  __int64 v44; // rax
  __int64 v45; // rax
  __int64 v46; // r12
  _QWORD *v47; // rdi
  _QWORD **v48; // rdi
  __int64 v49; // rsi
  unsigned __int64 *result; // rax
  unsigned __int64 *v51; // r12
  unsigned __int64 v52; // rcx
  _QWORD *v53; // r8
  __int64 v54; // r12
  __int64 *v55; // rsi
  unsigned __int64 v56; // rcx
  __int64 v57; // rdx
  _QWORD *v58; // rcx
  _QWORD *v59; // r15
  unsigned __int64 *v60; // rsi
  unsigned __int64 v61; // rdx
  _QWORD *i; // r12
  _QWORD *v63; // r14
  unsigned __int64 *v64; // rcx
  unsigned __int64 v65; // rdx
  _QWORD *j; // r14
  _QWORD *v67; // rbx
  unsigned __int64 *v68; // rcx
  unsigned __int64 v69; // rdx
  __int64 v70; // rdx
  __int64 v71; // rcx
  __int64 v72; // [rsp+0h] [rbp-50h]
  _QWORD *v73; // [rsp+8h] [rbp-48h]
  __int64 v74; // [rsp+8h] [rbp-48h]
  __int64 v75; // [rsp+8h] [rbp-48h]
  __int64 v76; // [rsp+8h] [rbp-48h]
  _QWORD *v77; // [rsp+8h] [rbp-48h]
  _QWORD *v78; // [rsp+10h] [rbp-40h]
  _QWORD *v79; // [rsp+10h] [rbp-40h]
  __int64 v80; // [rsp+10h] [rbp-40h]
  __int64 v81; // [rsp+10h] [rbp-40h]
  __int64 v82; // [rsp+10h] [rbp-40h]
  _QWORD *v83; // [rsp+10h] [rbp-40h]
  unsigned __int64 *v84; // [rsp+10h] [rbp-40h]
  _QWORD *v85; // [rsp+10h] [rbp-40h]
  _QWORD *v86; // [rsp+10h] [rbp-40h]
  __int64 v87; // [rsp+18h] [rbp-38h]

  sub_B6E7A0(*a1, (__int64)a1, a3, a4);
  sub_BA9A80(a1);
  v5 = a1[2];
  v87 = (__int64)(a1 + 1);
  while ( (_QWORD *)v87 != v5 )
  {
    v6 = v5;
    v5 = (_QWORD *)v5[1];
    sub_BA85F0(v87, (__int64)(v6 - 7));
    v7 = (unsigned __int64 *)v6[1];
    v8 = *v6 & 0xFFFFFFFFFFFFFFF8LL;
    *v7 = v8 | *v7 & 7;
    *(_QWORD *)(v8 + 8) = v7;
    *v6 &= 7uLL;
    v6[1] = 0;
    sub_B30220((__int64)(v6 - 7));
    *((_DWORD *)v6 - 13) = *((_DWORD *)v6 - 13) & 0xF8000000 | 1;
    sub_B2F9E0((__int64)(v6 - 7), (__int64)(v6 - 7), v9, v10);
    sub_BD2DD0(v6 - 7);
  }
  v11 = a1[4];
  v12 = a1 + 3;
  while ( v12 != v11 )
  {
    v13 = v11;
    v11 = (_QWORD *)v11[1];
    sub_BA8570((__int64)(a1 + 3), (__int64)(v13 - 7));
    v14 = (unsigned __int64 *)v13[1];
    v15 = *v13 & 0xFFFFFFFFFFFFFFF8LL;
    *v14 = v15 | *v14 & 7;
    *(_QWORD *)(v15 + 8) = v14;
    *v13 &= 7uLL;
    v13[1] = 0;
    sub_B2E780(v13 - 7);
    sub_BD2DD0(v13 - 7);
  }
  v16 = a1[6];
  v17 = a1 + 5;
  if ( a1 + 5 != v16 )
  {
    do
    {
      v18 = v16;
      v78 = (_QWORD *)v16[1];
      sub_BA8670((__int64)(a1 + 5), (__int64)(v16 - 6));
      v19 = (unsigned __int64 *)v18[1];
      v20 = *v18 & 0xFFFFFFFFFFFFFFF8LL;
      *v19 = v20 | *v19 & 7;
      *(_QWORD *)(v20 + 8) = v19;
      *v18 &= 7uLL;
      v18[1] = 0;
      sub_AD0030((__int64)(v18 - 6));
      sub_BD7260(v18 - 6);
      sub_BD2DD0(v18 - 6);
      v16 = v78;
    }
    while ( v17 != v78 );
  }
  v21 = a1[8];
  v22 = a1 + 7;
  if ( a1 + 7 != v21 )
  {
    do
    {
      v23 = (__int64)(v21 - 7);
      v79 = v21;
      v73 = (_QWORD *)v21[1];
      sub_BA86F0((__int64)(a1 + 7), (__int64)(v21 - 7));
      v24 = (__int64 *)v79[1];
      v25 = *v79 & 0xFFFFFFFFFFFFFFF8LL;
      v26 = v25 | *v24 & 7;
      *v24 = v26;
      *(_QWORD *)(v25 + 8) = v24;
      *v79 &= 7uLL;
      v79[1] = 0;
      sub_B2F9E0(v23, (__int64)v24, v26, v25);
      sub_BD2DD0(v23);
      v21 = v73;
    }
    while ( v22 != v73 );
  }
  v27 = 24LL * *((unsigned int *)a1 + 214);
  sub_C7D6A0(a1[105], v27, 8);
  if ( *((_DWORD *)a1 + 205) )
  {
    v28 = *((unsigned int *)a1 + 204);
    v29 = a1[101];
    if ( (_DWORD)v28 )
    {
      v30 = 8 * v28;
      v31 = 0;
      do
      {
        v32 = *(_QWORD **)((char *)v29 + v31);
        if ( v32 != (_QWORD *)-8LL && v32 )
        {
          v74 = v31;
          v80 = v30;
          v27 = *v32 + 17LL;
          sub_C7D6A0(v32, v27, 8);
          v29 = a1[101];
          v31 = v74;
          v30 = v80;
        }
        v31 += 8;
      }
      while ( v30 != v31 );
    }
  }
  else
  {
    v29 = a1[101];
  }
  _libc_free(v29, v27);
  sub_AE4030(a1 + 39, v27);
  if ( *((_DWORD *)a1 + 75) )
  {
    v33 = *((unsigned int *)a1 + 74);
    v34 = a1[36];
    if ( (_DWORD)v33 )
    {
      v35 = 8 * v33;
      v36 = 0;
      do
      {
        v37 = *(_QWORD **)((char *)v34 + v36);
        if ( v37 != (_QWORD *)-8LL && v37 )
        {
          v75 = v36;
          v81 = v35;
          v27 = *v37 + 17LL;
          sub_C7D6A0(v37, v27, 8);
          v34 = a1[36];
          v36 = v75;
          v35 = v81;
        }
        v36 += 8;
      }
      while ( v35 != v36 );
    }
  }
  else
  {
    v34 = a1[36];
  }
  _libc_free(v34, v27);
  v38 = (_QWORD **)a1[29];
  if ( v38 != a1 + 31 )
  {
    v27 = (__int64)a1[31] + 1;
    j_j___libc_free_0(v38, v27);
  }
  v39 = (_QWORD **)a1[25];
  if ( v39 != a1 + 27 )
  {
    v27 = (__int64)a1[27] + 1;
    j_j___libc_free_0(v39, v27);
  }
  v40 = (_QWORD **)a1[21];
  if ( v40 != a1 + 23 )
  {
    v27 = (__int64)a1[23] + 1;
    j_j___libc_free_0(v40, v27);
  }
  v41 = a1[20];
  if ( v41 )
    (*(void (__fastcall **)(_QWORD *))(*v41 + 8LL))(v41);
  v42 = a1[19];
  if ( v42 )
    (*(void (__fastcall **)(_QWORD *))(*v42 + 8LL))(v42);
  v43 = a1[16];
  if ( *((_DWORD *)a1 + 35) )
  {
    v44 = *((unsigned int *)a1 + 34);
    if ( (_DWORD)v44 )
    {
      v82 = 8 * v44;
      v45 = 0;
      do
      {
        v46 = *(_QWORD *)((char *)v43 + v45);
        if ( v46 && v46 != -8 )
        {
          v27 = *(_QWORD *)v46 + 73LL;
          if ( !*(_BYTE *)(v46 + 52) )
          {
            v72 = v45;
            _libc_free(*(_QWORD *)(v46 + 32), v27);
            v45 = v72;
          }
          v76 = v45;
          sub_C7D6A0(v46, v27, 8);
          v43 = a1[16];
          v45 = v76;
        }
        v45 += 8;
      }
      while ( v45 != v82 );
    }
  }
  _libc_free(v43, v27);
  v47 = a1[15];
  if ( v47 )
  {
    v83 = a1[15];
    sub_BD84F0(v47);
    j_j___libc_free_0(v83, 32);
  }
  v48 = (_QWORD **)a1[11];
  if ( v48 != a1 + 13 )
    j_j___libc_free_0(v48, (char *)a1[13] + 1);
  v49 = (__int64)(a1 + 9);
  result = a1[10];
  if ( a1 + 9 != (_QWORD **)result )
  {
    do
    {
      v51 = result;
      v84 = (unsigned __int64 *)result[1];
      v52 = *result & 0xFFFFFFFFFFFFFFF8LL;
      *v84 = v52 | *v84 & 7;
      *(_QWORD *)(v52 + 8) = v84;
      *result &= 7u;
      result[1] = 0;
      sub_B91A80(result, v49);
      v49 = 64;
      j_j___libc_free_0(v51, 64);
      result = v84;
    }
    while ( a1 + 9 != (_QWORD **)v84 );
  }
  v53 = a1[8];
  if ( v22 != v53 )
  {
    do
    {
      v54 = (__int64)(v53 - 7);
      v85 = v53;
      v77 = (_QWORD *)v53[1];
      sub_BA86F0((__int64)(a1 + 7), (__int64)(v53 - 7));
      v55 = (__int64 *)v85[1];
      v56 = *v85 & 0xFFFFFFFFFFFFFFF8LL;
      v57 = v56 | *v55 & 7;
      *v55 = v57;
      *(_QWORD *)(v56 + 8) = v55;
      *v85 &= 7uLL;
      v85[1] = 0;
      sub_B2F9E0(v54, (__int64)v55, v57, v56);
      result = (unsigned __int64 *)sub_BD2DD0(v54);
      v53 = v77;
    }
    while ( v22 != v77 );
  }
  v58 = a1[6];
  if ( v17 != v58 )
  {
    do
    {
      v59 = v58;
      v86 = (_QWORD *)v58[1];
      sub_BA8670((__int64)(a1 + 5), (__int64)(v58 - 6));
      v60 = (unsigned __int64 *)v59[1];
      v61 = *v59 & 0xFFFFFFFFFFFFFFF8LL;
      *v60 = v61 | *v60 & 7;
      *(_QWORD *)(v61 + 8) = v60;
      *v59 &= 7uLL;
      v59[1] = 0;
      sub_AD0030((__int64)(v59 - 6));
      sub_BD7260(v59 - 6);
      result = (unsigned __int64 *)sub_BD2DD0(v59 - 6);
      v58 = v86;
    }
    while ( v17 != v86 );
  }
  for ( i = a1[4]; v12 != i; result = (unsigned __int64 *)sub_BD2DD0(v63 - 7) )
  {
    v63 = i;
    i = (_QWORD *)i[1];
    sub_BA8570((__int64)(a1 + 3), (__int64)(v63 - 7));
    v64 = (unsigned __int64 *)v63[1];
    v65 = *v63 & 0xFFFFFFFFFFFFFFF8LL;
    *v64 = v65 | *v64 & 7;
    *(_QWORD *)(v65 + 8) = v64;
    *v63 &= 7uLL;
    v63[1] = 0;
    sub_B2E780(v63 - 7);
  }
  for ( j = a1[2]; (_QWORD *)v87 != j; result = (unsigned __int64 *)sub_BD2DD0(v67 - 7) )
  {
    v67 = j;
    j = (_QWORD *)j[1];
    sub_BA85F0(v87, (__int64)(v67 - 7));
    v68 = (unsigned __int64 *)v67[1];
    v69 = *v67 & 0xFFFFFFFFFFFFFFF8LL;
    *v68 = v69 | *v68 & 7;
    *(_QWORD *)(v69 + 8) = v68;
    *v67 &= 7uLL;
    v67[1] = 0;
    sub_B30220((__int64)(v67 - 7));
    *((_DWORD *)v67 - 13) = *((_DWORD *)v67 - 13) & 0xF8000000 | 1;
    sub_B2F9E0((__int64)(v67 - 7), (__int64)(v67 - 7), v70, v71);
  }
  return result;
}
