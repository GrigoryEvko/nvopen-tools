// Function: sub_1633490
// Address: 0x1633490
//
unsigned __int64 *__fastcall sub_1633490(_QWORD **a1)
{
  _QWORD *v2; // r15
  _QWORD *v3; // r13
  unsigned __int64 *v4; // rcx
  unsigned __int64 v5; // rdx
  _QWORD *v6; // r15
  _QWORD **v7; // r13
  _QWORD *v8; // r14
  unsigned __int64 *v9; // rcx
  unsigned __int64 v10; // rdx
  _QWORD *v11; // rcx
  _QWORD **v12; // r14
  _QWORD *v13; // r15
  unsigned __int64 *v14; // rsi
  unsigned __int64 v15; // rdx
  _QWORD *v16; // r9
  _QWORD **v17; // r15
  __int64 v18; // r12
  unsigned __int64 *v19; // rsi
  unsigned __int64 v20; // rcx
  unsigned __int64 *v21; // rax
  unsigned __int64 *v22; // r12
  unsigned __int64 v23; // rcx
  _QWORD *v24; // rdi
  unsigned __int64 *v25; // r8
  unsigned __int64 v26; // r9
  _QWORD **v27; // rdi
  _QWORD **v28; // rdi
  _QWORD **v29; // rdi
  _QWORD *v30; // rdi
  _QWORD *v31; // rdi
  unsigned __int64 v32; // r8
  __int64 v33; // rdx
  __int64 v34; // rdx
  __int64 v35; // rax
  unsigned __int64 v36; // rdi
  _QWORD **v37; // rdi
  unsigned __int64 *result; // rax
  unsigned __int64 *v39; // r12
  unsigned __int64 v40; // rcx
  _QWORD *v41; // r9
  __int64 v42; // r12
  unsigned __int64 *v43; // rsi
  unsigned __int64 v44; // rcx
  _QWORD *v45; // rcx
  _QWORD *v46; // r15
  unsigned __int64 *v47; // rsi
  unsigned __int64 v48; // rdx
  _QWORD *i; // r12
  _QWORD *v50; // r14
  unsigned __int64 *v51; // rcx
  unsigned __int64 v52; // rdx
  _QWORD *j; // r14
  _QWORD *v54; // rbx
  unsigned __int64 *v55; // rcx
  unsigned __int64 v56; // rdx
  __int64 v57; // rdx
  __int64 v58; // rdx
  __int64 v59; // rax
  unsigned __int64 v60; // rdi
  __int64 v61; // [rsp+8h] [rbp-58h]
  __int64 v62; // [rsp+10h] [rbp-50h]
  __int64 v63; // [rsp+10h] [rbp-50h]
  _QWORD *v64; // [rsp+18h] [rbp-48h]
  unsigned __int64 *v65; // [rsp+18h] [rbp-48h]
  _QWORD *v66; // [rsp+18h] [rbp-48h]
  unsigned __int64 *v67; // [rsp+18h] [rbp-48h]
  __int64 v68; // [rsp+18h] [rbp-48h]
  unsigned __int64 *v69; // [rsp+18h] [rbp-48h]
  _QWORD *v70; // [rsp+18h] [rbp-48h]
  unsigned __int64 *v71; // [rsp+18h] [rbp-48h]
  _QWORD *v72; // [rsp+20h] [rbp-40h]
  _QWORD *v73; // [rsp+20h] [rbp-40h]
  unsigned __int64 *v74; // [rsp+20h] [rbp-40h]
  _QWORD *v75; // [rsp+20h] [rbp-40h]
  _QWORD *v76; // [rsp+20h] [rbp-40h]
  __int64 v77; // [rsp+28h] [rbp-38h]

  sub_1602690(*a1, (__int64)a1);
  sub_16332E0(a1);
  v2 = a1[2];
  v77 = (__int64)(a1 + 1);
  while ( (_QWORD *)v77 != v2 )
  {
    v3 = v2;
    v2 = (_QWORD *)v2[1];
    sub_1631C10(v77, (__int64)(v3 - 7));
    v4 = (unsigned __int64 *)v3[1];
    v5 = *v3 & 0xFFFFFFFFFFFFFFF8LL;
    *v4 = v5 | *v4 & 7;
    *(_QWORD *)(v5 + 8) = v4;
    *v3 &= 7uLL;
    v3[1] = 0;
    sub_15E5530((__int64)(v3 - 7));
    sub_159D9E0((__int64)(v3 - 7));
    sub_164BE60(v3 - 7);
    *((_DWORD *)v3 - 9) = *((_DWORD *)v3 - 9) & 0xF0000000 | 1;
    sub_1648B90(v3 - 7);
  }
  v6 = a1[4];
  v7 = a1 + 3;
  while ( v7 != v6 )
  {
    v8 = v6;
    v6 = (_QWORD *)v6[1];
    sub_1631B90((__int64)(a1 + 3), (__int64)(v8 - 7));
    v9 = (unsigned __int64 *)v8[1];
    v10 = *v8 & 0xFFFFFFFFFFFFFFF8LL;
    *v9 = v10 | *v9 & 7;
    *(_QWORD *)(v10 + 8) = v9;
    *v8 &= 7uLL;
    v8[1] = 0;
    sub_15E3C20(v8 - 7);
    sub_1648B90(v8 - 7);
  }
  v11 = a1[6];
  v12 = a1 + 5;
  if ( a1 + 5 != v11 )
  {
    do
    {
      v13 = v11;
      v72 = (_QWORD *)v11[1];
      sub_1631C90((__int64)(a1 + 5), (__int64)(v11 - 6));
      v14 = (unsigned __int64 *)v13[1];
      v15 = *v13 & 0xFFFFFFFFFFFFFFF8LL;
      *v14 = v15 | *v14 & 7;
      *(_QWORD *)(v15 + 8) = v14;
      *v13 &= 7uLL;
      v13[1] = 0;
      sub_159D9E0((__int64)(v13 - 6));
      sub_164BE60(v13 - 6);
      sub_1648B90(v13 - 6);
      v11 = v72;
    }
    while ( v12 != v72 );
  }
  v16 = a1[8];
  v17 = a1 + 7;
  if ( a1 + 7 != v16 )
  {
    do
    {
      v18 = (__int64)(v16 - 6);
      v73 = v16;
      v64 = (_QWORD *)v16[1];
      sub_1631D10((__int64)(a1 + 7), (__int64)(v16 - 6));
      v19 = (unsigned __int64 *)v73[1];
      v20 = *v73 & 0xFFFFFFFFFFFFFFF8LL;
      *v19 = v20 | *v19 & 7;
      *(_QWORD *)(v20 + 8) = v19;
      *v73 &= 7uLL;
      v73[1] = 0;
      sub_159D9E0(v18);
      sub_164BE60(v18);
      sub_1648B90(v18);
      v16 = v64;
    }
    while ( v17 != v64 );
  }
  v74 = (unsigned __int64 *)(a1 + 9);
  v21 = a1[10];
  if ( a1 + 9 != (_QWORD **)v21 )
  {
    do
    {
      v22 = v21;
      v65 = (unsigned __int64 *)v21[1];
      v23 = *v21 & 0xFFFFFFFFFFFFFFF8LL;
      *v65 = v23 | *v65 & 7;
      *(_QWORD *)(v23 + 8) = v65;
      *v21 &= 7u;
      v21[1] = 0;
      sub_161F5A0(v21);
      j_j___libc_free_0(v22, 64);
      v21 = v65;
    }
    while ( v74 != v65 );
  }
  v24 = a1[15];
  if ( v24 )
  {
    v66 = a1[15];
    sub_164D180(v24);
    j_j___libc_free_0(v66, 40);
  }
  v25 = a1[34];
  if ( v25 )
  {
    v26 = *v25;
    if ( *((_DWORD *)v25 + 3) )
    {
      v57 = *((unsigned int *)v25 + 2);
      if ( (_DWORD)v57 )
      {
        v58 = 8 * v57;
        v59 = 0;
        do
        {
          v60 = *(_QWORD *)(v26 + v59);
          if ( v60 != -8 && v60 )
          {
            v61 = v59;
            v63 = v58;
            v71 = v25;
            _libc_free(v60);
            v25 = v71;
            v59 = v61;
            v58 = v63;
            v26 = *v71;
          }
          v59 += 8;
        }
        while ( v58 != v59 );
      }
    }
    v67 = v25;
    _libc_free(v26);
    j_j___libc_free_0(v67, 32);
  }
  sub_15A93E0(a1 + 35);
  v27 = (_QWORD **)a1[30];
  if ( v27 != a1 + 32 )
    j_j___libc_free_0(v27, (char *)a1[32] + 1);
  v28 = (_QWORD **)a1[26];
  if ( v28 != a1 + 28 )
    j_j___libc_free_0(v28, (char *)a1[28] + 1);
  v29 = (_QWORD **)a1[22];
  if ( v29 != a1 + 24 )
    j_j___libc_free_0(v29, (char *)a1[24] + 1);
  v30 = a1[21];
  if ( v30 )
    (*(void (__fastcall **)(_QWORD *))(*v30 + 8LL))(v30);
  v31 = a1[20];
  if ( v31 )
    (*(void (__fastcall **)(_QWORD *))(*v31 + 8LL))(v31);
  v32 = (unsigned __int64)a1[16];
  if ( *((_DWORD *)a1 + 35) )
  {
    v33 = *((unsigned int *)a1 + 34);
    if ( (_DWORD)v33 )
    {
      v34 = 8 * v33;
      v35 = 0;
      do
      {
        v36 = *(_QWORD *)(v32 + v35);
        if ( v36 != -8 && v36 )
        {
          v62 = v34;
          v68 = v35;
          _libc_free(v36);
          v32 = (unsigned __int64)a1[16];
          v34 = v62;
          v35 = v68;
        }
        v35 += 8;
      }
      while ( v35 != v34 );
    }
  }
  _libc_free(v32);
  v37 = (_QWORD **)a1[11];
  if ( v37 != a1 + 13 )
    j_j___libc_free_0(v37, (char *)a1[13] + 1);
  result = a1[10];
  if ( v74 != result )
  {
    do
    {
      v39 = result;
      v69 = (unsigned __int64 *)result[1];
      v40 = *result & 0xFFFFFFFFFFFFFFF8LL;
      *v69 = v40 | *v69 & 7;
      *(_QWORD *)(v40 + 8) = v69;
      *result &= 7u;
      result[1] = 0;
      sub_161F5A0(result);
      j_j___libc_free_0(v39, 64);
      result = v69;
    }
    while ( v74 != v69 );
  }
  v41 = a1[8];
  if ( v17 != v41 )
  {
    do
    {
      v42 = (__int64)(v41 - 6);
      v75 = v41;
      v70 = (_QWORD *)v41[1];
      sub_1631D10((__int64)(a1 + 7), (__int64)(v41 - 6));
      v43 = (unsigned __int64 *)v75[1];
      v44 = *v75 & 0xFFFFFFFFFFFFFFF8LL;
      *v43 = v44 | *v43 & 7;
      *(_QWORD *)(v44 + 8) = v43;
      *v75 &= 7uLL;
      v75[1] = 0;
      sub_159D9E0(v42);
      sub_164BE60(v42);
      result = (unsigned __int64 *)sub_1648B90(v42);
      v41 = v70;
    }
    while ( v17 != v70 );
  }
  v45 = a1[6];
  if ( v12 != v45 )
  {
    do
    {
      v46 = v45;
      v76 = (_QWORD *)v45[1];
      sub_1631C90((__int64)(a1 + 5), (__int64)(v45 - 6));
      v47 = (unsigned __int64 *)v46[1];
      v48 = *v46 & 0xFFFFFFFFFFFFFFF8LL;
      *v47 = v48 | *v47 & 7;
      *(_QWORD *)(v48 + 8) = v47;
      *v46 &= 7uLL;
      v46[1] = 0;
      sub_159D9E0((__int64)(v46 - 6));
      sub_164BE60(v46 - 6);
      result = (unsigned __int64 *)sub_1648B90(v46 - 6);
      v45 = v76;
    }
    while ( v12 != v76 );
  }
  for ( i = a1[4]; v7 != i; result = (unsigned __int64 *)sub_1648B90(v50 - 7) )
  {
    v50 = i;
    i = (_QWORD *)i[1];
    sub_1631B90((__int64)(a1 + 3), (__int64)(v50 - 7));
    v51 = (unsigned __int64 *)v50[1];
    v52 = *v50 & 0xFFFFFFFFFFFFFFF8LL;
    *v51 = v52 | *v51 & 7;
    *(_QWORD *)(v52 + 8) = v51;
    *v50 &= 7uLL;
    v50[1] = 0;
    sub_15E3C20(v50 - 7);
  }
  for ( j = a1[2]; (_QWORD *)v77 != j; result = (unsigned __int64 *)sub_1648B90(v54 - 7) )
  {
    v54 = j;
    j = (_QWORD *)j[1];
    sub_1631C10(v77, (__int64)(v54 - 7));
    v55 = (unsigned __int64 *)v54[1];
    v56 = *v54 & 0xFFFFFFFFFFFFFFF8LL;
    *v55 = v56 | *v55 & 7;
    *(_QWORD *)(v56 + 8) = v55;
    *v54 &= 7uLL;
    v54[1] = 0;
    sub_15E5530((__int64)(v54 - 7));
    sub_159D9E0((__int64)(v54 - 7));
    sub_164BE60(v54 - 7);
    *((_DWORD *)v54 - 9) = *((_DWORD *)v54 - 9) & 0xF0000000 | 1;
  }
  return result;
}
