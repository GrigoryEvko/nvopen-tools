// Function: sub_1880140
// Address: 0x1880140
//
__int64 *__fastcall sub_1880140(__int64 *a1, char *a2, int *a3)
{
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // rdx
  bool v7; // cf
  unsigned __int64 v8; // rax
  char *v9; // rdx
  __int64 v10; // rbx
  char *v11; // rax
  int v12; // ecx
  __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // rdx
  __int64 v16; // rdx
  __int64 v17; // rdx
  __int64 v18; // rdx
  __int64 v19; // rdx
  __int64 v20; // rdx
  __int64 v21; // rdx
  __int64 v22; // rdx
  __int64 v23; // rdx
  __int64 v24; // rdx
  __int64 v25; // rdx
  __int64 v26; // rdx
  __int64 v27; // rdx
  __int64 v28; // rdx
  __int64 v29; // rdx
  __int64 v30; // rdx
  char *v31; // r15
  __int64 v32; // rbx
  __int64 v33; // rdx
  __int64 v34; // rdx
  __int64 v35; // rdx
  __int64 v36; // rdx
  __int64 v37; // rdx
  __int64 v38; // r12
  __int64 v39; // r14
  __int64 v40; // rdi
  __int64 v41; // rdi
  __int64 v42; // rdi
  __int64 v43; // rdi
  __int64 v44; // rdi
  __int64 v45; // rdx
  __int64 v46; // r12
  __int64 v47; // r14
  __int64 v48; // rdi
  __int64 v49; // rsi
  char *v50; // rax
  __int64 v51; // rdx
  int v52; // ecx
  __int64 v54; // rbx
  __int64 v55; // rax
  __int64 v56; // [rsp+8h] [rbp-58h]
  char *v58; // [rsp+18h] [rbp-48h]
  __int64 v59; // [rsp+20h] [rbp-40h]
  char *v60; // [rsp+28h] [rbp-38h]

  v60 = (char *)a1[1];
  v58 = (char *)*a1;
  v4 = 0x86BCA1AF286BCA1BLL * ((__int64)&v60[-*a1] >> 3);
  if ( v4 == 0xD79435E50D7943LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v5 = 1;
  if ( v4 )
    v5 = 0x86BCA1AF286BCA1BLL * ((__int64)&v60[-*a1] >> 3);
  v7 = __CFADD__(v5, v4);
  v8 = v5 - 0x79435E50D79435E5LL * ((__int64)&v60[-*a1] >> 3);
  v9 = (char *)(a2 - v58);
  if ( v7 )
  {
    v54 = 0x7FFFFFFFFFFFFFC8LL;
  }
  else
  {
    if ( !v8 )
    {
      v56 = 0;
      v10 = 152;
      v59 = 0;
      goto LABEL_7;
    }
    if ( v8 > 0xD79435E50D7943LL )
      v8 = 0xD79435E50D7943LL;
    v54 = 152 * v8;
  }
  v55 = sub_22077B0(v54);
  v9 = (char *)(a2 - v58);
  v59 = v55;
  v56 = v55 + v54;
  v10 = v55 + 152;
LABEL_7:
  v11 = &v9[v59];
  if ( &v9[v59] )
  {
    v12 = *a3;
    v11[6] = *((_BYTE *)a3 + 6);
    v13 = *((_QWORD *)a3 + 1);
    *(_DWORD *)v11 = v12;
    LOWORD(v12) = *((_WORD *)a3 + 2);
    *((_QWORD *)v11 + 1) = v13;
    v14 = *((_QWORD *)a3 + 2);
    *((_WORD *)v11 + 2) = v12;
    *((_QWORD *)v11 + 2) = v14;
    v15 = *((_QWORD *)a3 + 3);
    *((_QWORD *)a3 + 2) = 0;
    *((_QWORD *)v11 + 3) = v15;
    v16 = *((_QWORD *)a3 + 4);
    *((_QWORD *)a3 + 3) = 0;
    *((_QWORD *)v11 + 4) = v16;
    v17 = *((_QWORD *)a3 + 5);
    *((_QWORD *)a3 + 1) = 0;
    *((_QWORD *)v11 + 5) = v17;
    v18 = *((_QWORD *)a3 + 6);
    *((_QWORD *)a3 + 5) = 0;
    *((_QWORD *)v11 + 6) = v18;
    v19 = *((_QWORD *)a3 + 7);
    *((_QWORD *)a3 + 6) = 0;
    *((_QWORD *)v11 + 7) = v19;
    v20 = *((_QWORD *)a3 + 8);
    *((_QWORD *)a3 + 4) = 0;
    *((_QWORD *)v11 + 8) = v20;
    v21 = *((_QWORD *)a3 + 9);
    *((_QWORD *)a3 + 8) = 0;
    *((_QWORD *)v11 + 9) = v21;
    *((_QWORD *)a3 + 9) = 0;
    *((_QWORD *)a3 + 7) = 0;
    v22 = *((_QWORD *)a3 + 10);
    *((_QWORD *)a3 + 10) = 0;
    *((_QWORD *)v11 + 10) = v22;
    v23 = *((_QWORD *)a3 + 11);
    *((_QWORD *)a3 + 11) = 0;
    *((_QWORD *)v11 + 11) = v23;
    v24 = *((_QWORD *)a3 + 12);
    *((_QWORD *)a3 + 12) = 0;
    *((_QWORD *)v11 + 12) = v24;
    v25 = *((_QWORD *)a3 + 13);
    *((_QWORD *)a3 + 13) = 0;
    *((_QWORD *)v11 + 13) = v25;
    v26 = *((_QWORD *)a3 + 14);
    *((_QWORD *)a3 + 14) = 0;
    *((_QWORD *)v11 + 14) = v26;
    v27 = *((_QWORD *)a3 + 15);
    *((_QWORD *)a3 + 15) = 0;
    *((_QWORD *)v11 + 15) = v27;
    v28 = *((_QWORD *)a3 + 16);
    *((_QWORD *)a3 + 16) = 0;
    *((_QWORD *)v11 + 16) = v28;
    v29 = *((_QWORD *)a3 + 17);
    *((_QWORD *)a3 + 17) = 0;
    *((_QWORD *)v11 + 17) = v29;
    v30 = *((_QWORD *)a3 + 18);
    *((_QWORD *)a3 + 18) = 0;
    *((_QWORD *)v11 + 18) = v30;
  }
  v31 = v58;
  if ( a2 != v58 )
  {
    v32 = v59;
    if ( !v59 )
      goto LABEL_29;
LABEL_11:
    *(_DWORD *)v32 = *(_DWORD *)v31;
    *(_BYTE *)(v32 + 4) = v31[4];
    *(_BYTE *)(v32 + 5) = v31[5];
    *(_BYTE *)(v32 + 6) = v31[6];
    *(_QWORD *)(v32 + 8) = *((_QWORD *)v31 + 1);
    *(_QWORD *)(v32 + 16) = *((_QWORD *)v31 + 2);
    *(_QWORD *)(v32 + 24) = *((_QWORD *)v31 + 3);
    v33 = *((_QWORD *)v31 + 4);
    *((_QWORD *)v31 + 3) = 0;
    *((_QWORD *)v31 + 2) = 0;
    *((_QWORD *)v31 + 1) = 0;
    *(_QWORD *)(v32 + 32) = v33;
    *(_QWORD *)(v32 + 40) = *((_QWORD *)v31 + 5);
    *(_QWORD *)(v32 + 48) = *((_QWORD *)v31 + 6);
    v34 = *((_QWORD *)v31 + 7);
    *((_QWORD *)v31 + 6) = 0;
    *((_QWORD *)v31 + 5) = 0;
    *((_QWORD *)v31 + 4) = 0;
    *(_QWORD *)(v32 + 56) = v34;
    *(_QWORD *)(v32 + 64) = *((_QWORD *)v31 + 8);
    *(_QWORD *)(v32 + 72) = *((_QWORD *)v31 + 9);
    *((_QWORD *)v31 + 9) = 0;
    v35 = *((_QWORD *)v31 + 10);
    *((_QWORD *)v31 + 8) = 0;
    *((_QWORD *)v31 + 7) = 0;
    *(_QWORD *)(v32 + 80) = v35;
    *(_QWORD *)(v32 + 88) = *((_QWORD *)v31 + 11);
    *(_QWORD *)(v32 + 96) = *((_QWORD *)v31 + 12);
    v36 = *((_QWORD *)v31 + 13);
    *((_QWORD *)v31 + 12) = 0;
    *((_QWORD *)v31 + 11) = 0;
    *((_QWORD *)v31 + 10) = 0;
    *(_QWORD *)(v32 + 104) = v36;
    *(_QWORD *)(v32 + 112) = *((_QWORD *)v31 + 14);
    *(_QWORD *)(v32 + 120) = *((_QWORD *)v31 + 15);
    v37 = *((_QWORD *)v31 + 16);
    *((_QWORD *)v31 + 15) = 0;
    *((_QWORD *)v31 + 14) = 0;
    *((_QWORD *)v31 + 13) = 0;
    *(_QWORD *)(v32 + 128) = v37;
    *(_QWORD *)(v32 + 136) = *((_QWORD *)v31 + 17);
    *(_QWORD *)(v32 + 144) = *((_QWORD *)v31 + 18);
    *((_QWORD *)v31 + 18) = 0;
    *((_QWORD *)v31 + 17) = 0;
    *((_QWORD *)v31 + 16) = 0;
    while ( 1 )
    {
      v38 = *((_QWORD *)v31 + 14);
      v39 = *((_QWORD *)v31 + 13);
      if ( v38 != v39 )
      {
        do
        {
          v40 = *(_QWORD *)(v39 + 16);
          if ( v40 )
            j_j___libc_free_0(v40, *(_QWORD *)(v39 + 32) - v40);
          v39 += 40;
        }
        while ( v38 != v39 );
        v39 = *((_QWORD *)v31 + 13);
      }
      if ( v39 )
        j_j___libc_free_0(v39, *((_QWORD *)v31 + 15) - v39);
      v41 = *((_QWORD *)v31 + 10);
      if ( v41 )
        j_j___libc_free_0(v41, *((_QWORD *)v31 + 12) - v41);
      v42 = *((_QWORD *)v31 + 7);
      if ( v42 )
        j_j___libc_free_0(v42, *((_QWORD *)v31 + 9) - v42);
      v43 = *((_QWORD *)v31 + 4);
      if ( v43 )
        j_j___libc_free_0(v43, *((_QWORD *)v31 + 6) - v43);
      v44 = *((_QWORD *)v31 + 1);
      if ( v44 )
        j_j___libc_free_0(v44, *((_QWORD *)v31 + 3) - v44);
      v31 += 152;
      v45 = v32 + 152;
      if ( v31 == a2 )
        break;
      v32 += 152;
      if ( v45 )
        goto LABEL_11;
LABEL_29:
      v46 = *((_QWORD *)v31 + 17);
      v47 = *((_QWORD *)v31 + 16);
      if ( v46 == v47 )
      {
        v49 = *((_QWORD *)v31 + 18) - v47;
      }
      else
      {
        do
        {
          v48 = *(_QWORD *)(v47 + 16);
          if ( v48 )
            j_j___libc_free_0(v48, *(_QWORD *)(v47 + 32) - v48);
          v47 += 40;
        }
        while ( v46 != v47 );
        v47 = *((_QWORD *)v31 + 16);
        v49 = *((_QWORD *)v31 + 18) - v47;
      }
      if ( v47 )
        j_j___libc_free_0(v47, v49);
    }
    v10 = v32 + 304;
  }
  if ( a2 != v60 )
  {
    v50 = a2;
    v51 = v10;
    do
    {
      v52 = *(_DWORD *)v50;
      v51 += 152;
      v50 += 152;
      *(_DWORD *)(v51 - 152) = v52;
      *(_BYTE *)(v51 - 148) = *(v50 - 148);
      *(_BYTE *)(v51 - 147) = *(v50 - 147);
      *(_BYTE *)(v51 - 146) = *(v50 - 146);
      *(_QWORD *)(v51 - 144) = *((_QWORD *)v50 - 18);
      *(_QWORD *)(v51 - 136) = *((_QWORD *)v50 - 17);
      *(_QWORD *)(v51 - 128) = *((_QWORD *)v50 - 16);
      *(_QWORD *)(v51 - 120) = *((_QWORD *)v50 - 15);
      *(_QWORD *)(v51 - 112) = *((_QWORD *)v50 - 14);
      *(_QWORD *)(v51 - 104) = *((_QWORD *)v50 - 13);
      *(_QWORD *)(v51 - 96) = *((_QWORD *)v50 - 12);
      *(_QWORD *)(v51 - 88) = *((_QWORD *)v50 - 11);
      *(_QWORD *)(v51 - 80) = *((_QWORD *)v50 - 10);
      *(_QWORD *)(v51 - 72) = *((_QWORD *)v50 - 9);
      *(_QWORD *)(v51 - 64) = *((_QWORD *)v50 - 8);
      *(_QWORD *)(v51 - 56) = *((_QWORD *)v50 - 7);
      *(_QWORD *)(v51 - 48) = *((_QWORD *)v50 - 6);
      *(_QWORD *)(v51 - 40) = *((_QWORD *)v50 - 5);
      *(_QWORD *)(v51 - 32) = *((_QWORD *)v50 - 4);
      *(_QWORD *)(v51 - 24) = *((_QWORD *)v50 - 3);
      *(_QWORD *)(v51 - 16) = *((_QWORD *)v50 - 2);
      *(_QWORD *)(v51 - 8) = *((_QWORD *)v50 - 1);
    }
    while ( v50 != v60 );
    v10 += 152 * (((0x6BCA1AF286BCA1BLL * ((unsigned __int64)(v50 - a2 - 152) >> 3)) & 0x1FFFFFFFFFFFFFFFLL) + 1);
  }
  if ( v58 )
    j_j___libc_free_0(v58, a1[2] - (_QWORD)v58);
  a1[1] = v10;
  *a1 = v59;
  a1[2] = v56;
  return a1;
}
