// Function: sub_38DFBA0
// Address: 0x38dfba0
//
__int64 __fastcall sub_38DFBA0(unsigned __int64 a1, char *a2, __int64 *a3)
{
  __int64 v3; // rcx
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // rdx
  _QWORD *v7; // r12
  bool v8; // cf
  unsigned __int64 v9; // rax
  char *v10; // rbx
  signed __int64 v11; // rdx
  bool v12; // zf
  char *v13; // rbx
  char *v14; // r14
  __int64 v15; // rax
  __int64 v16; // r10
  __int64 v17; // rcx
  __int64 v18; // rax
  unsigned __int64 v19; // r15
  __int64 v20; // rax
  __int64 v21; // rbx
  __int64 v22; // r13
  __int64 v23; // r12
  __int64 *v24; // r15
  signed __int64 v25; // r14
  char *v26; // rdi
  char *v27; // rdx
  __int64 v28; // r14
  size_t v29; // rax
  __int64 v30; // rax
  unsigned __int64 v31; // r14
  __int64 i; // r15
  int v33; // eax
  __int64 v34; // rbx
  unsigned __int64 v35; // r13
  unsigned __int64 v36; // rdi
  __int64 v37; // rbx
  _QWORD *v38; // rax
  __int64 v39; // rdx
  __int64 v40; // rcx
  __int64 result; // rax
  __int64 v42; // rax
  char *v43; // [rsp+8h] [rbp-68h]
  __int64 *v44; // [rsp+10h] [rbp-60h]
  char *v45; // [rsp+10h] [rbp-60h]
  __int64 *v46; // [rsp+10h] [rbp-60h]
  _QWORD *v47; // [rsp+18h] [rbp-58h]
  __int64 v48; // [rsp+20h] [rbp-50h]
  _QWORD *v49; // [rsp+28h] [rbp-48h]
  unsigned __int64 v50; // [rsp+30h] [rbp-40h]
  __int64 v51; // [rsp+38h] [rbp-38h]

  v3 = 0x199999999999999LL;
  v49 = (_QWORD *)a1;
  v47 = *(_QWORD **)(a1 + 8);
  v50 = *(_QWORD *)a1;
  v5 = 0xCCCCCCCCCCCCCCCDLL * (((__int64)v47 - *(_QWORD *)a1) >> 4);
  if ( v5 == 0x199999999999999LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v6 = 1;
  v7 = a2;
  if ( v5 )
    v6 = 0xCCCCCCCCCCCCCCCDLL * (((__int64)v47 - *(_QWORD *)a1) >> 4);
  v8 = __CFADD__(v6, v5);
  v9 = v6 - 0x3333333333333333LL * (((__int64)v47 - *(_QWORD *)a1) >> 4);
  v48 = v9;
  v10 = &a2[-v50];
  v11 = v8;
  if ( v8 )
  {
    a1 = 0x7FFFFFFFFFFFFFD0LL;
    v48 = 0x199999999999999LL;
  }
  else
  {
    if ( !v9 )
    {
      v51 = 0;
      goto LABEL_7;
    }
    if ( v9 <= 0x199999999999999LL )
      v3 = v9;
    v48 = v3;
    a1 = 80 * v3;
  }
  v46 = a3;
  v42 = sub_22077B0(a1);
  a3 = v46;
  v51 = v42;
LABEL_7:
  v12 = &v10[v51] == 0;
  v13 = &v10[v51];
  v14 = v13;
  if ( !v12 )
  {
    v15 = *a3;
    v16 = a3[5];
    *((_QWORD *)v13 + 4) = 0;
    v17 = a3[4];
    *((_QWORD *)v13 + 5) = 0;
    *(_QWORD *)v13 = v15;
    v18 = a3[1];
    *((_QWORD *)v13 + 6) = 0;
    *((_QWORD *)v13 + 1) = v18;
    *((_QWORD *)v13 + 2) = a3[2];
    *((_QWORD *)v13 + 3) = a3[3];
    v19 = v16 - v17;
    if ( v16 == v17 )
    {
      v21 = 0;
    }
    else
    {
      if ( v19 > 0x7FFFFFFFFFFFFFE0LL )
LABEL_49:
        sub_4261EA(a1, a2, v11);
      a1 = v16 - v17;
      v44 = a3;
      v20 = sub_22077B0(v16 - v17);
      a3 = v44;
      v21 = v20;
      v16 = v44[5];
      v17 = v44[4];
    }
    *((_QWORD *)v14 + 4) = v21;
    *((_QWORD *)v14 + 5) = v21;
    *((_QWORD *)v14 + 6) = v21 + v19;
    if ( v16 != v17 )
    {
      v43 = a2;
      v22 = v16;
      v23 = v17;
      v24 = a3;
      v45 = v14;
      do
      {
        if ( v21 )
        {
          *(_DWORD *)v21 = *(_DWORD *)v23;
          *(_QWORD *)(v21 + 8) = *(_QWORD *)(v23 + 8);
          *(_DWORD *)(v21 + 16) = *(_DWORD *)(v23 + 16);
          *(_DWORD *)(v21 + 20) = *(_DWORD *)(v23 + 20);
          v11 = *(_QWORD *)(v23 + 32) - *(_QWORD *)(v23 + 24);
          *(_QWORD *)(v21 + 24) = 0;
          *(_QWORD *)(v21 + 32) = 0;
          v25 = v11;
          *(_QWORD *)(v21 + 40) = 0;
          if ( v11 )
          {
            if ( v11 < 0 )
              goto LABEL_49;
            v26 = (char *)sub_22077B0(v11);
          }
          else
          {
            v26 = 0;
          }
          v27 = &v26[v25];
          *(_QWORD *)(v21 + 24) = v26;
          v28 = 0;
          *(_QWORD *)(v21 + 32) = v26;
          *(_QWORD *)(v21 + 40) = v27;
          a2 = *(char **)(v23 + 24);
          v29 = *(_QWORD *)(v23 + 32) - (_QWORD)a2;
          if ( v29 )
          {
            v28 = *(_QWORD *)(v23 + 32) - (_QWORD)a2;
            v26 = (char *)memmove(v26, a2, v29);
          }
          a1 = (unsigned __int64)&v26[v28];
          *(_QWORD *)(v21 + 32) = a1;
        }
        v23 += 48;
        v21 += 48;
      }
      while ( v22 != v23 );
      v14 = v45;
      v7 = v43;
      a3 = v24;
    }
    v30 = a3[7];
    *((_QWORD *)v14 + 5) = v21;
    *((_QWORD *)v14 + 7) = v30;
    *((_QWORD *)v14 + 8) = a3[8];
    *((_WORD *)v14 + 36) = *((_WORD *)a3 + 36);
    *((_DWORD *)v14 + 19) = *((_DWORD *)a3 + 19);
  }
  v31 = v50;
  for ( i = v51; (_QWORD *)v31 != v7; i += 80 )
  {
    if ( i )
    {
      *(_QWORD *)i = *(_QWORD *)v31;
      *(_QWORD *)(i + 8) = *(_QWORD *)(v31 + 8);
      *(_QWORD *)(i + 16) = *(_QWORD *)(v31 + 16);
      *(_QWORD *)(i + 24) = *(_QWORD *)(v31 + 24);
      *(_QWORD *)(i + 32) = *(_QWORD *)(v31 + 32);
      *(_QWORD *)(i + 40) = *(_QWORD *)(v31 + 40);
      *(_QWORD *)(i + 48) = *(_QWORD *)(v31 + 48);
      v33 = *(_DWORD *)(v31 + 56);
      *(_QWORD *)(v31 + 48) = 0;
      *(_QWORD *)(v31 + 40) = 0;
      *(_QWORD *)(v31 + 32) = 0;
      *(_DWORD *)(i + 56) = v33;
      *(_DWORD *)(i + 60) = *(_DWORD *)(v31 + 60);
      *(_DWORD *)(i + 64) = *(_DWORD *)(v31 + 64);
      *(_DWORD *)(i + 68) = *(_DWORD *)(v31 + 68);
      *(_BYTE *)(i + 72) = *(_BYTE *)(v31 + 72);
      *(_BYTE *)(i + 73) = *(_BYTE *)(v31 + 73);
      *(_DWORD *)(i + 76) = *(_DWORD *)(v31 + 76);
    }
    v34 = *(_QWORD *)(v31 + 40);
    v35 = *(_QWORD *)(v31 + 32);
    if ( v34 != v35 )
    {
      do
      {
        v36 = *(_QWORD *)(v35 + 24);
        if ( v36 )
          j_j___libc_free_0(v36);
        v35 += 48LL;
      }
      while ( v34 != v35 );
      v35 = *(_QWORD *)(v31 + 32);
    }
    if ( v35 )
      j_j___libc_free_0(v35);
    v31 += 80LL;
  }
  v37 = i + 80;
  if ( v7 != v47 )
  {
    v38 = v7;
    v39 = i + 80;
    do
    {
      v40 = *v38;
      v38 += 10;
      v39 += 80;
      *(_QWORD *)(v39 - 80) = v40;
      *(_QWORD *)(v39 - 72) = *(v38 - 9);
      *(_QWORD *)(v39 - 64) = *(v38 - 8);
      *(_QWORD *)(v39 - 56) = *(v38 - 7);
      *(_QWORD *)(v39 - 48) = *(v38 - 6);
      *(_QWORD *)(v39 - 40) = *(v38 - 5);
      *(_QWORD *)(v39 - 32) = *(v38 - 4);
      *(_DWORD *)(v39 - 24) = *((_DWORD *)v38 - 6);
      *(_DWORD *)(v39 - 20) = *((_DWORD *)v38 - 5);
      *(_DWORD *)(v39 - 16) = *((_DWORD *)v38 - 4);
      *(_DWORD *)(v39 - 12) = *((_DWORD *)v38 - 3);
      *(_BYTE *)(v39 - 8) = *((_BYTE *)v38 - 8);
      *(_BYTE *)(v39 - 7) = *((_BYTE *)v38 - 7);
      *(_DWORD *)(v39 - 4) = *((_DWORD *)v38 - 1);
    }
    while ( v38 != v47 );
    v37 += 16
         * (5 * ((0xCCCCCCCCCCCCCCDLL * ((unsigned __int64)((char *)v38 - (char *)v7 - 80) >> 4)) & 0xFFFFFFFFFFFFFFFLL)
          + 5);
  }
  if ( v50 )
    j_j___libc_free_0(v50);
  *v49 = v51;
  result = v51 + 80 * v48;
  v49[1] = v37;
  v49[2] = result;
  return result;
}
