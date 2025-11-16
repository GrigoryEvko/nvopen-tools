// Function: sub_31172F0
// Address: 0x31172f0
//
void __fastcall sub_31172F0(__int64 *a1, int **a2)
{
  int *v2; // rax
  int *v3; // rax
  int v4; // edi
  unsigned int v6; // edx
  char *v7; // r14
  char *v8; // rbx
  const void *v9; // r13
  __int64 v10; // rdi
  int *v11; // rdx
  int *v12; // rax
  int v13; // r10d
  int v14; // r12d
  _BYTE *v15; // r8
  int v16; // edx
  signed __int64 v17; // r9
  __int64 v18; // rax
  __int64 v19; // rcx
  bool v20; // cf
  unsigned __int64 v21; // rax
  unsigned __int64 v22; // r13
  char *v23; // rcx
  _QWORD *v24; // rax
  int *v25; // rsi
  __int64 v26; // rcx
  __int64 v27; // rdx
  unsigned __int64 v28; // rdi
  char *v29; // rax
  unsigned __int64 v30; // r13
  __int64 v31; // rax
  _BYTE *v33; // [rsp+8h] [rbp-A8h]
  int v34; // [rsp+10h] [rbp-A0h]
  int v35; // [rsp+14h] [rbp-9Ch]
  int v36; // [rsp+18h] [rbp-98h]
  int v37; // [rsp+18h] [rbp-98h]
  signed __int64 v38; // [rsp+18h] [rbp-98h]
  _BYTE *v39; // [rsp+20h] [rbp-90h]
  char *v40; // [rsp+20h] [rbp-90h]
  int v41; // [rsp+20h] [rbp-90h]
  __int64 v42; // [rsp+30h] [rbp-80h]
  int v43; // [rsp+38h] [rbp-78h]
  int v44; // [rsp+3Ch] [rbp-74h]
  unsigned int v45; // [rsp+44h] [rbp-6Ch] BYREF
  unsigned int *v46; // [rsp+48h] [rbp-68h] BYREF
  __int64 v47; // [rsp+50h] [rbp-60h] BYREF
  int v48; // [rsp+58h] [rbp-58h] BYREF
  _QWORD *v49; // [rsp+60h] [rbp-50h]
  int *v50; // [rsp+68h] [rbp-48h]
  int *v51; // [rsp+70h] [rbp-40h]
  __int64 v52; // [rsp+78h] [rbp-38h]

  v50 = &v48;
  v51 = &v48;
  v2 = *a2;
  v48 = 0;
  v3 = v2 + 1;
  v49 = 0;
  v52 = 0;
  v4 = *(v3 - 1);
  *a2 = v3;
  v34 = v4;
  if ( !v4 )
    goto LABEL_36;
  v44 = 0;
  while ( 2 )
  {
    v6 = *v3;
    v7 = 0;
    v8 = 0;
    *a2 = v3 + 1;
    v9 = 0;
    v45 = v6;
    v10 = *(_QWORD *)(v3 + 1);
    *a2 = v3 + 3;
    v11 = v3 + 4;
    v12 = v3 + 5;
    v42 = v10;
    LODWORD(v10) = *(v12 - 2);
    *a2 = v11;
    v13 = *(v12 - 1);
    v43 = v10;
    *a2 = v12;
    if ( !v13 )
      goto LABEL_22;
    v14 = 0;
    v15 = 0;
    while ( 1 )
    {
      v16 = *v12;
      *a2 = v12 + 1;
      if ( v7 != v8 )
        break;
      v17 = v7 - v15;
      v18 = (v7 - v15) >> 2;
      if ( v18 == 0x1FFFFFFFFFFFFFFFLL )
        sub_4262D8((__int64)"vector::_M_realloc_insert");
      v19 = 1;
      if ( v18 )
        v19 = (v7 - v15) >> 2;
      v20 = __CFADD__(v19, v18);
      v21 = v19 + v18;
      if ( v20 )
      {
        v30 = 0x7FFFFFFFFFFFFFFCLL;
      }
      else
      {
        if ( !v21 )
        {
          v22 = 0;
          v23 = 0;
          goto LABEL_16;
        }
        if ( v21 > 0x1FFFFFFFFFFFFFFFLL )
          v21 = 0x1FFFFFFFFFFFFFFFLL;
        v30 = 4 * v21;
      }
      v33 = v15;
      v35 = v13;
      v38 = v7 - v15;
      v41 = v16;
      v31 = sub_22077B0(v30);
      v16 = v41;
      v17 = v38;
      v13 = v35;
      v15 = v33;
      v23 = (char *)v31;
      v22 = v31 + v30;
LABEL_16:
      if ( &v23[v17] )
        *(_DWORD *)&v23[v17] = v16;
      v8 = &v23[v17 + 4];
      if ( v17 > 0 )
      {
        v36 = v13;
        v39 = v15;
        v29 = (char *)memmove(v23, v15, v17);
        v15 = v39;
        v13 = v36;
        v23 = v29;
      }
      else if ( !v15 )
      {
        goto LABEL_20;
      }
      v37 = v13;
      v40 = v23;
      j_j___libc_free_0((unsigned __int64)v15);
      v13 = v37;
      v23 = v40;
LABEL_20:
      ++v14;
      v7 = (char *)v22;
      v15 = v23;
      if ( v14 == v13 )
        goto LABEL_21;
LABEL_8:
      v12 = *a2;
    }
    if ( v8 )
      *(_DWORD *)v8 = v16;
    ++v14;
    v8 += 4;
    if ( v14 != v13 )
      goto LABEL_8;
LABEL_21:
    v9 = v15;
LABEL_22:
    v24 = v49;
    if ( !v49 )
    {
      v25 = &v48;
      goto LABEL_29;
    }
    v25 = &v48;
    do
    {
      while ( 1 )
      {
        v26 = v24[2];
        v27 = v24[3];
        if ( *((_DWORD *)v24 + 8) >= v45 )
          break;
        v24 = (_QWORD *)v24[3];
        if ( !v27 )
          goto LABEL_27;
      }
      v25 = (int *)v24;
      v24 = (_QWORD *)v24[2];
    }
    while ( v26 );
LABEL_27:
    if ( v25 == &v48 || v45 < v25[8] )
    {
LABEL_29:
      v46 = &v45;
      v25 = (int *)sub_3115870(&v47, (__int64)v25, &v46);
    }
    v28 = *((_QWORD *)v25 + 7);
    *((_QWORD *)v25 + 8) = v8;
    *((_QWORD *)v25 + 7) = v9;
    *((_QWORD *)v25 + 5) = v42;
    v25[12] = v43;
    *((_QWORD *)v25 + 9) = v7;
    if ( v28 )
      j_j___libc_free_0(v28);
    if ( ++v44 != v34 )
    {
      v3 = *a2;
      continue;
    }
    break;
  }
LABEL_36:
  sub_3116A00(a1, (__int64)&v47);
  sub_31152F0(v49);
}
