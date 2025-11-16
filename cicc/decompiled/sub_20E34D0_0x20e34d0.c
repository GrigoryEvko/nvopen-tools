// Function: sub_20E34D0
// Address: 0x20e34d0
//
__int64 __fastcall sub_20E34D0(__int64 a1, char *a2, int *a3)
{
  __int64 v4; // rsi
  char *v5; // rbx
  __int64 v6; // rcx
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // r12
  _QWORD *v9; // r13
  bool v10; // cf
  unsigned __int64 v11; // r12
  char *v12; // r8
  char *v13; // r8
  int v14; // eax
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // r12
  char *v18; // rax
  char *v19; // r8
  char *v20; // rdi
  const void *v21; // r9
  __int64 v22; // rdx
  size_t v23; // rdx
  char *v24; // rax
  __int64 v25; // r15
  __int64 i; // r12
  __int64 v27; // rdi
  __int64 v28; // r8
  __int64 v29; // rdx
  char *v30; // rax
  int v31; // esi
  __int64 v32; // rdi
  __int64 v34; // rax
  int *v35; // [rsp+8h] [rbp-58h]
  char *v36; // [rsp+8h] [rbp-58h]
  int *v37; // [rsp+8h] [rbp-58h]
  char *v38; // [rsp+10h] [rbp-50h]
  size_t v39; // [rsp+10h] [rbp-50h]
  __int64 v40; // [rsp+18h] [rbp-48h]
  __int64 v41; // [rsp+20h] [rbp-40h]
  __int64 v42; // [rsp+20h] [rbp-40h]
  __int64 v43; // [rsp+28h] [rbp-38h]

  v4 = 0x2AAAAAAAAAAAAAALL;
  v5 = *(char **)(a1 + 8);
  v41 = *(_QWORD *)a1;
  v6 = (__int64)&v5[-*(_QWORD *)a1];
  v7 = 0xAAAAAAAAAAAAAAABLL * (v6 >> 4);
  if ( v7 == 0x2AAAAAAAAAAAAAALL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v8 = 1;
  v9 = (_QWORD *)a1;
  if ( v7 )
    v8 = 0xAAAAAAAAAAAAAAABLL * (v6 >> 4);
  v10 = __CFADD__(v7, v8);
  v11 = v7 + v8;
  v40 = v11;
  v12 = &a2[-v41];
  if ( v10 )
  {
    a1 = 0x7FFFFFFFFFFFFFE0LL;
    v40 = 0x2AAAAAAAAAAAAAALL;
  }
  else
  {
    if ( !v11 )
    {
      v43 = 0;
      goto LABEL_7;
    }
    if ( v11 <= 0x2AAAAAAAAAAAAAALL )
      v4 = v11;
    v40 = v4;
    a1 = 48 * v4;
  }
  v37 = a3;
  v34 = sub_22077B0(a1);
  v12 = &a2[-v41];
  a3 = v37;
  v43 = v34;
LABEL_7:
  v13 = &v12[v43];
  if ( v13 )
  {
    v14 = *a3;
    *((_QWORD *)v13 + 3) = 0;
    *((_QWORD *)v13 + 4) = 0;
    *(_DWORD *)v13 = v14;
    v15 = *((_QWORD *)a3 + 1);
    *((_QWORD *)v13 + 5) = 0;
    *((_QWORD *)v13 + 1) = v15;
    *((_DWORD *)v13 + 4) = a3[4];
    *((_DWORD *)v13 + 5) = a3[5];
    v16 = *((_QWORD *)a3 + 4) - *((_QWORD *)a3 + 3);
    v17 = v16;
    if ( v16 )
    {
      if ( v16 < 0 )
        sub_4261EA(a1, v4, a3);
      v35 = a3;
      v38 = v13;
      v18 = (char *)sub_22077B0(v16);
      v19 = v38;
      v20 = v18;
      v21 = (const void *)*((_QWORD *)v35 + 3);
      v22 = *((_QWORD *)v35 + 4);
      *((_QWORD *)v38 + 3) = v18;
      *((_QWORD *)v38 + 4) = v18;
      *((_QWORD *)v38 + 5) = &v18[v17];
      v23 = v22 - (_QWORD)v21;
      if ( v23 )
      {
        v36 = v38;
        v39 = v23;
        v24 = (char *)memmove(v18, v21, v23);
        v19 = v36;
        v23 = v39;
        v20 = v24;
      }
      *((_QWORD *)v19 + 4) = &v20[v23];
    }
    else
    {
      *((_QWORD *)v13 + 4) = 0;
    }
  }
  v25 = v41;
  for ( i = v43; a2 != (char *)v25; i = 48 )
  {
    while ( i )
    {
      *(_DWORD *)i = *(_DWORD *)v25;
      *(_QWORD *)(i + 8) = *(_QWORD *)(v25 + 8);
      *(_DWORD *)(i + 16) = *(_DWORD *)(v25 + 16);
      *(_DWORD *)(i + 20) = *(_DWORD *)(v25 + 20);
      *(_QWORD *)(i + 24) = *(_QWORD *)(v25 + 24);
      *(_QWORD *)(i + 32) = *(_QWORD *)(v25 + 32);
      *(_QWORD *)(i + 40) = *(_QWORD *)(v25 + 40);
      *(_QWORD *)(v25 + 40) = 0;
      *(_QWORD *)(v25 + 24) = 0;
LABEL_16:
      v25 += 48;
      i += 48;
      if ( a2 == (char *)v25 )
        goto LABEL_20;
    }
    v27 = *(_QWORD *)(v25 + 24);
    if ( !v27 )
      goto LABEL_16;
    j_j___libc_free_0(v27, *(_QWORD *)(v25 + 40) - v27);
    v25 += 48;
  }
LABEL_20:
  v28 = i + 48;
  if ( a2 != v5 )
  {
    v29 = i + 48;
    v30 = a2;
    do
    {
      v31 = *(_DWORD *)v30;
      v30 += 48;
      v29 += 48;
      *(_DWORD *)(v29 - 48) = v31;
      *(_QWORD *)(v29 - 40) = *((_QWORD *)v30 - 5);
      *(_DWORD *)(v29 - 32) = *((_DWORD *)v30 - 8);
      *(_DWORD *)(v29 - 28) = *((_DWORD *)v30 - 7);
      *(_QWORD *)(v29 - 24) = *((_QWORD *)v30 - 3);
      *(_QWORD *)(v29 - 16) = *((_QWORD *)v30 - 2);
      *(_QWORD *)(v29 - 8) = *((_QWORD *)v30 - 1);
    }
    while ( v5 != v30 );
    v28 += 16 * (3 * ((0xAAAAAAAAAAAAAABLL * ((unsigned __int64)(v5 - a2 - 48) >> 4)) & 0xFFFFFFFFFFFFFFFLL) + 3);
  }
  v32 = v41;
  if ( v41 )
  {
    v42 = v28;
    j_j___libc_free_0(v32, v9[2] - v32);
    v28 = v42;
  }
  v9[1] = v28;
  *v9 = v43;
  v9[2] = v43 + 48 * v40;
  return 48 * v40;
}
