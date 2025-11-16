// Function: sub_1A974C0
// Address: 0x1a974c0
//
__int64 __fastcall sub_1A974C0(__int64 *a1, char *a2, __int64 *a3)
{
  __int64 v3; // rcx
  char *v5; // rbx
  __int64 v6; // rax
  __int64 v7; // rdx
  char *v9; // r14
  bool v10; // cf
  unsigned __int64 v11; // rax
  _BYTE *v12; // rsi
  signed __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // rcx
  __int64 v18; // rdx
  __int64 v19; // rdx
  __int64 v20; // rdx
  char *v21; // r13
  __int64 v22; // r15
  void *v23; // rax
  __int64 v24; // rdx
  unsigned __int64 v25; // rdi
  __int64 v26; // rdx
  unsigned __int64 v27; // rbx
  char *v28; // rcx
  __int64 v29; // rbx
  __int64 v30; // rax
  __int64 v31; // r15
  void *v32; // rax
  __int64 v33; // rdx
  const void *v34; // rsi
  _BYTE *v35; // rax
  unsigned __int64 v36; // r12
  __int64 v37; // rax
  char *v38; // rcx
  size_t v39; // r13
  __int64 v40; // rax
  char *i; // r12
  __int64 v42; // rdi
  __int64 v43; // rdi
  __int64 result; // rax
  __int64 v45; // rdi
  __int64 v46; // rax
  char *v47; // [rsp+8h] [rbp-58h]
  __int64 v48; // [rsp+10h] [rbp-50h]
  __int64 v50; // [rsp+20h] [rbp-40h]
  char *v51; // [rsp+28h] [rbp-38h]

  v3 = 0x1FFFFFFFFFFFFFFLL;
  v5 = (char *)a1[1];
  v51 = (char *)*a1;
  v6 = (__int64)&v5[-*a1] >> 6;
  if ( v6 == 0x1FFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v7 = 1;
  v9 = a2;
  if ( v6 )
    v7 = (__int64)&v5[-*a1] >> 6;
  v10 = __CFADD__(v7, v6);
  v11 = v7 + v6;
  v12 = (_BYTE *)v10;
  v48 = v11;
  v13 = a2 - v51;
  if ( v10 )
  {
    v45 = 0x7FFFFFFFFFFFFFC0LL;
    v48 = 0x1FFFFFFFFFFFFFFLL;
  }
  else
  {
    if ( !v11 )
    {
      v50 = 0;
      goto LABEL_7;
    }
    if ( v11 <= 0x1FFFFFFFFFFFFFFLL )
      v3 = v11;
    v48 = v3;
    v45 = v3 << 6;
  }
  v46 = sub_22077B0(v45);
  v13 = a2 - v51;
  v50 = v46;
LABEL_7:
  v14 = v50 + v13;
  if ( v50 + v13 )
  {
    v15 = *a3;
    v16 = a3[2];
    *(_QWORD *)(v14 + 8) = 1;
    ++a3[1];
    *(_QWORD *)v14 = v15;
    LODWORD(v15) = *((_DWORD *)a3 + 8);
    *(_QWORD *)(v14 + 16) = v16;
    v17 = a3[3];
    *(_DWORD *)(v14 + 32) = v15;
    v18 = a3[5];
    *(_QWORD *)(v14 + 24) = v17;
    *(_QWORD *)(v14 + 40) = v18;
    v19 = a3[6];
    a3[2] = 0;
    *(_QWORD *)(v14 + 48) = v19;
    v20 = a3[7];
    a3[3] = 0;
    *((_DWORD *)a3 + 8) = 0;
    *(_QWORD *)(v14 + 56) = v20;
    a3[7] = 0;
    a3[6] = 0;
    a3[5] = 0;
  }
  v21 = v51;
  v22 = v50;
  if ( a2 != v51 )
  {
    v47 = v5;
    do
    {
      if ( v22 )
      {
        v30 = *(_QWORD *)v21;
        *(_QWORD *)(v22 + 8) = 0;
        *(_DWORD *)(v22 + 32) = 0;
        *(_QWORD *)v22 = v30;
        *(_QWORD *)(v22 + 16) = 0;
        *(_DWORD *)(v22 + 24) = 0;
        *(_DWORD *)(v22 + 28) = 0;
        j___libc_free_0(0);
        v25 = *((unsigned int *)v21 + 8);
        *(_DWORD *)(v22 + 32) = v25;
        if ( (_DWORD)v25 )
        {
          v23 = (void *)sub_22077B0(8 * v25);
          v24 = *(unsigned int *)(v22 + 32);
          *(_QWORD *)(v22 + 16) = v23;
          v25 = (unsigned __int64)v23;
          *(_DWORD *)(v22 + 24) = *((_DWORD *)v21 + 6);
          *(_DWORD *)(v22 + 28) = *((_DWORD *)v21 + 7);
          v12 = (_BYTE *)*((_QWORD *)v21 + 2);
          memcpy(v23, v12, 8 * v24);
        }
        else
        {
          *(_QWORD *)(v22 + 16) = 0;
          *(_DWORD *)(v22 + 24) = 0;
          *(_DWORD *)(v22 + 28) = 0;
        }
        v26 = *((_QWORD *)v21 + 6);
        v27 = v26 - *((_QWORD *)v21 + 5);
        *(_QWORD *)(v22 + 40) = 0;
        *(_QWORD *)(v22 + 48) = 0;
        *(_QWORD *)(v22 + 56) = 0;
        if ( v27 )
        {
          if ( v27 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_48:
            sub_4261EA(v25, v12, v26);
          v28 = (char *)sub_22077B0(v27);
        }
        else
        {
          v28 = 0;
        }
        *(_QWORD *)(v22 + 40) = v28;
        *(_QWORD *)(v22 + 48) = v28;
        *(_QWORD *)(v22 + 56) = &v28[v27];
        v12 = (_BYTE *)*((_QWORD *)v21 + 5);
        v29 = *((_QWORD *)v21 + 6) - (_QWORD)v12;
        if ( *((_BYTE **)v21 + 6) != v12 )
          v28 = (char *)memmove(v28, v12, *((_QWORD *)v21 + 6) - (_QWORD)v12);
        *(_QWORD *)(v22 + 48) = &v28[v29];
      }
      v21 += 64;
      v22 += 64;
    }
    while ( a2 != v21 );
    v5 = v47;
  }
  v31 = v22 + 64;
  if ( a2 != v5 )
  {
    do
    {
      v40 = *(_QWORD *)v9;
      *(_QWORD *)(v31 + 8) = 0;
      *(_DWORD *)(v31 + 32) = 0;
      *(_QWORD *)v31 = v40;
      *(_QWORD *)(v31 + 16) = 0;
      *(_DWORD *)(v31 + 24) = 0;
      *(_DWORD *)(v31 + 28) = 0;
      j___libc_free_0(0);
      v25 = *((unsigned int *)v9 + 8);
      *(_DWORD *)(v31 + 32) = v25;
      if ( (_DWORD)v25 )
      {
        v32 = (void *)sub_22077B0(8 * v25);
        v33 = *(unsigned int *)(v31 + 32);
        v34 = (const void *)*((_QWORD *)v9 + 2);
        *(_QWORD *)(v31 + 16) = v32;
        v25 = (unsigned __int64)v32;
        *(_DWORD *)(v31 + 24) = *((_DWORD *)v9 + 6);
        *(_DWORD *)(v31 + 28) = *((_DWORD *)v9 + 7);
        memcpy(v32, v34, 8 * v33);
      }
      else
      {
        *(_QWORD *)(v31 + 16) = 0;
        *(_DWORD *)(v31 + 24) = 0;
        *(_DWORD *)(v31 + 28) = 0;
      }
      v35 = (_BYTE *)*((_QWORD *)v9 + 6);
      v12 = (_BYTE *)*((_QWORD *)v9 + 5);
      *(_QWORD *)(v31 + 40) = 0;
      *(_QWORD *)(v31 + 48) = 0;
      *(_QWORD *)(v31 + 56) = 0;
      v36 = v35 - v12;
      if ( v35 == v12 )
      {
        v39 = 0;
        v38 = 0;
      }
      else
      {
        if ( v36 > 0x7FFFFFFFFFFFFFF8LL )
          goto LABEL_48;
        v37 = sub_22077B0(v36);
        v12 = (_BYTE *)*((_QWORD *)v9 + 5);
        v38 = (char *)v37;
        v35 = (_BYTE *)*((_QWORD *)v9 + 6);
        v39 = v35 - v12;
      }
      *(_QWORD *)(v31 + 40) = v38;
      *(_QWORD *)(v31 + 48) = v38;
      *(_QWORD *)(v31 + 56) = &v38[v36];
      if ( v12 != v35 )
        v38 = (char *)memmove(v38, v12, v39);
      v9 += 64;
      v31 += 64;
      *(_QWORD *)(v31 - 16) = &v38[v39];
    }
    while ( v5 != v9 );
  }
  for ( i = v51; i != v5; i += 64 )
  {
    v42 = *((_QWORD *)i + 5);
    if ( v42 )
      j_j___libc_free_0(v42, *((_QWORD *)i + 7) - v42);
    v43 = *((_QWORD *)i + 2);
    j___libc_free_0(v43);
  }
  if ( v51 )
    j_j___libc_free_0(v51, a1[2] - (_QWORD)v51);
  result = v50 + (v48 << 6);
  *a1 = v50;
  a1[1] = v31;
  a1[2] = result;
  return result;
}
