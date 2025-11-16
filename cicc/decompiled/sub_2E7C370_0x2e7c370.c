// Function: sub_2E7C370
// Address: 0x2e7c370
//
unsigned __int64 *__fastcall sub_2E7C370(unsigned __int64 *a1, char *a2, __int64 *a3)
{
  char *v3; // r12
  __int64 v4; // rax
  char *v6; // r14
  __int64 v7; // rdi
  bool v8; // zf
  __int64 v9; // rax
  bool v10; // cf
  unsigned __int64 v11; // rax
  char *v12; // rsi
  __int64 v13; // rbx
  char *v14; // rsi
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  unsigned __int64 v18; // r13
  unsigned __int64 v19; // rbx
  int v20; // esi
  __int64 v21; // rax
  __int64 v22; // rcx
  unsigned __int64 v24; // rbx
  __int64 v25; // rax
  __int64 *v26; // [rsp+0h] [rbp-60h]
  unsigned __int64 v27; // [rsp+10h] [rbp-50h]
  unsigned __int64 v29; // [rsp+20h] [rbp-40h]
  unsigned __int64 v30; // [rsp+28h] [rbp-38h]

  v3 = (char *)a1[1];
  v29 = *a1;
  v4 = (__int64)&v3[-*a1] >> 5;
  if ( v4 == 0x3FFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v6 = a2;
  v7 = (__int64)&v3[-v29] >> 5;
  v8 = v4 == 0;
  v9 = 1;
  if ( !v8 )
    v9 = (__int64)&v3[-v29] >> 5;
  v10 = __CFADD__(v7, v9);
  v11 = v7 + v9;
  v12 = &a2[-v29];
  if ( v10 )
  {
    v24 = 0x7FFFFFFFFFFFFFE0LL;
  }
  else
  {
    if ( !v11 )
    {
      v27 = 0;
      v13 = 32;
      v30 = 0;
      goto LABEL_7;
    }
    if ( v11 > 0x3FFFFFFFFFFFFFFLL )
      v11 = 0x3FFFFFFFFFFFFFFLL;
    v24 = 32 * v11;
  }
  v26 = a3;
  v25 = sub_22077B0(v24);
  a3 = v26;
  v30 = v25;
  v27 = v25 + v24;
  v13 = v25 + 32;
LABEL_7:
  v14 = &v12[v30];
  if ( v14 )
  {
    v15 = *a3;
    *a3 = 0;
    *(_QWORD *)v14 = v15;
    v16 = a3[1];
    a3[1] = 0;
    *((_QWORD *)v14 + 1) = v16;
    v17 = a3[2];
    a3[2] = 0;
    *((_QWORD *)v14 + 2) = v17;
    *((_DWORD *)v14 + 6) = *((_DWORD *)a3 + 6);
  }
  if ( a2 != (char *)v29 )
  {
    v18 = v30;
    v19 = v29;
    while ( 1 )
    {
      if ( v18 )
      {
        *(_QWORD *)v18 = *(_QWORD *)v19;
        *(_QWORD *)(v18 + 8) = *(_QWORD *)(v19 + 8);
        *(_QWORD *)(v18 + 16) = *(_QWORD *)(v19 + 16);
        v20 = *(_DWORD *)(v19 + 24);
        *(_QWORD *)(v19 + 16) = 0;
        *(_QWORD *)(v19 + 8) = 0;
        *(_QWORD *)v19 = 0;
        *(_DWORD *)(v18 + 24) = v20;
      }
      if ( *(_QWORD *)v19 )
        j_j___libc_free_0(*(_QWORD *)v19);
      v19 += 32LL;
      if ( (char *)v19 == a2 )
        break;
      v18 += 32LL;
    }
    v13 = v18 + 64;
  }
  if ( a2 != v3 )
  {
    v21 = v13;
    do
    {
      v22 = *(_QWORD *)v6;
      v6 += 32;
      v21 += 32;
      *(_QWORD *)(v21 - 32) = v22;
      *(_QWORD *)(v21 - 24) = *((_QWORD *)v6 - 3);
      *(_QWORD *)(v21 - 16) = *((_QWORD *)v6 - 2);
      *(_DWORD *)(v21 - 8) = *((_DWORD *)v6 - 2);
    }
    while ( v6 != v3 );
    v13 += v3 - a2;
  }
  if ( v29 )
    j_j___libc_free_0(v29);
  *a1 = v30;
  a1[1] = v13;
  a1[2] = v27;
  return a1;
}
