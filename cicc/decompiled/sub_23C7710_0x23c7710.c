// Function: sub_23C7710
// Address: 0x23c7710
//
unsigned __int64 __fastcall sub_23C7710(unsigned __int64 *a1, char *a2, __int64 *a3)
{
  char *v4; // r12
  __int64 v5; // rax
  char *v7; // r14
  __int64 v8; // rdi
  bool v9; // zf
  __int64 v10; // rax
  bool v11; // cf
  unsigned __int64 v12; // rax
  char *v13; // rsi
  __int64 v14; // rax
  char *v15; // rsi
  __int64 v16; // rdi
  char *v17; // rbx
  __int64 i; // r13
  int v19; // esi
  __int64 v20; // rcx
  __int64 v21; // rsi
  unsigned __int64 v22; // rdx
  unsigned __int64 v24; // r13
  __int64 v25; // rax
  unsigned __int64 v26; // r13
  unsigned __int64 v27; // [rsp+8h] [rbp-58h]
  unsigned __int64 v29; // [rsp+18h] [rbp-48h]
  __int64 v30; // [rsp+18h] [rbp-48h]
  __int64 v31; // [rsp+20h] [rbp-40h]
  unsigned __int64 v32; // [rsp+28h] [rbp-38h]

  v4 = (char *)a1[1];
  v29 = *a1;
  v5 = (__int64)&v4[-*a1] >> 4;
  if ( v5 == 0x7FFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v7 = a2;
  v8 = (__int64)&v4[-*a1] >> 4;
  v9 = v5 == 0;
  v10 = 1;
  if ( !v9 )
    v10 = v8;
  v11 = __CFADD__(v8, v10);
  v12 = v8 + v10;
  v13 = &a2[-v29];
  if ( v11 )
  {
    v24 = 0x7FFFFFFFFFFFFFF0LL;
  }
  else
  {
    if ( !v12 )
    {
      v27 = 0;
      v14 = 16;
      v31 = 0;
      goto LABEL_7;
    }
    if ( v12 > 0x7FFFFFFFFFFFFFFLL )
      v12 = 0x7FFFFFFFFFFFFFFLL;
    v24 = 16 * v12;
  }
  v25 = sub_22077B0(v24);
  v26 = v25 + v24;
  v31 = v25;
  v14 = v25 + 16;
  v27 = v26;
LABEL_7:
  v15 = &v13[v31];
  if ( v15 )
  {
    v16 = *a3;
    *a3 = 0;
    *(_QWORD *)v15 = v16;
    *((_DWORD *)v15 + 2) = *((_DWORD *)a3 + 2);
  }
  v17 = (char *)v29;
  if ( a2 != (char *)v29 )
  {
    for ( i = v31; ; i += 16 )
    {
      if ( i )
      {
        *(_QWORD *)i = *(_QWORD *)v17;
        v19 = *((_DWORD *)v17 + 2);
        *(_QWORD *)v17 = 0;
        *(_DWORD *)(i + 8) = v19;
      }
      if ( *(_QWORD *)v17 )
      {
        v32 = *(_QWORD *)v17;
        sub_C88FF0(*(_QWORD **)v17);
        j_j___libc_free_0(v32);
      }
      v17 += 16;
      if ( v17 == a2 )
        break;
    }
    v14 = i + 32;
  }
  if ( a2 != v4 )
  {
    v20 = v14;
    do
    {
      v21 = *(_QWORD *)v7;
      v7 += 16;
      v20 += 16;
      *(_QWORD *)(v20 - 16) = v21;
      *(_DWORD *)(v20 - 8) = *((_DWORD *)v7 - 2);
    }
    while ( v7 != v4 );
    v14 += v4 - a2;
  }
  v22 = v29;
  if ( v29 )
  {
    v30 = v14;
    j_j___libc_free_0(v22);
    v14 = v30;
  }
  a1[1] = v14;
  *a1 = v31;
  a1[2] = v27;
  return v27;
}
