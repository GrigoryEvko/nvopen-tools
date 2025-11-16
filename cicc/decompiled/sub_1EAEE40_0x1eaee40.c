// Function: sub_1EAEE40
// Address: 0x1eaee40
//
__int64 __fastcall sub_1EAEE40(const void **a1, char *a2, __int64 a3)
{
  char *v4; // r15
  char *v5; // r14
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rdx
  bool v9; // cf
  unsigned __int64 v10; // rax
  char *v11; // rdx
  unsigned __int64 v12; // rbx
  __int64 v13; // r8
  char *v14; // rdx
  __int64 v15; // rdx
  char *v16; // rax
  void *v17; // rdi
  size_t v18; // rdx
  __int64 v20; // rbx
  __int64 v21; // rax
  __int64 v22; // [rsp+8h] [rbp-48h]
  __int64 v23; // [rsp+10h] [rbp-40h]
  __int64 v24; // [rsp+10h] [rbp-40h]
  __int64 v25; // [rsp+18h] [rbp-38h]

  v4 = (char *)a1[1];
  v5 = (char *)*a1;
  v6 = 0xAAAAAAAAAAAAAAABLL * ((v4 - (_BYTE *)*a1) >> 2);
  if ( v6 == 0xAAAAAAAAAAAAAAALL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v7 = 1;
  if ( v6 )
    v7 = 0xAAAAAAAAAAAAAAABLL * (((_BYTE *)a1[1] - (_BYTE *)*a1) >> 2);
  v9 = __CFADD__(v7, v6);
  v10 = v7 - 0x5555555555555555LL * (((_BYTE *)a1[1] - (_BYTE *)*a1) >> 2);
  v11 = (char *)(a2 - v5);
  if ( v9 )
  {
    v20 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v10 )
    {
      v25 = 0;
      v12 = 12;
      v13 = 0;
      goto LABEL_7;
    }
    if ( v10 > 0xAAAAAAAAAAAAAAALL )
      v10 = 0xAAAAAAAAAAAAAAALL;
    v20 = 12 * v10;
  }
  v22 = a3;
  v21 = sub_22077B0(v20);
  v11 = (char *)(a2 - v5);
  a3 = v22;
  v13 = v21;
  v25 = v20 + v21;
  v12 = v21 + 12;
LABEL_7:
  v14 = &v11[v13];
  if ( v14 )
  {
    *(_QWORD *)v14 = *(_QWORD *)a3;
    *((_DWORD *)v14 + 2) = *(_DWORD *)(a3 + 8);
  }
  if ( a2 != v5 )
  {
    v15 = v13;
    v16 = v5;
    do
    {
      if ( v15 )
      {
        *(_QWORD *)v15 = *(_QWORD *)v16;
        *(_DWORD *)(v15 + 8) = *((_DWORD *)v16 + 2);
      }
      v16 += 12;
      v15 += 12;
    }
    while ( v16 != a2 );
    v12 = v13 + 4 * ((unsigned __int64)(a2 - 12 - v5) >> 2) + 24;
  }
  if ( a2 != v4 )
  {
    v17 = (void *)v12;
    v23 = v13;
    v18 = 4 * ((unsigned __int64)(v4 - a2 - 12) >> 2) + 12;
    v12 += v18;
    memcpy(v17, a2, v18);
    v13 = v23;
  }
  if ( v5 )
  {
    v24 = v13;
    j_j___libc_free_0(v5, (_BYTE *)a1[2] - v5);
    v13 = v24;
  }
  *a1 = (const void *)v13;
  a1[1] = (const void *)v12;
  a1[2] = (const void *)v25;
  return v25;
}
