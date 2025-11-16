// Function: sub_25EFFB0
// Address: 0x25effb0
//
unsigned __int64 __fastcall sub_25EFFB0(
        unsigned __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6)
{
  __int64 v6; // rdx
  unsigned __int64 *v7; // r12
  unsigned __int64 v8; // r13
  __int64 v9; // rax
  __int64 v10; // r15
  bool v11; // zf
  __int64 v13; // rdi
  __int64 v14; // rax
  bool v15; // cf
  unsigned __int64 v16; // rax
  unsigned __int64 v17; // rsi
  __int64 v18; // rbx
  unsigned __int64 v19; // rax
  __int64 v20; // rbx
  __int64 v21; // rax
  __int64 v22; // rax
  int v23; // eax
  unsigned __int64 *i; // r15
  unsigned __int64 v26; // rbx
  unsigned __int64 v27; // [rsp+8h] [rbp-48h]
  __int64 v28; // [rsp+10h] [rbp-40h]
  __int64 v29; // [rsp+18h] [rbp-38h]

  v6 = 0x3FFFFFFFFFFFFFFLL;
  v7 = (unsigned __int64 *)a1[1];
  v8 = *a1;
  v9 = (__int64)((__int64)v7 - *a1) >> 5;
  if ( v9 == 0x3FFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v10 = a2;
  v11 = v9 == 0;
  v13 = (__int64)(a1[1] - *a1) >> 5;
  v14 = 1;
  if ( !v11 )
    v14 = v13;
  v15 = __CFADD__(v13, v14);
  v16 = v13 + v14;
  v17 = a2 - v8;
  if ( v15 )
  {
    v26 = 0x7FFFFFFFFFFFFFE0LL;
  }
  else
  {
    if ( !v16 )
    {
      v27 = 0;
      v18 = 32;
      v29 = 0;
      goto LABEL_7;
    }
    if ( v16 > 0x3FFFFFFFFFFFFFFLL )
      v16 = 0x3FFFFFFFFFFFFFFLL;
    v26 = 32 * v16;
  }
  v29 = sub_22077B0(v26);
  v27 = v29 + v26;
  v18 = v29 + 32;
LABEL_7:
  v19 = v29 + v17;
  if ( v29 + v17 )
  {
    *(_QWORD *)(v19 + 8) = 0;
    *(_QWORD *)v19 = v19 + 16;
    *(_QWORD *)(v19 + 16) = 0;
    *(_BYTE *)(v19 + 24) = 0;
  }
  if ( v10 != v8 )
  {
    v20 = v29;
    v21 = v8;
    while ( 1 )
    {
      if ( v20 )
      {
        *(_DWORD *)(v20 + 8) = 0;
        *(_QWORD *)v20 = v20 + 16;
        *(_DWORD *)(v20 + 12) = 0;
        v6 = *(unsigned int *)(v21 + 8);
        if ( (_DWORD)v6 )
        {
          v28 = v21;
          sub_25EFC00(v20, v21, v6, a4, a5, a6);
          v21 = v28;
        }
        *(_QWORD *)(v20 + 16) = *(_QWORD *)(v21 + 16);
        *(_BYTE *)(v20 + 24) = *(_BYTE *)(v21 + 24);
      }
      v21 += 32;
      if ( v10 == v21 )
        break;
      v20 += 32;
    }
    v18 = v20 + 64;
  }
  while ( (unsigned __int64 *)v10 != v7 )
  {
    *(_DWORD *)(v18 + 8) = 0;
    *(_QWORD *)v18 = v18 + 16;
    v23 = *(_DWORD *)(v10 + 8);
    *(_DWORD *)(v18 + 12) = 0;
    if ( v23 )
      sub_25EFC00(v18, v10, v6, a4, a5, a6);
    v22 = *(_QWORD *)(v10 + 16);
    v18 += 32;
    v10 += 32;
    *(_QWORD *)(v18 - 16) = v22;
    *(_BYTE *)(v18 - 8) = *(_BYTE *)(v10 - 8);
  }
  for ( i = (unsigned __int64 *)v8; v7 != i; i += 4 )
  {
    if ( (unsigned __int64 *)*i != i + 2 )
      _libc_free(*i);
  }
  if ( v8 )
    j_j___libc_free_0(v8);
  a1[1] = v18;
  *a1 = v29;
  a1[2] = v27;
  return v27;
}
