// Function: sub_398F450
// Address: 0x398f450
//
unsigned __int64 __fastcall sub_398F450(unsigned __int64 *a1, __int64 *a2, __int64 a3, __int64 a4)
{
  unsigned __int64 v5; // r15
  unsigned __int64 v6; // r12
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rdi
  __int64 *v9; // r8
  __int64 *v10; // r14
  bool v11; // cf
  unsigned __int64 v12; // rax
  __int64 v13; // r9
  __int64 v14; // rbx
  _QWORD *v15; // rdi
  unsigned __int64 v16; // rbx
  unsigned __int64 v17; // rax
  __int64 v18; // rsi
  __int64 v19; // rax
  unsigned __int64 i; // r14
  unsigned __int64 v21; // rdi
  unsigned __int64 v23; // rbx
  __int64 v24; // rax
  __int64 v25; // [rsp+8h] [rbp-58h]
  unsigned __int64 v26; // [rsp+10h] [rbp-50h]
  __int64 *v27; // [rsp+18h] [rbp-48h]
  unsigned __int64 v28; // [rsp+20h] [rbp-40h]
  __int64 *v29; // [rsp+20h] [rbp-40h]
  unsigned __int64 v30; // [rsp+28h] [rbp-38h]

  v5 = a1[1];
  v6 = *a1;
  v7 = 0x86BCA1AF286BCA1BLL * ((__int64)(v5 - *a1) >> 3);
  if ( v7 == 0xD79435E50D7943LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v8 = 1;
  v9 = a2;
  if ( v7 )
    v8 = 0x86BCA1AF286BCA1BLL * ((__int64)(v5 - v6) >> 3);
  v10 = a2;
  v11 = __CFADD__(v8, v7);
  v12 = v8 - 0x79435E50D79435E5LL * ((__int64)(v5 - v6) >> 3);
  v13 = (__int64)a2 - v6;
  if ( v11 )
  {
    v23 = 0x7FFFFFFFFFFFFFC8LL;
LABEL_31:
    v25 = a3;
    v24 = sub_22077B0(v23);
    v13 = (__int64)a2 - v6;
    v9 = a2;
    v30 = v24;
    a3 = v25;
    v26 = v24 + v23;
    v14 = v24 + 152;
    goto LABEL_7;
  }
  if ( v12 )
  {
    if ( v12 > 0xD79435E50D7943LL )
      v12 = 0xD79435E50D7943LL;
    v23 = 152 * v12;
    goto LABEL_31;
  }
  v26 = 0;
  v14 = 152;
  v30 = 0;
LABEL_7:
  v15 = (_QWORD *)(v30 + v13);
  if ( v30 + v13 )
  {
    a4 = *(unsigned int *)(a3 + 16);
    *v15 = *(_QWORD *)a3;
    v15[1] = v15 + 3;
    v15[2] = 0x800000000LL;
    if ( (_DWORD)a4 )
    {
      v29 = v9;
      sub_39847E0((__int64)(v15 + 1), (char **)(a3 + 8), a3, a4, (int)v9, v13);
      v9 = v29;
    }
  }
  if ( v9 != (__int64 *)v6 )
  {
    v16 = v30;
    v17 = v6;
    while ( 1 )
    {
      if ( v16 )
      {
        v18 = *(_QWORD *)v17;
        *(_DWORD *)(v16 + 16) = 0;
        *(_DWORD *)(v16 + 20) = 8;
        *(_QWORD *)v16 = v18;
        *(_QWORD *)(v16 + 8) = v16 + 24;
        a3 = *(unsigned int *)(v17 + 16);
        if ( (_DWORD)a3 )
        {
          v27 = v9;
          v28 = v17;
          sub_39844D0(v16 + 8, v17 + 8, a3, a4, (int)v9, v13);
          v9 = v27;
          v17 = v28;
        }
      }
      v17 += 152LL;
      if ( v9 == (__int64 *)v17 )
        break;
      v16 += 152LL;
    }
    v14 = v16 + 304;
  }
  if ( v9 != (__int64 *)v5 )
  {
    do
    {
      v19 = *v10;
      *(_DWORD *)(v14 + 16) = 0;
      *(_DWORD *)(v14 + 20) = 8;
      *(_QWORD *)v14 = v19;
      *(_QWORD *)(v14 + 8) = v14 + 24;
      if ( *((_DWORD *)v10 + 4) )
        sub_39844D0(v14 + 8, (__int64)(v10 + 1), a3, a4, (int)v9, v13);
      v10 += 19;
      v14 += 152;
    }
    while ( (__int64 *)v5 != v10 );
  }
  for ( i = v6; v5 != i; i += 152LL )
  {
    v21 = *(_QWORD *)(i + 8);
    if ( v21 != i + 24 )
      _libc_free(v21);
  }
  if ( v6 )
    j_j___libc_free_0(v6);
  a1[1] = v14;
  *a1 = v30;
  a1[2] = v26;
  return v26;
}
