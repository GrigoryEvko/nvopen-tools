// Function: sub_35BC7B0
// Address: 0x35bc7b0
//
void __fastcall sub_35BC7B0(unsigned __int64 *a1, _DWORD *a2, __int64 a3)
{
  _DWORD *v5; // rsi
  unsigned __int64 v6; // r14
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rdx
  bool v9; // cf
  unsigned __int64 v10; // rax
  char *v11; // rdx
  unsigned __int64 v12; // r8
  unsigned __int64 v13; // r15
  unsigned __int64 v14; // r13
  char *v15; // rax
  unsigned __int64 v16; // rdx
  unsigned __int64 v17; // rax
  _DWORD *v18; // rax
  unsigned __int64 v19; // rdx
  int v20; // ecx
  unsigned __int64 v21; // r15
  __int64 v22; // rax
  __int64 v23; // [rsp+8h] [rbp-48h]
  unsigned __int64 v24; // [rsp+18h] [rbp-38h]

  v5 = (_DWORD *)a1[1];
  v6 = *a1;
  v7 = 0xAAAAAAAAAAAAAAABLL * ((__int64)((__int64)v5 - *a1) >> 3);
  if ( v7 == 0x555555555555555LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v8 = 1;
  if ( v7 )
    v8 = 0xAAAAAAAAAAAAAAABLL * ((__int64)((__int64)v5 - v6) >> 3);
  v9 = __CFADD__(v8, v7);
  v10 = v8 - 0x5555555555555555LL * ((__int64)((__int64)v5 - v6) >> 3);
  v11 = (char *)a2 - v6;
  if ( v9 )
  {
    v21 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v10 )
    {
      v12 = 24;
      v13 = 0;
      v14 = 0;
      goto LABEL_7;
    }
    if ( v10 > 0x555555555555555LL )
      v10 = 0x555555555555555LL;
    v21 = 24 * v10;
  }
  v23 = a3;
  v22 = sub_22077B0(v21);
  v11 = (char *)a2 - v6;
  a3 = v23;
  v14 = v22;
  v13 = v22 + v21;
  v12 = v22 + 24;
LABEL_7:
  v15 = &v11[v14];
  if ( &v11[v14] )
  {
    *(_DWORD *)v15 = *(_DWORD *)a3;
    *((_QWORD *)v15 + 1) = *(_QWORD *)(a3 + 8);
    *((_QWORD *)v15 + 2) = *(_QWORD *)(a3 + 16);
  }
  if ( a2 != (_DWORD *)v6 )
  {
    v16 = v14;
    v17 = v6;
    do
    {
      if ( v16 )
      {
        *(_DWORD *)v16 = *(_DWORD *)v17;
        *(_QWORD *)(v16 + 8) = *(_QWORD *)(v17 + 8);
        *(_QWORD *)(v16 + 16) = *(_QWORD *)(v17 + 16);
      }
      v17 += 24LL;
      v16 += 24LL;
    }
    while ( (_DWORD *)v17 != a2 );
    v12 = v14 + 8 * (((unsigned __int64)a2 - v6 - 24) >> 3) + 48;
  }
  if ( a2 != v5 )
  {
    v18 = a2;
    v19 = v12;
    do
    {
      v20 = *v18;
      v18 += 6;
      v19 += 24LL;
      *(_DWORD *)(v19 - 24) = v20;
      *(_QWORD *)(v19 - 16) = *((_QWORD *)v18 - 2);
      *(_QWORD *)(v19 - 8) = *((_QWORD *)v18 - 1);
    }
    while ( v18 != v5 );
    v12 += 8 * ((unsigned __int64)((char *)v18 - (char *)a2 - 24) >> 3) + 24;
  }
  if ( v6 )
  {
    v24 = v12;
    j_j___libc_free_0(v6);
    v12 = v24;
  }
  *a1 = v14;
  a1[2] = v13;
  a1[1] = v12;
}
