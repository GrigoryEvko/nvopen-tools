// Function: sub_31FCBC0
// Address: 0x31fcbc0
//
void __fastcall sub_31FCBC0(unsigned __int64 *a1, char *a2, _QWORD *a3)
{
  char *v5; // rsi
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
  _QWORD *v16; // rdx
  char *v17; // rax
  char *v18; // rax
  unsigned __int64 v19; // rdx
  __int64 v20; // rcx
  unsigned __int64 v21; // r15
  __int64 v22; // rax
  _QWORD *v23; // [rsp+8h] [rbp-48h]
  unsigned __int64 v24; // [rsp+18h] [rbp-38h]

  v5 = (char *)a1[1];
  v6 = *a1;
  v7 = 0xAAAAAAAAAAAAAAABLL * ((__int64)&v5[-*a1] >> 3);
  if ( v7 == 0x555555555555555LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v8 = 1;
  if ( v7 )
    v8 = 0xAAAAAAAAAAAAAAABLL * ((__int64)&v5[-v6] >> 3);
  v9 = __CFADD__(v8, v7);
  v10 = v8 - 0x5555555555555555LL * ((__int64)&v5[-v6] >> 3);
  v11 = &a2[-v6];
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
  v11 = &a2[-v6];
  a3 = v23;
  v14 = v22;
  v13 = v22 + v21;
  v12 = v22 + 24;
LABEL_7:
  v15 = &v11[v14];
  if ( &v11[v14] )
  {
    *(_QWORD *)v15 = *a3;
    *((_QWORD *)v15 + 1) = a3[1];
    *((_QWORD *)v15 + 2) = a3[2];
  }
  if ( a2 != (char *)v6 )
  {
    v16 = (_QWORD *)v14;
    v17 = (char *)v6;
    do
    {
      if ( v16 )
      {
        *v16 = *(_QWORD *)v17;
        v16[1] = *((_QWORD *)v17 + 1);
        v16[2] = *((_QWORD *)v17 + 2);
      }
      v17 += 24;
      v16 += 3;
    }
    while ( v17 != a2 );
    v12 = v14 + 8 * ((unsigned __int64)&a2[-v6 - 24] >> 3) + 48;
  }
  if ( a2 != v5 )
  {
    v18 = a2;
    v19 = v12;
    do
    {
      v20 = *(_QWORD *)v18;
      v18 += 24;
      v19 += 24LL;
      *(_QWORD *)(v19 - 24) = v20;
      *(_QWORD *)(v19 - 16) = *((_QWORD *)v18 - 2);
      *(_QWORD *)(v19 - 8) = *((_QWORD *)v18 - 1);
    }
    while ( v18 != v5 );
    v12 += 8 * ((unsigned __int64)(v18 - a2 - 24) >> 3) + 24;
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
