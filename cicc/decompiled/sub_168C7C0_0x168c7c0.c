// Function: sub_168C7C0
// Address: 0x168c7c0
//
__int64 *__fastcall sub_168C7C0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v5; // r12
  __int64 v6; // r13
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rdx
  bool v9; // cf
  unsigned __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rbx
  __int64 v13; // r14
  __int64 v14; // rbx
  __int64 v15; // rdi
  __int64 i; // r15
  __int64 v17; // rdi
  __int64 v19; // rbx
  __int64 v20; // rax
  __int64 v21; // [rsp+18h] [rbp-48h]
  __int64 v23; // [rsp+28h] [rbp-38h]

  v5 = a1[1];
  v6 = *a1;
  v7 = 0xAAAAAAAAAAAAAAABLL * ((v5 - *a1) >> 3);
  if ( v7 == 0x555555555555555LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v8 = 1;
  if ( v7 )
    v8 = 0xAAAAAAAAAAAAAAABLL * ((v5 - v6) >> 3);
  v9 = __CFADD__(v8, v7);
  v10 = v8 - 0x5555555555555555LL * ((v5 - v6) >> 3);
  v11 = a2 - v6;
  if ( v9 )
  {
    v19 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v10 )
    {
      v21 = 0;
      v12 = 24;
      v23 = 0;
      goto LABEL_7;
    }
    if ( v10 > 0x555555555555555LL )
      v10 = 0x555555555555555LL;
    v19 = 24 * v10;
  }
  v20 = sub_22077B0(v19);
  v11 = a2 - v6;
  v23 = v20;
  v21 = v20 + v19;
  v12 = v20 + 24;
LABEL_7:
  if ( v23 + v11 )
    sub_16CE2D0(v23 + v11, a3);
  if ( a2 != v6 )
  {
    v13 = v23;
    v14 = v6;
    while ( 1 )
    {
      if ( v13 )
      {
        a3 = v14;
        sub_16CE2D0(v13, v14);
      }
      v14 += 24;
      if ( a2 == v14 )
        break;
      v13 += 24;
    }
    v12 = v13 + 48;
  }
  while ( v5 != a2 )
  {
    a3 = a2;
    v15 = v12;
    a2 += 24;
    v12 += 24;
    sub_16CE2D0(v15, a3);
  }
  for ( i = v6; i != v5; i += 24 )
  {
    v17 = i;
    sub_16CE300(v17, a3);
  }
  if ( v6 )
    j_j___libc_free_0(v6, a1[2] - v6);
  *a1 = v23;
  a1[1] = v12;
  a1[2] = v21;
  return a1;
}
