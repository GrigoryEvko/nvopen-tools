// Function: sub_25A3900
// Address: 0x25a3900
//
void __fastcall sub_25A3900(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v2; // rcx
  unsigned __int64 v4; // r13
  unsigned __int64 *v5; // rbx
  __int64 v6; // r15
  unsigned __int64 v7; // r12
  unsigned __int64 v8; // rdx
  _QWORD *v9; // rax
  unsigned __int64 v10; // rax
  bool v11; // cf
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // rdx
  _QWORD *v14; // rax
  __int64 v15; // r15
  unsigned __int64 v16; // rax
  unsigned __int64 v17; // rsi
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // rdx
  __int64 v20; // rax
  unsigned __int64 v21; // [rsp-50h] [rbp-50h]
  __int64 v22; // [rsp-48h] [rbp-48h]
  unsigned __int64 v23; // [rsp-40h] [rbp-40h]
  unsigned __int64 v24; // [rsp-40h] [rbp-40h]
  unsigned __int64 v25; // [rsp-40h] [rbp-40h]

  if ( !a2 )
    return;
  v2 = a2;
  v4 = *(_QWORD *)(a1 + 8);
  v5 = *(unsigned __int64 **)a1;
  v6 = v4 - *(_QWORD *)a1;
  v7 = 0xAAAAAAAAAAAAAAABLL * (v6 >> 3);
  if ( a2 <= 0xAAAAAAAAAAAAAAABLL * ((__int64)(*(_QWORD *)(a1 + 16) - v4) >> 3) )
  {
    v8 = a2;
    v9 = *(_QWORD **)(a1 + 8);
    do
    {
      if ( v9 )
      {
        *v9 = 0;
        v9[1] = 0;
        v9[2] = 0;
      }
      v9 += 3;
      --v8;
    }
    while ( v8 );
    *(_QWORD *)(a1 + 8) = v4 + 24 * a2;
    return;
  }
  if ( 0x555555555555555LL - v7 < a2 )
    sub_4262D8((__int64)"vector::_M_default_append");
  v10 = 0xAAAAAAAAAAAAAAABLL * ((__int64)(v4 - *(_QWORD *)a1) >> 3);
  if ( a2 >= v7 )
    v10 = a2;
  v11 = __CFADD__(v7, v10);
  v12 = v7 + v10;
  if ( v11 )
  {
    v19 = 0x7FFFFFFFFFFFFFF8LL;
LABEL_33:
    v25 = v19;
    v20 = sub_22077B0(v19);
    v4 = *(_QWORD *)(a1 + 8);
    v5 = *(unsigned __int64 **)a1;
    v2 = a2;
    v22 = v20;
    v21 = v20 + v25;
    goto LABEL_15;
  }
  if ( v12 )
  {
    if ( v12 > 0x555555555555555LL )
      v12 = 0x555555555555555LL;
    v19 = 24 * v12;
    goto LABEL_33;
  }
  v21 = 0;
  v22 = 0;
LABEL_15:
  v13 = v2;
  v14 = (_QWORD *)(v6 + v22);
  do
  {
    if ( v14 )
    {
      *v14 = 0;
      v14[1] = 0;
      v14[2] = 0;
    }
    v14 += 3;
    --v13;
  }
  while ( v13 );
  if ( v5 != (unsigned __int64 *)v4 )
  {
    v15 = v22;
    while ( 1 )
    {
      while ( 1 )
      {
        v17 = v5[2];
        v18 = *v5;
        if ( !v15 )
          break;
        *(_QWORD *)v15 = v18;
        v16 = v5[1];
        *(_QWORD *)(v15 + 16) = v17;
        *(_QWORD *)(v15 + 8) = v16;
        v5[2] = 0;
        *v5 = 0;
LABEL_22:
        v5 += 3;
        v15 += 24;
        if ( v5 == (unsigned __int64 *)v4 )
          goto LABEL_26;
      }
      if ( !v18 )
        goto LABEL_22;
      v5 += 3;
      v23 = v2;
      v15 = 24;
      j_j___libc_free_0(v18);
      v2 = v23;
      if ( v5 == (unsigned __int64 *)v4 )
      {
LABEL_26:
        v4 = *(_QWORD *)a1;
        break;
      }
    }
  }
  if ( v4 )
  {
    v24 = v2;
    j_j___libc_free_0(v4);
    v2 = v24;
  }
  *(_QWORD *)a1 = v22;
  *(_QWORD *)(a1 + 8) = v22 + 24 * (v7 + v2);
  *(_QWORD *)(a1 + 16) = v21;
}
