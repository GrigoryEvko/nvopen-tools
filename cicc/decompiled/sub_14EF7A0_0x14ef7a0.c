// Function: sub_14EF7A0
// Address: 0x14ef7a0
//
void __fastcall sub_14EF7A0(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v2; // r13
  _QWORD *v3; // r12
  _QWORD *v4; // rdx
  __int64 v5; // r15
  unsigned __int64 v6; // rbx
  unsigned __int64 v7; // rdx
  _QWORD *v8; // rax
  unsigned __int64 v9; // rax
  bool v10; // cf
  unsigned __int64 v11; // rax
  _QWORD *v12; // rax
  _QWORD *v13; // r15
  __int64 v14; // rax
  _QWORD *v15; // r15
  __int64 v16; // rax
  __int64 v17; // rcx
  __int64 v18; // rax
  __int64 v19; // [rsp-50h] [rbp-50h]
  _QWORD *v20; // [rsp-48h] [rbp-48h]
  __int64 v21; // [rsp-48h] [rbp-48h]
  _QWORD *v22; // [rsp-40h] [rbp-40h]

  if ( !a2 )
    return;
  v2 = a2;
  v3 = *(_QWORD **)(a1 + 8);
  v4 = *(_QWORD **)a1;
  v5 = (__int64)v3 - *(_QWORD *)a1;
  v6 = 0xAAAAAAAAAAAAAAABLL * (v5 >> 3);
  if ( 0xAAAAAAAAAAAAAAABLL * ((__int64)(*(_QWORD *)(a1 + 16) - (_QWORD)v3) >> 3) >= a2 )
  {
    v7 = a2;
    v8 = *(_QWORD **)(a1 + 8);
    do
    {
      if ( v8 )
      {
        *v8 = 6;
        v8[1] = 0;
        v8[2] = 0;
      }
      v8 += 3;
      --v7;
    }
    while ( v7 );
    *(_QWORD *)(a1 + 8) = &v3[3 * a2];
    return;
  }
  if ( 0x555555555555555LL - v6 < a2 )
    sub_4262D8((__int64)"vector::_M_default_append");
  v9 = 0xAAAAAAAAAAAAAAABLL * (v3 - v4);
  if ( a2 >= v6 )
    v9 = a2;
  v10 = __CFADD__(v6, v9);
  v11 = v6 + v9;
  if ( v10 )
  {
    v17 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v11 )
    {
      v19 = 0;
      v22 = 0;
      goto LABEL_15;
    }
    if ( v11 > 0x555555555555555LL )
      v11 = 0x555555555555555LL;
    v17 = 24 * v11;
  }
  v21 = v17;
  v18 = sub_22077B0(v17);
  v3 = *(_QWORD **)(a1 + 8);
  v22 = (_QWORD *)v18;
  v4 = *(_QWORD **)a1;
  v19 = v18 + v21;
LABEL_15:
  v12 = (_QWORD *)((char *)v22 + v5);
  do
  {
    if ( v12 )
    {
      *v12 = 6;
      v12[1] = 0;
      v12[2] = 0;
    }
    v12 += 3;
    --a2;
  }
  while ( a2 );
  if ( v4 != v3 )
  {
    v13 = v22;
    do
    {
      if ( v13 )
      {
        *v13 = 6;
        v13[1] = 0;
        v14 = v4[2];
        v13[2] = v14;
        if ( v14 != 0 && v14 != -8 && v14 != -16 )
        {
          v20 = v4;
          sub_1649AC0(v13, *v4 & 0xFFFFFFFFFFFFFFF8LL);
          v4 = v20;
        }
      }
      v4 += 3;
      v13 += 3;
    }
    while ( v4 != v3 );
    v15 = *(_QWORD **)(a1 + 8);
    v3 = *(_QWORD **)a1;
    if ( v15 != *(_QWORD **)a1 )
    {
      do
      {
        v16 = v3[2];
        if ( v16 != -8 && v16 != 0 && v16 != -16 )
          sub_1649B30(v3);
        v3 += 3;
      }
      while ( v15 != v3 );
      v3 = *(_QWORD **)a1;
    }
  }
  if ( v3 )
    j_j___libc_free_0(v3, *(_QWORD *)(a1 + 16) - (_QWORD)v3);
  *(_QWORD *)a1 = v22;
  *(_QWORD *)(a1 + 8) = &v22[3 * v2 + 3 * v6];
  *(_QWORD *)(a1 + 16) = v19;
}
