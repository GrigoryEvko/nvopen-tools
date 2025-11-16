// Function: sub_22B2160
// Address: 0x22b2160
//
__int64 __fastcall sub_22B2160(unsigned __int64 a1, unsigned __int64 *a2, __int64 **a3)
{
  __int64 v4; // rsi
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // rcx
  bool v7; // cf
  unsigned __int64 v8; // rax
  char *v9; // rbx
  __int64 *v10; // r15
  __int64 *v11; // rbx
  __int64 *v12; // r12
  unsigned __int64 v13; // rcx
  __int64 v14; // rax
  __int64 v15; // r13
  unsigned __int64 *v16; // r12
  __int64 v17; // r13
  unsigned __int64 v18; // rbx
  unsigned __int64 v19; // r15
  __int64 v20; // rsi
  __int64 v21; // rdi
  __int64 v22; // r13
  unsigned __int64 *v23; // rax
  __int64 v24; // rdx
  unsigned __int64 v25; // rcx
  __int64 result; // rax
  __int64 v27; // rax
  __int64 **v28; // [rsp+8h] [rbp-68h]
  __int64 **v29; // [rsp+10h] [rbp-60h]
  __int64 v30; // [rsp+18h] [rbp-58h]
  _QWORD *v31; // [rsp+20h] [rbp-50h]
  unsigned __int64 v32; // [rsp+28h] [rbp-48h]
  __int64 v33; // [rsp+30h] [rbp-40h]
  unsigned __int64 *v34; // [rsp+38h] [rbp-38h]

  v4 = 0x555555555555555LL;
  v31 = (_QWORD *)a1;
  v34 = *(unsigned __int64 **)(a1 + 8);
  v32 = *(_QWORD *)a1;
  v5 = 0xAAAAAAAAAAAAAAABLL * (((__int64)v34 - *(_QWORD *)a1) >> 3);
  if ( v5 == 0x555555555555555LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v6 = 1;
  if ( v5 )
    v6 = 0xAAAAAAAAAAAAAAABLL * (((__int64)v34 - *(_QWORD *)a1) >> 3);
  v7 = __CFADD__(v6, v5);
  v8 = v6 - 0x5555555555555555LL * (((__int64)v34 - *(_QWORD *)a1) >> 3);
  v30 = v8;
  v9 = (char *)a2 - v32;
  if ( v7 )
  {
    a1 = 0x7FFFFFFFFFFFFFF8LL;
    v30 = 0x555555555555555LL;
  }
  else
  {
    if ( !v8 )
    {
      v33 = 0;
      goto LABEL_7;
    }
    if ( v8 <= 0x555555555555555LL )
      v4 = v6 - 0x5555555555555555LL * (((__int64)v34 - *(_QWORD *)a1) >> 3);
    v30 = v4;
    a1 = 24 * v4;
  }
  v29 = a3;
  v27 = sub_22077B0(a1);
  a3 = v29;
  v33 = v27;
LABEL_7:
  v10 = (__int64 *)&v9[v33];
  if ( &v9[v33] )
  {
    v11 = a3[1];
    v12 = *a3;
    *v10 = 0;
    v10[1] = 0;
    v10[2] = 0;
    v13 = (char *)v11 - (char *)v12;
    if ( v11 == v12 )
    {
      v15 = 0;
    }
    else
    {
      if ( v13 > 0x7FFFFFFFFFFFFFC8LL )
        sub_4261EA(a1, v4, a3);
      v28 = a3;
      v14 = sub_22077B0((char *)v11 - (char *)v12);
      v13 = (char *)v11 - (char *)v12;
      v15 = v14;
      v11 = v28[1];
      v12 = *v28;
    }
    *v10 = v15;
    v10[1] = v15;
    for ( v10[2] = v15 + v13; v11 != v12; v15 += 152 )
    {
      if ( v15 )
        sub_22AF6E0(v15, v12);
      v12 += 19;
    }
    v10[1] = v15;
  }
  v16 = (unsigned __int64 *)v32;
  v17 = v33;
  while ( v16 != a2 )
  {
    while ( 1 )
    {
      v18 = v16[1];
      v19 = *v16;
      if ( !v17 )
        break;
      *(_QWORD *)v17 = v19;
      *(_QWORD *)(v17 + 8) = v18;
      *(_QWORD *)(v17 + 16) = v16[2];
      v16[2] = 0;
      v16[1] = 0;
      *v16 = 0;
LABEL_19:
      v16 += 3;
      v17 += 24;
      if ( v16 == a2 )
        goto LABEL_26;
    }
    if ( v19 != v18 )
    {
      do
      {
        v20 = *(unsigned int *)(v19 + 144);
        v21 = *(_QWORD *)(v19 + 128);
        v19 += 152LL;
        sub_C7D6A0(v21, 8 * v20, 4);
        sub_C7D6A0(*(_QWORD *)(v19 - 56), 8LL * *(unsigned int *)(v19 - 40), 4);
        sub_C7D6A0(*(_QWORD *)(v19 - 88), 16LL * *(unsigned int *)(v19 - 72), 8);
        sub_C7D6A0(*(_QWORD *)(v19 - 120), 16LL * *(unsigned int *)(v19 - 104), 8);
      }
      while ( v19 != v18 );
      v19 = *v16;
    }
    if ( !v19 )
      goto LABEL_19;
    v16 += 3;
    v17 = 24;
    j_j___libc_free_0(v19);
  }
LABEL_26:
  v22 = v17 + 24;
  if ( a2 != v34 )
  {
    v23 = a2;
    v24 = v22;
    do
    {
      v25 = *v23;
      v24 += 24;
      v23 += 3;
      *(_QWORD *)(v24 - 24) = v25;
      *(_QWORD *)(v24 - 16) = *(v23 - 2);
      *(_QWORD *)(v24 - 8) = *(v23 - 1);
    }
    while ( v23 != v34 );
    v22 += 8 * ((unsigned __int64)((char *)v23 - (char *)a2 - 24) >> 3) + 24;
  }
  if ( v32 )
    j_j___libc_free_0(v32);
  result = v33 + 24 * v30;
  *v31 = v33;
  v31[1] = v22;
  v31[2] = result;
  return result;
}
