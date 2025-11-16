// Function: sub_2DD28A0
// Address: 0x2dd28a0
//
unsigned __int64 *__fastcall sub_2DD28A0(unsigned __int64 *a1, _QWORD *a2, _QWORD *a3, __int64 *a4)
{
  _QWORD *v4; // rbx
  unsigned __int64 v5; // r13
  __int64 v6; // rax
  __int64 v9; // rcx
  _QWORD *v10; // r15
  bool v11; // cf
  unsigned __int64 v12; // rax
  char *v13; // r10
  __int64 v14; // r12
  __int64 v15; // rsi
  __int64 v16; // r11
  char *v17; // r10
  unsigned __int8 *v18; // rsi
  _QWORD *v19; // r12
  _QWORD *v20; // r10
  __int64 v21; // rsi
  __int64 v22; // rsi
  unsigned __int64 i; // r14
  __int64 v24; // rsi
  unsigned __int64 v26; // r12
  __int64 v27; // rax
  __int64 v28; // [rsp+0h] [rbp-70h]
  char *v29; // [rsp+8h] [rbp-68h]
  _QWORD *v30; // [rsp+8h] [rbp-68h]
  unsigned __int64 v31; // [rsp+10h] [rbp-60h]
  unsigned __int64 v32; // [rsp+20h] [rbp-50h]
  _QWORD *v33; // [rsp+28h] [rbp-48h]
  __int64 v34[7]; // [rsp+38h] [rbp-38h] BYREF

  v4 = (_QWORD *)a1[1];
  v5 = *a1;
  v6 = (__int64)((__int64)v4 - *a1) >> 4;
  if ( v6 == 0x7FFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v9 = 1;
  if ( v6 )
    v9 = (__int64)((__int64)v4 - v5) >> 4;
  v10 = a2;
  v11 = __CFADD__(v9, v6);
  v12 = v9 + v6;
  v13 = (char *)a2 - v5;
  if ( v11 )
  {
    v26 = 0x7FFFFFFFFFFFFFF0LL;
  }
  else
  {
    if ( !v12 )
    {
      v31 = 0;
      v14 = 16;
      v32 = 0;
      goto LABEL_7;
    }
    if ( v12 > 0x7FFFFFFFFFFFFFFLL )
      v12 = 0x7FFFFFFFFFFFFFFLL;
    v26 = 16 * v12;
  }
  v30 = a3;
  v27 = sub_22077B0(v26);
  v13 = (char *)a2 - v5;
  a3 = v30;
  v32 = v27;
  v31 = v27 + v26;
  v14 = v27 + 16;
LABEL_7:
  v15 = *a4;
  v16 = *a3;
  v17 = &v13[v32];
  v34[0] = v15;
  if ( v15 )
  {
    v28 = v16;
    v29 = v17;
    sub_B96E90((__int64)v34, v15, 1);
    v18 = (unsigned __int8 *)v34[0];
    if ( v29 )
    {
      *(_QWORD *)v29 = v28;
      *((_QWORD *)v29 + 1) = v18;
      if ( v18 )
        sub_B976B0((__int64)v34, v18, (__int64)(v29 + 8));
    }
    else if ( v34[0] )
    {
      sub_B91220((__int64)v34, v34[0]);
    }
  }
  else if ( v17 )
  {
    *(_QWORD *)v17 = v16;
    *((_QWORD *)v17 + 1) = 0;
  }
  if ( a2 != (_QWORD *)v5 )
  {
    v19 = (_QWORD *)v32;
    v20 = (_QWORD *)v5;
    while ( 1 )
    {
      if ( v19 )
      {
        *v19 = *v20;
        v21 = v20[1];
        v19[1] = v21;
        if ( v21 )
        {
          v33 = v20;
          sub_B96E90((__int64)(v19 + 1), v21, 1);
          v20 = v33;
        }
      }
      v20 += 2;
      if ( v20 == a2 )
        break;
      v19 += 2;
    }
    v14 = (__int64)(v19 + 4);
  }
  if ( a2 != v4 )
  {
    do
    {
      v22 = v10[1];
      *(_QWORD *)v14 = *v10;
      *(_QWORD *)(v14 + 8) = v22;
      if ( v22 )
        sub_B96E90(v14 + 8, v22, 1);
      v10 += 2;
      v14 += 16;
    }
    while ( v10 != v4 );
  }
  for ( i = v5; (_QWORD *)i != v4; i += 16LL )
  {
    v24 = *(_QWORD *)(i + 8);
    if ( v24 )
      sub_B91220(i + 8, v24);
  }
  if ( v5 )
    j_j___libc_free_0(v5);
  *a1 = v32;
  a1[1] = v14;
  a1[2] = v31;
  return a1;
}
