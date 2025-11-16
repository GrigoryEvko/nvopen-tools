// Function: sub_1383AB0
// Address: 0x1383ab0
//
void __fastcall sub_1383AB0(_QWORD *a1, unsigned __int64 a2)
{
  _QWORD *v3; // r13
  _QWORD *v4; // r12
  __int64 v5; // r15
  _QWORD *v6; // rax
  unsigned __int64 v7; // rdx
  __int64 v8; // rax
  bool v9; // cf
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // rdx
  _QWORD *v12; // rax
  _QWORD *v13; // r15
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rdi
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // [rsp-50h] [rbp-50h]
  __int64 v20; // [rsp-50h] [rbp-50h]
  _QWORD *v21; // [rsp-48h] [rbp-48h]
  unsigned __int64 v22; // [rsp-40h] [rbp-40h]

  if ( !a2 )
    return;
  v3 = (_QWORD *)a1[1];
  v4 = (_QWORD *)*a1;
  v5 = (__int64)v3 - *a1;
  v22 = 0x6DB6DB6DB6DB6DB7LL * (v5 >> 3);
  if ( 0x6DB6DB6DB6DB6DB7LL * ((__int64)(a1[2] - (_QWORD)v3) >> 3) >= a2 )
  {
    v6 = (_QWORD *)a1[1];
    v7 = a2;
    do
    {
      if ( v6 )
      {
        v6[6] = 0;
        *v6 = 0;
        v6[1] = 0;
        v6[2] = 0;
        v6[3] = 0;
        v6[4] = 0;
        v6[5] = 0;
      }
      v6 += 7;
      --v7;
    }
    while ( v7 );
    a1[1] = &v3[7 * a2];
    return;
  }
  if ( 0x249249249249249LL - v22 < a2 )
    sub_4262D8((__int64)"vector::_M_default_append");
  v8 = 0x6DB6DB6DB6DB6DB7LL * (((__int64)v3 - *a1) >> 3);
  if ( v22 < a2 )
    v8 = a2;
  v9 = __CFADD__(v22, v8);
  v10 = v22 + v8;
  if ( v9 )
  {
    v17 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v10 )
    {
      v19 = 0;
      v21 = 0;
      goto LABEL_15;
    }
    if ( v10 > 0x249249249249249LL )
      v10 = 0x249249249249249LL;
    v17 = 56 * v10;
  }
  v20 = v17;
  v18 = sub_22077B0(v17);
  v3 = (_QWORD *)a1[1];
  v21 = (_QWORD *)v18;
  v4 = (_QWORD *)*a1;
  v19 = v18 + v20;
LABEL_15:
  v11 = a2;
  v12 = (_QWORD *)((char *)v21 + v5);
  do
  {
    if ( v12 )
    {
      v12[6] = 0;
      *v12 = 0;
      v12[1] = 0;
      v12[2] = 0;
      v12[3] = 0;
      v12[4] = 0;
      v12[5] = 0;
    }
    v12 += 7;
    --v11;
  }
  while ( v11 );
  if ( v3 != v4 )
  {
    v13 = v21;
    do
    {
      if ( v13 )
      {
        *v13 = *v4;
        v13[1] = v4[1];
        v13[2] = v4[2];
        v14 = v4[3];
        v4[2] = 0;
        v4[1] = 0;
        *v4 = 0;
        v13[3] = v14;
        v13[4] = v4[4];
        v13[5] = v4[5];
        v15 = v4[6];
        v4[5] = 0;
        v4[4] = 0;
        v4[3] = 0;
        v13[6] = v15;
      }
      v16 = v4[3];
      if ( v16 )
        j_j___libc_free_0(v16, v4[5] - v16);
      if ( *v4 )
        j_j___libc_free_0(*v4, v4[2] - *v4);
      v4 += 7;
      v13 += 7;
    }
    while ( v4 != v3 );
    v4 = (_QWORD *)*a1;
  }
  if ( v4 )
    j_j___libc_free_0(v4, a1[2] - (_QWORD)v4);
  *a1 = v21;
  a1[1] = &v21[7 * a2 + 7 * v22];
  a1[2] = v19;
}
