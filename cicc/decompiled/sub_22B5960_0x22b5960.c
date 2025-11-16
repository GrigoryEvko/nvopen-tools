// Function: sub_22B5960
// Address: 0x22b5960
//
unsigned __int64 *__fastcall sub_22B5960(
        unsigned __int64 *a1,
        __int64 *a2,
        unsigned int *a3,
        int *a4,
        __int64 *a5,
        __int64 *a6)
{
  unsigned __int64 v6; // r12
  unsigned __int64 v7; // r13
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rdi
  __int64 *v14; // r15
  bool v15; // cf
  unsigned __int64 v16; // rax
  char *v17; // r8
  __int64 v18; // rbx
  __int64 v19; // rax
  __int64 *v20; // rbx
  __int64 *v21; // rsi
  __int64 v22; // rdi
  unsigned __int64 i; // r14
  __int64 v24; // rsi
  __int64 v25; // rdi
  unsigned __int64 v27; // rbx
  __int64 v28; // rax
  __int64 *v29; // [rsp+0h] [rbp-70h]
  __int64 *v30; // [rsp+8h] [rbp-68h]
  int *v31; // [rsp+10h] [rbp-60h]
  unsigned __int64 v32; // [rsp+20h] [rbp-50h]
  __int64 v34; // [rsp+30h] [rbp-40h]
  __int64 v35; // [rsp+38h] [rbp-38h]

  v6 = a1[1];
  v7 = *a1;
  v8 = 0x86BCA1AF286BCA1BLL * ((__int64)(v6 - *a1) >> 3);
  if ( v8 == 0xD79435E50D7943LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v9 = 1;
  if ( v8 )
    v9 = 0x86BCA1AF286BCA1BLL * ((__int64)(v6 - v7) >> 3);
  v14 = a2;
  v15 = __CFADD__(v9, v8);
  v16 = v9 - 0x79435E50D79435E5LL * ((__int64)(v6 - v7) >> 3);
  v17 = (char *)a2 - v7;
  if ( v15 )
  {
    v27 = 0x7FFFFFFFFFFFFFC8LL;
  }
  else
  {
    if ( !v16 )
    {
      v32 = 0;
      v18 = 152;
      v34 = 0;
      goto LABEL_7;
    }
    if ( v16 > 0xD79435E50D7943LL )
      v16 = 0xD79435E50D7943LL;
    v27 = 152 * v16;
  }
  v29 = a6;
  v30 = a5;
  v31 = a4;
  v28 = sub_22077B0(v27);
  v17 = (char *)a2 - v7;
  a4 = v31;
  a5 = v30;
  v34 = v28;
  v32 = v28 + v27;
  a6 = v29;
  v18 = v28 + 152;
LABEL_7:
  if ( &v17[v34] )
    sub_22B4EF0((__int64)&v17[v34], *a3, *a4, *a5, *a6);
  if ( a2 != (__int64 *)v7 )
  {
    v19 = v34;
    v20 = (__int64 *)v7;
    while ( 1 )
    {
      if ( v19 )
      {
        v35 = v19;
        sub_22AF6E0(v19, v20);
        v19 = v35;
      }
      v20 += 19;
      if ( a2 == v20 )
        break;
      v19 += 152;
    }
    v18 = v19 + 304;
  }
  if ( a2 != (__int64 *)v6 )
  {
    do
    {
      v21 = v14;
      v22 = v18;
      v14 += 19;
      v18 += 152;
      sub_22AF6E0(v22, v21);
    }
    while ( (__int64 *)v6 != v14 );
  }
  for ( i = v7; i != v6; sub_C7D6A0(*(_QWORD *)(i - 120), 16LL * *(unsigned int *)(i - 104), 8) )
  {
    v24 = *(unsigned int *)(i + 144);
    v25 = *(_QWORD *)(i + 128);
    i += 152LL;
    sub_C7D6A0(v25, 8 * v24, 4);
    sub_C7D6A0(*(_QWORD *)(i - 56), 8LL * *(unsigned int *)(i - 40), 4);
    sub_C7D6A0(*(_QWORD *)(i - 88), 16LL * *(unsigned int *)(i - 72), 8);
  }
  if ( v7 )
    j_j___libc_free_0(v7);
  *a1 = v34;
  a1[1] = v18;
  a1[2] = v32;
  return a1;
}
