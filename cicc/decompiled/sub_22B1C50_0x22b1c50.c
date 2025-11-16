// Function: sub_22B1C50
// Address: 0x22b1c50
//
unsigned __int64 *__fastcall sub_22B1C50(unsigned __int64 *a1, __int64 *a2, __int64 *a3)
{
  unsigned __int64 v3; // r12
  unsigned __int64 v4; // r13
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // rcx
  __int64 *v8; // rsi
  __int64 *v9; // r15
  bool v10; // cf
  unsigned __int64 v11; // rax
  char *v12; // rcx
  __int64 v13; // rbx
  __int64 v14; // rax
  __int64 *v15; // rbx
  __int64 *v16; // rsi
  __int64 v17; // rdi
  unsigned __int64 i; // r14
  __int64 v19; // rsi
  __int64 v20; // rdi
  unsigned __int64 v22; // rbx
  __int64 v23; // rax
  unsigned __int64 v25; // [rsp+10h] [rbp-50h]
  __int64 v27; // [rsp+20h] [rbp-40h]
  __int64 v28; // [rsp+28h] [rbp-38h]

  v3 = a1[1];
  v4 = *a1;
  v5 = 0x86BCA1AF286BCA1BLL * ((__int64)(v3 - *a1) >> 3);
  if ( v5 == 0xD79435E50D7943LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v6 = 1;
  v8 = a3;
  if ( v5 )
    v6 = 0x86BCA1AF286BCA1BLL * ((__int64)(v3 - v4) >> 3);
  v9 = a2;
  v10 = __CFADD__(v6, v5);
  v11 = v6 - 0x79435E50D79435E5LL * ((__int64)(v3 - v4) >> 3);
  v12 = (char *)a2 - v4;
  if ( v10 )
  {
    v22 = 0x7FFFFFFFFFFFFFC8LL;
  }
  else
  {
    if ( !v11 )
    {
      v25 = 0;
      v13 = 152;
      v27 = 0;
      goto LABEL_7;
    }
    if ( v11 > 0xD79435E50D7943LL )
      v11 = 0xD79435E50D7943LL;
    v22 = 152 * v11;
  }
  v23 = sub_22077B0(v22);
  v12 = (char *)a2 - v4;
  v8 = a3;
  v27 = v23;
  v25 = v23 + v22;
  v13 = v23 + 152;
LABEL_7:
  if ( &v12[v27] )
    sub_22AF6E0((__int64)&v12[v27], v8);
  if ( a2 != (__int64 *)v4 )
  {
    v14 = v27;
    v15 = (__int64 *)v4;
    while ( 1 )
    {
      if ( v14 )
      {
        v28 = v14;
        sub_22AF6E0(v14, v15);
        v14 = v28;
      }
      v15 += 19;
      if ( a2 == v15 )
        break;
      v14 += 152;
    }
    v13 = v14 + 304;
  }
  if ( a2 != (__int64 *)v3 )
  {
    do
    {
      v16 = v9;
      v17 = v13;
      v9 += 19;
      v13 += 152;
      sub_22AF6E0(v17, v16);
    }
    while ( (__int64 *)v3 != v9 );
  }
  for ( i = v4; i != v3; sub_C7D6A0(*(_QWORD *)(i - 120), 16LL * *(unsigned int *)(i - 104), 8) )
  {
    v19 = *(unsigned int *)(i + 144);
    v20 = *(_QWORD *)(i + 128);
    i += 152LL;
    sub_C7D6A0(v20, 8 * v19, 4);
    sub_C7D6A0(*(_QWORD *)(i - 56), 8LL * *(unsigned int *)(i - 40), 4);
    sub_C7D6A0(*(_QWORD *)(i - 88), 16LL * *(unsigned int *)(i - 72), 8);
  }
  if ( v4 )
    j_j___libc_free_0(v4);
  *a1 = v27;
  a1[1] = v13;
  a1[2] = v25;
  return a1;
}
