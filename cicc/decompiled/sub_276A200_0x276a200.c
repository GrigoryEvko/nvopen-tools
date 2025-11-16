// Function: sub_276A200
// Address: 0x276a200
//
void __fastcall sub_276A200(__int64 a1, const void *a2, _QWORD *a3)
{
  __int64 v4; // r14
  _BYTE *v5; // r12
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rdx
  const void *v8; // r8
  bool v9; // cf
  unsigned __int64 v10; // rax
  signed __int64 v11; // rdx
  unsigned __int64 v12; // r13
  char *v13; // r15
  char *v14; // r9
  signed __int64 v15; // r10
  char *v16; // r14
  unsigned __int64 v17; // r13
  __int64 v18; // rax
  const void *v19; // [rsp+0h] [rbp-50h]
  __int64 v20; // [rsp+8h] [rbp-48h]
  _QWORD *v21; // [rsp+8h] [rbp-48h]
  const void *v22; // [rsp+10h] [rbp-40h]
  __int64 v23; // [rsp+10h] [rbp-40h]
  size_t n; // [rsp+18h] [rbp-38h]

  v4 = *(_QWORD *)(a1 + 8);
  v5 = *(_BYTE **)a1;
  v6 = 0xCCCCCCCCCCCCCCCDLL * ((v4 - *(_QWORD *)a1) >> 4);
  if ( v6 == 0x199999999999999LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v7 = 1;
  v8 = a2;
  if ( v6 )
    v7 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)(*(_QWORD *)(a1 + 8) - *(_QWORD *)a1) >> 4);
  v9 = __CFADD__(v7, v6);
  v10 = v7 - 0x3333333333333333LL * ((__int64)(*(_QWORD *)(a1 + 8) - *(_QWORD *)a1) >> 4);
  v11 = (_BYTE *)a2 - v5;
  if ( v9 )
  {
    v17 = 0x7FFFFFFFFFFFFFD0LL;
  }
  else
  {
    if ( !v10 )
    {
      v12 = 0;
      v13 = 0;
      goto LABEL_7;
    }
    if ( v10 > 0x199999999999999LL )
      v10 = 0x199999999999999LL;
    v17 = 80 * v10;
  }
  v21 = a3;
  v18 = sub_22077B0(v17);
  v11 = (_BYTE *)a2 - v5;
  v8 = a2;
  a3 = v21;
  v13 = (char *)v18;
  v12 = v18 + v17;
LABEL_7:
  if ( &v13[v11] )
  {
    v22 = v8;
    n = v11;
    sub_276A0C0((__int64 *)&v13[v11], a3);
    v8 = v22;
    v11 = n;
  }
  v14 = &v13[v11 + 80];
  v15 = v4 - (_QWORD)v8;
  v16 = &v14[v4 - (_QWORD)v8];
  if ( v11 > 0 )
  {
    v19 = v8;
    v20 = v15;
    v23 = (__int64)&v13[v11 + 80];
    memmove(v13, v5, v11);
    v15 = v20;
    v14 = (char *)v23;
    v8 = v19;
    if ( v20 <= 0 )
      goto LABEL_13;
LABEL_15:
    memcpy(v14, v8, v15);
    if ( !v5 )
      goto LABEL_12;
    goto LABEL_13;
  }
  if ( v15 > 0 )
    goto LABEL_15;
  if ( v5 )
LABEL_13:
    j_j___libc_free_0((unsigned __int64)v5);
LABEL_12:
  *(_QWORD *)a1 = v13;
  *(_QWORD *)(a1 + 8) = v16;
  *(_QWORD *)(a1 + 16) = v12;
}
