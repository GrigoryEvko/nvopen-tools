// Function: sub_3711700
// Address: 0x3711700
//
void __fastcall sub_3711700(__int64 a1, const void *a2, _BYTE *a3)
{
  __int64 v3; // r14
  _BYTE *v4; // r9
  __int64 v5; // rax
  __int64 v7; // rdx
  const void *v8; // r8
  bool v9; // cf
  __int64 v10; // rax
  unsigned __int64 v11; // r13
  signed __int64 v12; // rdx
  unsigned __int64 v13; // r13
  _BYTE *v14; // r12
  __int64 v15; // rax
  char *v16; // r11
  signed __int64 v17; // r10
  char *v18; // r14
  const void *v19; // [rsp+0h] [rbp-50h]
  __int64 v20; // [rsp+8h] [rbp-48h]
  size_t n; // [rsp+10h] [rbp-40h]
  signed __int64 na; // [rsp+10h] [rbp-40h]
  _BYTE *v23; // [rsp+18h] [rbp-38h]
  _BYTE *v24; // [rsp+18h] [rbp-38h]
  _BYTE *v25; // [rsp+18h] [rbp-38h]

  v3 = *(_QWORD *)(a1 + 8);
  v4 = *(_BYTE **)a1;
  v5 = v3 - *(_QWORD *)a1;
  if ( v5 == 0x7FFFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v7 = 1;
  v8 = a2;
  if ( v5 )
    v7 = *(_QWORD *)(a1 + 8) - *(_QWORD *)a1;
  v9 = __CFADD__(v7, v5);
  v10 = v7 + v5;
  v11 = v10;
  v12 = (_BYTE *)a2 - v4;
  if ( v9 || v10 < 0 )
  {
    v11 = 0x7FFFFFFFFFFFFFFFLL;
  }
  else if ( !v10 )
  {
    v13 = 0;
    v14 = 0;
    goto LABEL_10;
  }
  n = (_BYTE *)a2 - v4;
  v23 = *(_BYTE **)a1;
  v15 = sub_22077B0(v11);
  v4 = v23;
  v12 = n;
  v8 = a2;
  v14 = (_BYTE *)v15;
  v13 = v15 + v11;
LABEL_10:
  if ( &v14[v12] )
    v14[v12] = *a3;
  v16 = &v14[v12 + 1];
  v17 = v3 - (_QWORD)v8;
  v18 = &v16[v3 - (_QWORD)v8];
  if ( v12 > 0 )
  {
    v19 = v8;
    v20 = (__int64)&v14[v12 + 1];
    na = v17;
    v24 = v4;
    memmove(v14, v4, v12);
    v4 = v24;
    v17 = na;
    v16 = (char *)v20;
    v8 = v19;
    if ( na <= 0 )
      goto LABEL_16;
LABEL_18:
    v25 = v4;
    memcpy(v16, v8, v17);
    v4 = v25;
    if ( !v25 )
      goto LABEL_15;
    goto LABEL_16;
  }
  if ( v17 > 0 )
    goto LABEL_18;
  if ( v4 )
LABEL_16:
    j_j___libc_free_0((unsigned __int64)v4);
LABEL_15:
  *(_QWORD *)a1 = v14;
  *(_QWORD *)(a1 + 8) = v18;
  *(_QWORD *)(a1 + 16) = v13;
}
