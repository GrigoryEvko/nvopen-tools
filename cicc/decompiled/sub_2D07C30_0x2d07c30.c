// Function: sub_2D07C30
// Address: 0x2d07c30
//
void __fastcall sub_2D07C30(__int64 a1, _BYTE *a2, _QWORD *a3)
{
  __int64 v3; // r14
  _BYTE *v4; // r8
  __int64 v5; // rax
  __int64 v7; // rdx
  bool v8; // cf
  unsigned __int64 v9; // rax
  char *v10; // rcx
  signed __int64 v11; // rdx
  unsigned __int64 v12; // rbx
  char *v13; // r9
  signed __int64 v14; // r10
  char *v15; // r14
  char *v16; // rax
  unsigned __int64 v17; // rbx
  __int64 v18; // rax
  __int64 v19; // [rsp+8h] [rbp-48h]
  __int64 v20; // [rsp+10h] [rbp-40h]
  _BYTE *v21; // [rsp+10h] [rbp-40h]
  signed __int64 v22; // [rsp+10h] [rbp-40h]
  char *v23; // [rsp+18h] [rbp-38h]
  _BYTE *v24; // [rsp+18h] [rbp-38h]
  char *v25; // [rsp+18h] [rbp-38h]
  _BYTE *v26; // [rsp+18h] [rbp-38h]

  v3 = *(_QWORD *)(a1 + 8);
  v4 = *(_BYTE **)a1;
  v5 = (v3 - *(_QWORD *)a1) >> 3;
  if ( v5 == 0xFFFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v7 = 1;
  if ( v5 )
    v7 = (__int64)(*(_QWORD *)(a1 + 8) - *(_QWORD *)a1) >> 3;
  v8 = __CFADD__(v7, v5);
  v9 = v7 + v5;
  v10 = (char *)v8;
  v11 = a2 - v4;
  if ( v8 )
  {
    v17 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v9 )
    {
      v12 = 0;
      goto LABEL_7;
    }
    if ( v9 > 0xFFFFFFFFFFFFFFFLL )
      v9 = 0xFFFFFFFFFFFFFFFLL;
    v17 = 8 * v9;
  }
  v22 = a2 - v4;
  v26 = *(_BYTE **)a1;
  v18 = sub_22077B0(v17);
  v4 = v26;
  v11 = v22;
  v10 = (char *)v18;
  v12 = v18 + v17;
LABEL_7:
  if ( &v10[v11] )
    *(_QWORD *)&v10[v11] = *a3;
  v13 = &v10[v11 + 8];
  v14 = v3 - (_QWORD)a2;
  v15 = &v13[v3 - (_QWORD)a2];
  if ( v11 > 0 )
  {
    v19 = v14;
    v20 = (__int64)&v10[v11 + 8];
    v24 = v4;
    v16 = (char *)memmove(v10, v4, v11);
    v4 = v24;
    v14 = v19;
    v13 = (char *)v20;
    v10 = v16;
    if ( v19 <= 0 )
      goto LABEL_13;
LABEL_15:
    v21 = v4;
    v25 = v10;
    memcpy(v13, a2, v14);
    v4 = v21;
    v10 = v25;
    if ( !v21 )
      goto LABEL_12;
    goto LABEL_13;
  }
  if ( v14 > 0 )
    goto LABEL_15;
  if ( v4 )
  {
LABEL_13:
    v23 = v10;
    j_j___libc_free_0((unsigned __int64)v4);
    v10 = v23;
  }
LABEL_12:
  *(_QWORD *)a1 = v10;
  *(_QWORD *)(a1 + 8) = v15;
  *(_QWORD *)(a1 + 16) = v12;
}
