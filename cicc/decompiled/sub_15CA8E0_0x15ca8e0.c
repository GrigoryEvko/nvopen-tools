// Function: sub_15CA8E0
// Address: 0x15ca8e0
//
__int64 *__fastcall sub_15CA8E0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbx
  __int64 v7; // rax
  __int64 v8; // rcx
  __int64 v9; // r12
  __int64 v10; // rdx
  const char *v11; // rsi
  _QWORD **v12; // rax
  __int64 v14; // rax
  _QWORD *v15; // [rsp+0h] [rbp-80h] BYREF
  __int64 v16; // [rsp+8h] [rbp-78h]
  _QWORD v17[2]; // [rsp+10h] [rbp-70h] BYREF
  void *v18; // [rsp+20h] [rbp-60h] BYREF
  __int64 v19; // [rsp+28h] [rbp-58h]
  __int64 v20; // [rsp+30h] [rbp-50h]
  __int64 v21; // [rsp+38h] [rbp-48h]
  int v22; // [rsp+40h] [rbp-40h]
  _QWORD *v23; // [rsp+48h] [rbp-38h]

  v15 = v17;
  v16 = 0;
  v6 = *(_QWORD *)(a2 + 88);
  LOBYTE(v17[0]) = 0;
  v18 = &unk_49EFBE0;
  v7 = *(int *)(a2 + 460);
  v22 = 1;
  v21 = 0;
  v20 = 0;
  v19 = 0;
  v23 = &v15;
  if ( (_DWORD)v7 == -1 )
  {
    v14 = *(unsigned int *)(a2 + 96);
    v8 = 5 * v14;
    v9 = v6 + 88 * v14;
    if ( v9 != v6 )
      goto LABEL_3;
LABEL_11:
    v12 = &v15;
    goto LABEL_7;
  }
  v8 = 5 * v7;
  v9 = v6 + 88 * v7;
  if ( v9 == v6 )
    goto LABEL_11;
  do
  {
LABEL_3:
    v10 = *(_QWORD *)(v6 + 40);
    v11 = *(const char **)(v6 + 32);
    v6 += 88;
    sub_16E7EE0(&v18, v11, v10, v8, a5, a6, v15, v16, v17[0]);
  }
  while ( v9 != v6 );
  if ( v21 != v19 )
    sub_16E7BA0(&v18);
  v12 = (_QWORD **)v23;
LABEL_7:
  *a1 = (__int64)(a1 + 2);
  sub_15C7F50(a1, *v12, (__int64)v12[1] + (_QWORD)*v12);
  sub_16E7BC0(&v18);
  if ( v15 != v17 )
    j_j___libc_free_0(v15, v17[0] + 1LL);
  return a1;
}
