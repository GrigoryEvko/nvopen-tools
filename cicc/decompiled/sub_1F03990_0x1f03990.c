// Function: sub_1F03990
// Address: 0x1f03990
//
__int64 *__fastcall sub_1F03990(__int64 *a1, __int64 a2, __int64 a3)
{
  _QWORD *v4; // rax
  void *v5; // rdi
  _BYTE *v6; // r15
  size_t v7; // r13
  __int64 v9; // rax
  size_t v10; // [rsp+8h] [rbp-88h] BYREF
  _QWORD v11[2]; // [rsp+10h] [rbp-80h] BYREF
  _QWORD v12[2]; // [rsp+20h] [rbp-70h] BYREF
  void *v13; // [rsp+30h] [rbp-60h] BYREF
  __int64 v14; // [rsp+38h] [rbp-58h]
  __int64 v15; // [rsp+40h] [rbp-50h]
  __int64 v16; // [rsp+48h] [rbp-48h]
  int v17; // [rsp+50h] [rbp-40h]
  _QWORD *v18; // [rsp+58h] [rbp-38h]

  v18 = v11;
  v11[0] = v12;
  v11[1] = 0;
  LOBYTE(v12[0]) = 0;
  v17 = 1;
  v16 = 0;
  v15 = 0;
  v14 = 0;
  v13 = &unk_49EFBE0;
  if ( a3 == a2 + 72 )
  {
    sub_16E7EE0((__int64)&v13, "<entry>", 7u);
  }
  else if ( a3 == a2 + 344 )
  {
    sub_16E7EE0((__int64)&v13, "<exit>", 6u);
  }
  else
  {
    sub_1E1A330(*(_QWORD *)(a3 + 8), (__int64)&v13, 1, 0, 0, 1, 0);
  }
  if ( v16 != v14 )
    sub_16E7BA0((__int64 *)&v13);
  v4 = v18;
  v5 = a1 + 2;
  *a1 = (__int64)(a1 + 2);
  v6 = (_BYTE *)*v4;
  v7 = v4[1];
  if ( v7 + *v4 && !v6 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v10 = v4[1];
  if ( v7 > 0xF )
  {
    v9 = sub_22409D0(a1, &v10, 0);
    *a1 = v9;
    v5 = (void *)v9;
    a1[2] = v10;
LABEL_17:
    memcpy(v5, v6, v7);
    v7 = v10;
    v5 = (void *)*a1;
    goto LABEL_11;
  }
  if ( v7 == 1 )
  {
    *((_BYTE *)a1 + 16) = *v6;
    goto LABEL_11;
  }
  if ( v7 )
    goto LABEL_17;
LABEL_11:
  a1[1] = v7;
  *((_BYTE *)v5 + v7) = 0;
  sub_16E7BC0((__int64 *)&v13);
  if ( (_QWORD *)v11[0] != v12 )
    j_j___libc_free_0(v11[0], v12[0] + 1LL);
  return a1;
}
