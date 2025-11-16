// Function: sub_12BD5E0
// Address: 0x12bd5e0
//
__int64 *__fastcall sub_12BD5E0(__int64 *a1, __int64 *a2)
{
  __int64 v2; // rax
  _QWORD *v3; // rax
  _QWORD v5[2]; // [rsp+0h] [rbp-70h] BYREF
  _QWORD v6[2]; // [rsp+10h] [rbp-60h] BYREF
  void *v7; // [rsp+20h] [rbp-50h] BYREF
  __int64 v8; // [rsp+28h] [rbp-48h]
  __int64 v9; // [rsp+30h] [rbp-40h]
  __int64 v10; // [rsp+38h] [rbp-38h]
  int v11; // [rsp+40h] [rbp-30h]
  _QWORD *v12; // [rsp+48h] [rbp-28h]

  v12 = v5;
  v2 = *a2;
  v5[0] = v6;
  v5[1] = 0;
  LOBYTE(v6[0]) = 0;
  v11 = 1;
  v10 = 0;
  v9 = 0;
  v8 = 0;
  v7 = &unk_49EFBE0;
  (*(void (__fastcall **)(__int64 *, void **))(v2 + 16))(a2, &v7);
  if ( v10 != v8 )
    sub_16E7BA0(&v7);
  v3 = v12;
  *a1 = (__int64)(a1 + 2);
  sub_12BCB70(a1, (_BYTE *)*v3, *v3 + v3[1]);
  sub_16E7BC0(&v7);
  if ( (_QWORD *)v5[0] != v6 )
    j_j___libc_free_0(v5[0], v6[0] + 1LL);
  return a1;
}
