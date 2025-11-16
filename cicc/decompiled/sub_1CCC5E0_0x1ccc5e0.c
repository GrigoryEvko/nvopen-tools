// Function: sub_1CCC5E0
// Address: 0x1ccc5e0
//
__int64 __fastcall sub_1CCC5E0(__int64 a1, unsigned int a2)
{
  _QWORD *v2; // rax
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 *v5; // r13
  __int64 *v6; // rax
  __int64 v7; // r12
  __int64 v9; // rbx
  _QWORD *v10; // [rsp+8h] [rbp-48h] BYREF
  __int64 *v11; // [rsp+10h] [rbp-40h] BYREF
  __int64 v12; // [rsp+18h] [rbp-38h]
  __int64 v13; // [rsp+20h] [rbp-30h]

  v11 = 0;
  v12 = 0;
  v13 = 0;
  v2 = (_QWORD *)sub_15E0530(a1);
  v3 = sub_1643350(v2);
  v4 = sub_159C470(v3, a2, 0);
  if ( *(_BYTE *)(v4 + 16) == 19 )
    v10 = *(_QWORD **)(v4 + 24);
  else
    v10 = sub_1624210(v4);
  sub_1273E00((__int64)&v11, 0, &v10);
  v9 = v12;
  v5 = v11;
  v6 = (__int64 *)sub_15E0530(a1);
  v7 = sub_1627350(v6, v5, (__int64 *)((v9 - (__int64)v5) >> 3), 0, 1);
  if ( v11 )
    j_j___libc_free_0(v11, v13 - (_QWORD)v11);
  return v7;
}
