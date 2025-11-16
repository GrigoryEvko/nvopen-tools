// Function: sub_15C99E0
// Address: 0x15c99e0
//
__int64 __fastcall sub_15C99E0(__int64 a1, _BYTE *a2, __int64 a3, float a4)
{
  _QWORD *v4; // rax
  __int64 result; // rax
  _QWORD v6[2]; // [rsp+10h] [rbp-70h] BYREF
  _QWORD v7[2]; // [rsp+20h] [rbp-60h] BYREF
  void *v8; // [rsp+30h] [rbp-50h] BYREF
  __int64 v9; // [rsp+38h] [rbp-48h]
  __int64 v10; // [rsp+40h] [rbp-40h]
  __int64 v11; // [rsp+48h] [rbp-38h]
  int v12; // [rsp+50h] [rbp-30h]
  _QWORD *v13; // [rsp+58h] [rbp-28h]

  *(_QWORD *)a1 = a1 + 16;
  if ( a2 )
  {
    sub_15C7EA0((__int64 *)a1, a2, (__int64)&a2[a3]);
  }
  else
  {
    *(_QWORD *)(a1 + 8) = 0;
    *(_BYTE *)(a1 + 16) = 0;
  }
  LOBYTE(v7[0]) = 0;
  v6[1] = 0;
  v6[0] = v7;
  v12 = 1;
  v11 = 0;
  v10 = 0;
  v9 = 0;
  v8 = &unk_49EFBE0;
  v13 = v6;
  sub_16E7B70(&v8, a4);
  if ( v11 != v9 )
    sub_16E7BA0(&v8);
  v4 = v13;
  *(_QWORD *)(a1 + 32) = a1 + 48;
  sub_15C7F50((__int64 *)(a1 + 32), (_BYTE *)*v4, *v4 + v4[1]);
  result = sub_16E7BC0(&v8);
  if ( (_QWORD *)v6[0] != v7 )
    result = j_j___libc_free_0(v6[0], v7[0] + 1LL);
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  return result;
}
