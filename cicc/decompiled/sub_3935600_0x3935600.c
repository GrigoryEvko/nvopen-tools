// Function: sub_3935600
// Address: 0x3935600
//
__int64 __fastcall sub_3935600(__int64 a1, __int64 *a2, __int64 **a3)
{
  __int64 v4; // rax
  __int64 v5; // r12
  _QWORD v7[2]; // [rsp+0h] [rbp-A0h] BYREF
  int v8; // [rsp+10h] [rbp-90h]
  int v9; // [rsp+18h] [rbp-88h]
  _BYTE v10[72]; // [rsp+20h] [rbp-80h] BYREF
  unsigned __int64 v11; // [rsp+68h] [rbp-38h]
  __int64 v12; // [rsp+70h] [rbp-30h]
  __int64 v13; // [rsp+78h] [rbp-28h]

  v4 = *(_QWORD *)(a1 + 112);
  v7[0] = a1;
  v9 = 0;
  v7[1] = v4;
  v8 = *(unsigned __int8 *)(a1 + 120);
  sub_167FAB0((__int64)v10, 0, 1);
  v11 = 0;
  v12 = 0;
  v13 = 0;
  v5 = sub_3932580((__int64)v7, a2, a3);
  if ( v11 )
    j_j___libc_free_0(v11);
  sub_167FA50((__int64)v10);
  return v5;
}
