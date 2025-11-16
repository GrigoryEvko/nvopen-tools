// Function: sub_39356B0
// Address: 0x39356b0
//
__int64 __fastcall sub_39356B0(__int64 a1, __int64 *a2, __int64 **a3)
{
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // r13
  __int64 v8; // [rsp+8h] [rbp-B8h]
  __int64 v9; // [rsp+10h] [rbp-B0h] BYREF
  __int64 v10; // [rsp+18h] [rbp-A8h]
  int v11; // [rsp+20h] [rbp-A0h]
  int v12; // [rsp+28h] [rbp-98h]
  _BYTE v13[72]; // [rsp+30h] [rbp-90h] BYREF
  unsigned __int64 v14; // [rsp+78h] [rbp-48h]
  __int64 v15; // [rsp+80h] [rbp-40h]
  __int64 v16; // [rsp+88h] [rbp-38h]

  v4 = *(_QWORD *)(a1 + 112);
  v9 = a1;
  v12 = 1;
  v10 = v4;
  v11 = *(unsigned __int8 *)(a1 + 128);
  sub_167FAB0((__int64)v13, 0, 1);
  v14 = 0;
  v15 = 0;
  v16 = 0;
  v8 = sub_3932580((__int64)&v9, a2, a3);
  if ( v14 )
    j_j___libc_free_0(v14);
  sub_167FA50((__int64)v13);
  v5 = *(_QWORD *)(a1 + 120);
  v9 = a1;
  v10 = v5;
  LODWORD(v5) = *(unsigned __int8 *)(a1 + 128);
  v12 = 2;
  v11 = v5;
  sub_167FAB0((__int64)v13, 0, 1);
  v14 = 0;
  v15 = 0;
  v16 = 0;
  v6 = sub_3932580((__int64)&v9, a2, a3) + v8;
  if ( v14 )
    j_j___libc_free_0(v14);
  sub_167FA50((__int64)v13);
  return v6;
}
