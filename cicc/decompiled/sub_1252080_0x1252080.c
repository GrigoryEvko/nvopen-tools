// Function: sub_1252080
// Address: 0x1252080
//
__int64 __fastcall sub_1252080(__int64 a1, __int64 *a2)
{
  __int64 v2; // rdx
  bool v3; // zf
  __int64 v4; // r12
  int v5; // eax
  __int64 v7; // [rsp+0h] [rbp-B0h] BYREF
  __int64 v8; // [rsp+8h] [rbp-A8h]
  int v9; // [rsp+10h] [rbp-A0h]
  int v10; // [rsp+18h] [rbp-98h]
  _BYTE v11[48]; // [rsp+20h] [rbp-90h] BYREF
  __int64 v12; // [rsp+50h] [rbp-60h]
  int v13; // [rsp+58h] [rbp-58h]
  __int64 v14; // [rsp+60h] [rbp-50h]
  __int64 v15; // [rsp+68h] [rbp-48h]
  __int64 v16; // [rsp+70h] [rbp-40h]

  v2 = *(_QWORD *)(a1 + 120);
  v3 = *(_QWORD *)(a1 + 128) == 0;
  v7 = a1;
  v8 = v2;
  LODWORD(v2) = *(unsigned __int8 *)(a1 + 200);
  v10 = !v3;
  v9 = v2;
  sub_C0BFB0((__int64)v11, 0, 0);
  v12 = -1;
  v13 = -1;
  v14 = 0;
  v15 = 0;
  v16 = 0;
  v4 = sub_124F410((__int64)&v7, a2);
  if ( v14 )
    j_j___libc_free_0(v14, v16 - v14);
  sub_C0BF30((__int64)v11);
  if ( *(_QWORD *)(a1 + 128) )
  {
    v8 = *(_QWORD *)(a1 + 128);
    v5 = *(unsigned __int8 *)(a1 + 200);
    v7 = a1;
    v10 = 2;
    v9 = v5;
    sub_C0BFB0((__int64)v11, 0, 0);
    v12 = -1;
    v13 = -1;
    v14 = 0;
    v15 = 0;
    v16 = 0;
    v4 += sub_124F410((__int64)&v7, a2);
    if ( v14 )
      j_j___libc_free_0(v14, v16 - v14);
    sub_C0BF30((__int64)v11);
  }
  return v4;
}
