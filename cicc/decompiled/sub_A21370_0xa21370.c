// Function: sub_A21370
// Address: 0xa21370
//
__int64 __fastcall sub_A21370(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v7; // rdi
  __int64 v8; // rsi
  __int64 v9; // rax
  __int64 v11[2]; // [rsp+0h] [rbp-B0h] BYREF
  _BYTE v12[8]; // [rsp+10h] [rbp-A0h] BYREF
  __int64 v13; // [rsp+18h] [rbp-98h]
  unsigned int v14; // [rsp+28h] [rbp-88h]
  __int64 v15; // [rsp+30h] [rbp-80h] BYREF
  __int64 v16; // [rsp+38h] [rbp-78h]
  __int64 v17; // [rsp+40h] [rbp-70h]
  __int64 v18; // [rsp+48h] [rbp-68h]
  __int64 v19; // [rsp+50h] [rbp-60h]
  __int64 v20; // [rsp+58h] [rbp-58h]
  unsigned int v21; // [rsp+60h] [rbp-50h]
  __int64 v22; // [rsp+68h] [rbp-48h]
  __int64 v23; // [rsp+70h] [rbp-40h]
  __int64 v24; // [rsp+78h] [rbp-38h]

  sub_C18300(v12);
  v15 = 0;
  v16 = 0;
  v17 = 0;
  v18 = 0;
  v19 = 0;
  v20 = 0;
  v21 = 0;
  v22 = 0;
  v23 = 0;
  v24 = 0;
  sub_C19C00(&v15, a2, 0, v12);
  v11[0] = v15;
  v11[1] = (v16 - v15) >> 2;
  sub_A212B0(a3, 0x20u, v11, a4);
  v7 = v22;
  v8 = 0;
  ++v18;
  *(_QWORD *)(a1 + 8) = v19;
  v9 = v20;
  *(_QWORD *)a1 = 1;
  *(_QWORD *)(a1 + 16) = v9;
  v19 = 0;
  *(_DWORD *)(a1 + 24) = v21;
  v20 = 0;
  v21 = 0;
  if ( v7 )
  {
    j_j___libc_free_0(v7, v24 - v7);
    v7 = v19;
    v8 = 16LL * v21;
  }
  sub_C7D6A0(v7, v8, 8);
  if ( v15 )
    j_j___libc_free_0(v15, v17 - v15);
  sub_C7D6A0(v13, 24LL * v14, 8);
  return a1;
}
