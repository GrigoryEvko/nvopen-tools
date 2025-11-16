// Function: sub_2147AE0
// Address: 0x2147ae0
//
__int64 __fastcall sub_2147AE0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  char *v7; // rax
  __int64 v8; // rsi
  char v9; // dl
  __int64 v10; // rax
  __int64 v11; // r8
  _QWORD *v12; // rdi
  unsigned int v13; // ecx
  __int64 v14; // r14
  __int64 v15; // r9
  _QWORD *v16; // rax
  int v17; // edx
  __int64 v18; // r9
  int v19; // ecx
  _QWORD *v20; // rdi
  _QWORD *v21; // rdi
  _QWORD *v22; // rbx
  unsigned int v23; // edx
  unsigned int v24; // r13d
  int v26; // [rsp+8h] [rbp-98h]
  _QWORD *v27; // [rsp+10h] [rbp-90h]
  unsigned __int8 v28; // [rsp+1Fh] [rbp-81h]
  _QWORD v29[2]; // [rsp+40h] [rbp-60h] BYREF
  __int64 v30; // [rsp+50h] [rbp-50h] BYREF
  __int64 v31; // [rsp+58h] [rbp-48h]
  unsigned __int8 v32; // [rsp+60h] [rbp-40h]
  __int64 v33; // [rsp+68h] [rbp-38h]

  v7 = *(char **)(a2 + 40);
  v8 = *(_QWORD *)(a1 + 8);
  v9 = *v7;
  v10 = *((_QWORD *)v7 + 1);
  LOBYTE(v29[0]) = v9;
  v29[1] = v10;
  sub_1D19A30((__int64)&v30, v8, v29);
  v11 = v31;
  v12 = *(_QWORD **)(a1 + 8);
  v13 = (unsigned __int8)v30;
  v28 = v32;
  v14 = v33;
  v30 = 0;
  LODWORD(v31) = 0;
  v16 = sub_1D2B300(v12, 0x30u, (__int64)&v30, v13, v11, v15);
  v19 = v17;
  v20 = v16;
  if ( v30 )
  {
    v26 = v17;
    v27 = v16;
    sub_161E7C0((__int64)&v30, v30);
    v19 = v26;
    v20 = v27;
  }
  *(_QWORD *)a3 = v20;
  v30 = 0;
  *(_DWORD *)(a3 + 8) = v19;
  v21 = *(_QWORD **)(a1 + 8);
  LODWORD(v31) = 0;
  v22 = sub_1D2B300(v21, 0x30u, (__int64)&v30, v28, v14, v18);
  v24 = v23;
  if ( v30 )
    sub_161E7C0((__int64)&v30, v30);
  *(_QWORD *)a4 = v22;
  *(_DWORD *)(a4 + 8) = v24;
  return v24;
}
