// Function: sub_384A050
// Address: 0x384a050
//
__int64 __fastcall sub_384A050(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int16 *v7; // rax
  __int64 v8; // rsi
  __int16 v9; // dx
  __int64 v10; // rax
  __int64 v11; // r8
  _QWORD *v12; // rdi
  unsigned int v13; // ecx
  __int64 v14; // r14
  _QWORD *v15; // rax
  int v16; // edx
  int v17; // ecx
  _QWORD *v18; // rdi
  _QWORD *v19; // rdi
  _QWORD *v20; // rbx
  unsigned int v21; // edx
  unsigned int v22; // r13d
  int v24; // [rsp+8h] [rbp-98h]
  _QWORD *v25; // [rsp+10h] [rbp-90h]
  unsigned __int16 v26; // [rsp+1Eh] [rbp-82h]
  __int64 v27[2]; // [rsp+40h] [rbp-60h] BYREF
  __int64 v28; // [rsp+50h] [rbp-50h] BYREF
  __int64 v29; // [rsp+58h] [rbp-48h]
  unsigned __int16 v30; // [rsp+60h] [rbp-40h]
  __int64 v31; // [rsp+68h] [rbp-38h]

  v7 = *(__int16 **)(a2 + 48);
  v8 = *(_QWORD *)(a1 + 8);
  v9 = *v7;
  v10 = *((_QWORD *)v7 + 1);
  LOWORD(v27[0]) = v9;
  v27[1] = v10;
  sub_33D0340((__int64)&v28, v8, v27);
  v11 = v29;
  v12 = *(_QWORD **)(a1 + 8);
  v13 = (unsigned __int16)v28;
  v26 = v30;
  v14 = v31;
  v28 = 0;
  LODWORD(v29) = 0;
  v15 = sub_33F17F0(v12, 51, (__int64)&v28, v13, v11);
  v17 = v16;
  v18 = v15;
  if ( v28 )
  {
    v24 = v16;
    v25 = v15;
    sub_B91220((__int64)&v28, v28);
    v17 = v24;
    v18 = v25;
  }
  *(_QWORD *)a3 = v18;
  v28 = 0;
  *(_DWORD *)(a3 + 8) = v17;
  v19 = *(_QWORD **)(a1 + 8);
  LODWORD(v29) = 0;
  v20 = sub_33F17F0(v19, 51, (__int64)&v28, v26, v14);
  v22 = v21;
  if ( v28 )
    sub_B91220((__int64)&v28, v28);
  *(_QWORD *)a4 = v20;
  *(_DWORD *)(a4 + 8) = v22;
  return v22;
}
