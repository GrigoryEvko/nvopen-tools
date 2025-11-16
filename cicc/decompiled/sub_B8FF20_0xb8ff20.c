// Function: sub_B8FF20
// Address: 0xb8ff20
//
__int64 __fastcall sub_B8FF20(__int64 a1, __int64 a2, __int64 a3)
{
  int v4; // edx
  __int64 v5; // rcx
  __int64 v6; // rdx
  __int64 v7; // rsi
  __int64 v8; // rcx
  unsigned int v9; // eax
  unsigned int v10; // r8d
  __int64 v12; // r13
  __int64 v13; // rax
  unsigned __int8 v14; // [rsp+8h] [rbp-B8h]
  unsigned __int8 v15; // [rsp+8h] [rbp-B8h]
  __int64 v16; // [rsp+8h] [rbp-B8h]
  unsigned __int8 v17; // [rsp+8h] [rbp-B8h]
  __int64 v18; // [rsp+10h] [rbp-B0h]
  __int64 v19; // [rsp+18h] [rbp-A8h]
  unsigned __int8 v20; // [rsp+18h] [rbp-A8h]
  unsigned __int8 v21; // [rsp+18h] [rbp-A8h]
  unsigned __int8 v22; // [rsp+18h] [rbp-A8h]
  unsigned __int8 v23; // [rsp+18h] [rbp-A8h]
  unsigned __int8 v24; // [rsp+18h] [rbp-A8h]
  __int64 v25; // [rsp+20h] [rbp-A0h] BYREF
  unsigned int v26; // [rsp+28h] [rbp-98h]
  __int64 v27; // [rsp+30h] [rbp-90h] BYREF
  unsigned int v28; // [rsp+38h] [rbp-88h]
  __int64 v29; // [rsp+40h] [rbp-80h] BYREF
  unsigned int v30; // [rsp+48h] [rbp-78h]
  __int64 v31; // [rsp+50h] [rbp-70h] BYREF
  unsigned int v32; // [rsp+58h] [rbp-68h]
  __int64 v33; // [rsp+60h] [rbp-60h] BYREF
  unsigned int v34; // [rsp+68h] [rbp-58h]
  __int64 v35; // [rsp+70h] [rbp-50h] BYREF
  unsigned int v36; // [rsp+78h] [rbp-48h]
  __int64 v37; // [rsp+80h] [rbp-40h] BYREF
  unsigned int v38; // [rsp+88h] [rbp-38h]

  v36 = *(_DWORD *)(a3 + 32);
  if ( v36 > 0x40 )
    sub_C43780(&v35, a3 + 24);
  else
    v35 = *(_QWORD *)(a3 + 24);
  v32 = *(_DWORD *)(a2 + 32);
  if ( v32 > 0x40 )
    sub_C43780(&v31, a2 + 24);
  else
    v31 = *(_QWORD *)(a2 + 24);
  sub_AADC30((__int64)&v27, (__int64)&v31, &v35);
  if ( v32 > 0x40 && v31 )
    j_j___libc_free_0_0(v31);
  if ( v36 > 0x40 && v35 )
    j_j___libc_free_0_0(v35);
  v4 = *(_DWORD *)(a1 + 8);
  v5 = (unsigned int)(v4 - 2);
  v6 = (unsigned int)(v4 - 1);
  v7 = *(_QWORD *)(*(_QWORD *)a1 + 8 * v6);
  v19 = v5;
  v8 = *(_QWORD *)(*(_QWORD *)a1 + 8 * v5);
  v18 = v6;
  v36 = *(_DWORD *)(v7 + 32);
  if ( v36 > 0x40 )
  {
    v16 = v8;
    sub_C43780(&v35, v7 + 24);
    v8 = v16;
  }
  else
  {
    v35 = *(_QWORD *)(v7 + 24);
  }
  v26 = *(_DWORD *)(v8 + 32);
  if ( v26 > 0x40 )
    sub_C43780(&v25, v8 + 24);
  else
    v25 = *(_QWORD *)(v8 + 24);
  sub_AADC30((__int64)&v31, (__int64)&v25, &v35);
  if ( v26 > 0x40 && v25 )
    j_j___libc_free_0_0(v25);
  if ( v36 > 0x40 && v35 )
    j_j___libc_free_0_0(v35);
  sub_AB2160((__int64)&v35, (__int64)&v27, (__int64)&v31, 0);
  LOBYTE(v9) = sub_AAF7D0((__int64)&v35);
  v10 = v9;
  if ( !(_BYTE)v9 )
  {
    v10 = 1;
    goto LABEL_44;
  }
  if ( v30 <= 0x40 )
  {
    if ( v29 == v31 )
      goto LABEL_44;
  }
  else
  {
    v10 = sub_C43C50(&v29, &v31);
    if ( (_BYTE)v10 )
      goto LABEL_44;
  }
  if ( v28 > 0x40 )
  {
    v10 = sub_C43C50(&v27, &v33);
    if ( v38 <= 0x40 )
      goto LABEL_26;
    goto LABEL_45;
  }
  LOBYTE(v10) = v27 == v33;
LABEL_44:
  if ( v38 <= 0x40 )
    goto LABEL_26;
LABEL_45:
  if ( v37 )
  {
    v15 = v10;
    j_j___libc_free_0_0(v37);
    v10 = v15;
  }
LABEL_26:
  if ( v36 > 0x40 && v35 )
  {
    v14 = v10;
    j_j___libc_free_0_0(v35);
    v10 = v14;
  }
  if ( (_BYTE)v10 )
  {
    v17 = v10;
    sub_AB3510((__int64)&v35, (__int64)&v31, (__int64)&v27, 0);
    v12 = *(_QWORD *)(a3 + 8);
    *(_QWORD *)(*(_QWORD *)a1 + 8 * v19) = sub_AD8D80(v12, (__int64)&v35);
    v13 = sub_AD8D80(v12, (__int64)&v37);
    v10 = v17;
    *(_QWORD *)(*(_QWORD *)a1 + 8 * v18) = v13;
    if ( v38 > 0x40 && v37 )
    {
      j_j___libc_free_0_0(v37);
      v10 = v17;
    }
    if ( v36 > 0x40 && v35 )
    {
      v24 = v10;
      j_j___libc_free_0_0(v35);
      v10 = v24;
    }
  }
  if ( v34 > 0x40 && v33 )
  {
    v20 = v10;
    j_j___libc_free_0_0(v33);
    v10 = v20;
  }
  if ( v32 > 0x40 && v31 )
  {
    v21 = v10;
    j_j___libc_free_0_0(v31);
    v10 = v21;
  }
  if ( v30 > 0x40 && v29 )
  {
    v22 = v10;
    j_j___libc_free_0_0(v29);
    v10 = v22;
  }
  if ( v28 > 0x40 && v27 )
  {
    v23 = v10;
    j_j___libc_free_0_0(v27);
    return v23;
  }
  return v10;
}
