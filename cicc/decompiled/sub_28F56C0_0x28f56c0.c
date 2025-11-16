// Function: sub_28F56C0
// Address: 0x28f56c0
//
__int64 __fastcall sub_28F56C0(__int64 a1, __int64 a2, __int16 a3, __int64 *a4, __int64 a5, __int64 *a6)
{
  unsigned int v6; // r12d
  unsigned int v8; // edx
  __int64 v9; // r10
  const void **v12; // r14
  int v13; // eax
  __int64 v14; // rax
  bool v16; // al
  __int64 *v17; // r9
  __int64 v18; // r11
  unsigned __int64 v19; // rcx
  __int64 v20; // rax
  bool v21; // cc
  unsigned __int64 v22; // rcx
  __int64 *v23; // [rsp+8h] [rbp-88h]
  __int64 *v24; // [rsp+10h] [rbp-80h]
  __int64 *v25; // [rsp+10h] [rbp-80h]
  __int64 v26; // [rsp+10h] [rbp-80h]
  __int64 v27; // [rsp+18h] [rbp-78h]
  __int64 v28; // [rsp+18h] [rbp-78h]
  __int64 *v29; // [rsp+18h] [rbp-78h]
  unsigned int v30; // [rsp+20h] [rbp-70h]
  unsigned int v31; // [rsp+20h] [rbp-70h]
  unsigned __int64 v32; // [rsp+30h] [rbp-60h] BYREF
  unsigned int v33; // [rsp+38h] [rbp-58h]
  unsigned __int64 v34; // [rsp+40h] [rbp-50h] BYREF
  unsigned int v35; // [rsp+48h] [rbp-48h]

  v6 = *((unsigned __int8 *)a4 + 36);
  if ( !(_BYTE)v6 )
    return v6;
  v8 = *((_DWORD *)a4 + 6);
  v9 = a2;
  v12 = (const void **)(a4 + 2);
  if ( v8 <= 0x40 )
  {
    if ( !a4[2] )
      return 0;
    v14 = *(_QWORD *)(*a4 + 16);
    if ( !v14 )
      return 0;
  }
  else
  {
    v24 = a6;
    v30 = *((_DWORD *)a4 + 6);
    v13 = sub_C444A0((__int64)(a4 + 2));
    v8 = v30;
    v9 = a2;
    a6 = v24;
    if ( v30 == v13 )
      return 0;
    v14 = *(_QWORD *)(*a4 + 16);
    if ( !v14 )
      return 0;
  }
  if ( *(_QWORD *)(v14 + 8) )
    return 0;
  if ( v8 <= 0x40 )
  {
    v22 = a4[2];
    if ( v22 == *(_QWORD *)a5 )
    {
      v18 = a4[1];
LABEL_26:
      v19 = ~v22 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v8);
      if ( !v8 )
        v19 = 0;
      v32 = v19;
      goto LABEL_14;
    }
    return 0;
  }
  v25 = a6;
  v27 = v9;
  v31 = v8;
  v16 = sub_C43C50((__int64)v12, (const void **)a5);
  v17 = v25;
  if ( !v16 )
    return 0;
  v26 = v27;
  v23 = v17;
  v28 = a4[1];
  v33 = v31;
  sub_C43780((__int64)&v32, v12);
  v8 = v33;
  v18 = v28;
  v9 = v26;
  a6 = v23;
  if ( v33 <= 0x40 )
  {
    v22 = v32;
    goto LABEL_26;
  }
  sub_C43D10((__int64)&v32);
  v8 = v33;
  v19 = v32;
  a6 = v23;
  v9 = v26;
  v18 = v28;
LABEL_14:
  v35 = v8;
  v34 = v19;
  v29 = a6;
  v33 = 0;
  v20 = sub_28EA8C0(v9, a3, SHIBYTE(a3), v18, (__int64)&v34);
  v21 = v35 <= 0x40;
  *v29 = v20;
  if ( !v21 && v34 )
    j_j___libc_free_0_0(v34);
  if ( v33 > 0x40 && v32 )
    j_j___libc_free_0_0(v32);
  if ( *(_DWORD *)(a5 + 8) > 0x40u )
    sub_C43C10((_QWORD *)a5, (__int64 *)v12);
  else
    *(_QWORD *)a5 ^= a4[2];
  if ( *(_BYTE *)*a4 > 0x1Cu )
  {
    sub_D68D20((__int64)&v34, 0, *a4);
    sub_28F19A0(a1 + 64, &v34);
    sub_D68D70(&v34);
  }
  return v6;
}
