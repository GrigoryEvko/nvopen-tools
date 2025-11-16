// Function: sub_FE8740
// Address: 0xfe8740
//
__int64 *__fastcall sub_FE8740(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  __int64 v6; // rax
  unsigned int v7; // eax
  unsigned int v8; // eax
  unsigned int v9; // ebx
  __int64 v10; // rax
  unsigned int v11; // ebx
  __int64 *v12; // [rsp+10h] [rbp-A0h] BYREF
  unsigned int v13; // [rsp+18h] [rbp-98h]
  __int64 v14; // [rsp+20h] [rbp-90h] BYREF
  unsigned int v15; // [rsp+28h] [rbp-88h]
  unsigned __int64 v16; // [rsp+30h] [rbp-80h] BYREF
  unsigned int v17; // [rsp+38h] [rbp-78h]
  unsigned __int64 v18; // [rsp+40h] [rbp-70h] BYREF
  unsigned int v19; // [rsp+48h] [rbp-68h]
  unsigned __int64 v20; // [rsp+50h] [rbp-60h] BYREF
  unsigned int v21; // [rsp+58h] [rbp-58h]
  __int64 *v22; // [rsp+60h] [rbp-50h] BYREF
  __int64 v23; // [rsp+68h] [rbp-48h]
  __int64 v24; // [rsp+70h] [rbp-40h] BYREF
  char v25; // [rsp+80h] [rbp-30h]

  sub_B2EE70((__int64)&v24, a2, a4);
  if ( !v25 )
  {
    LOBYTE(v23) = 0;
    return v22;
  }
  v13 = 128;
  sub_C43690((__int64)&v12, v24, 0);
  v15 = 128;
  sub_C43690((__int64)&v14, a3, 0);
  v6 = *(_QWORD *)(a1 + 8);
  v17 = 128;
  sub_C43690((__int64)&v16, *(_QWORD *)(v6 + 16), 0);
  sub_C47360((__int64)&v12, &v14);
  v7 = v17;
  v19 = v17;
  if ( v17 <= 0x40 )
  {
    v18 = v16;
    goto LABEL_6;
  }
  sub_C43780((__int64)&v18, (const void **)&v16);
  v7 = v19;
  if ( v19 <= 0x40 )
  {
LABEL_6:
    if ( v7 == 1 )
      v18 = 0;
    else
      v18 >>= 1;
    goto LABEL_8;
  }
  sub_C482E0((__int64)&v18, 1u);
LABEL_8:
  sub_C45EE0((__int64)&v18, (__int64 *)&v12);
  v8 = v19;
  v19 = 0;
  v21 = v8;
  v20 = v18;
  sub_C4A1D0((__int64)&v22, (__int64)&v20, (__int64)&v16);
  if ( v13 > 0x40 && v12 )
    j_j___libc_free_0_0(v12);
  v12 = v22;
  v13 = v23;
  if ( v21 > 0x40 && v20 )
    j_j___libc_free_0_0(v20);
  if ( v19 > 0x40 && v18 )
    j_j___libc_free_0_0(v18);
  v9 = v13;
  if ( v13 > 0x40 )
  {
    v11 = v9 - sub_C444A0((__int64)&v12);
    v10 = -1;
    if ( v11 <= 0x40 )
      v10 = *v12;
  }
  else
  {
    v10 = (__int64)v12;
  }
  v22 = (__int64 *)v10;
  LOBYTE(v23) = 1;
  if ( v17 > 0x40 && v16 )
    j_j___libc_free_0_0(v16);
  if ( v15 > 0x40 && v14 )
    j_j___libc_free_0_0(v14);
  if ( v13 > 0x40 && v12 )
    j_j___libc_free_0_0(v12);
  return v22;
}
