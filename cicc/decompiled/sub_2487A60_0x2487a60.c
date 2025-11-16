// Function: sub_2487A60
// Address: 0x2487a60
//
__int64 __fastcall sub_2487A60(__int64 a1, __int64 a2, unsigned __int64 *a3)
{
  int v5; // edx
  __int64 v6; // rax
  unsigned __int64 v7; // rax
  __int64 v8; // rdx
  _QWORD *v9; // rdi
  __int64 v11; // [rsp+10h] [rbp-E0h] BYREF
  char *v12; // [rsp+30h] [rbp-C0h] BYREF
  size_t v13; // [rsp+38h] [rbp-B8h]
  _QWORD v14[2]; // [rsp+40h] [rbp-B0h] BYREF
  _QWORD *v15; // [rsp+50h] [rbp-A0h] BYREF
  __int64 v16; // [rsp+58h] [rbp-98h]
  _BYTE v17[16]; // [rsp+60h] [rbp-90h] BYREF
  unsigned __int64 v18[2]; // [rsp+70h] [rbp-80h] BYREF
  _BYTE v19[16]; // [rsp+80h] [rbp-70h] BYREF
  unsigned __int64 v20; // [rsp+90h] [rbp-60h]
  unsigned __int64 v21; // [rsp+98h] [rbp-58h]
  unsigned __int64 v22; // [rsp+A0h] [rbp-50h]
  int v23; // [rsp+A8h] [rbp-48h]
  int v24; // [rsp+ACh] [rbp-44h]
  __int64 v25; // [rsp+B0h] [rbp-40h]
  __int64 v26; // [rsp+B8h] [rbp-38h]

  v18[0] = (unsigned __int64)v19;
  v18[1] = 0;
  v19[0] = 0;
  v20 = 0;
  v21 = 0;
  v22 = 0;
  v23 = qword_4FE9AE8;
  if ( (_BYTE)qword_4FE93C8 )
  {
    v6 = -8;
    v5 = 8;
  }
  else
  {
    v5 = qword_4FE9A08;
    v6 = -(int)qword_4FE9A08;
  }
  v24 = v5;
  v25 = v6;
  v26 = 0;
  sub_2240AE0(v18, a3 + 29);
  v7 = a3[33];
  v12 = (char *)v14;
  v20 = v7;
  v21 = a3[34];
  v22 = a3[35];
  sub_2240A50((__int64 *)&v12, 1u, 45);
  *v12 = 49;
  v15 = v17;
  if ( (_BYTE)qword_4FEA048 )
  {
    v16 = 0;
    v17[0] = 0;
    sub_2240E30((__int64)&v15, v13 + 34);
    if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v16) <= 0x21 )
      sub_4262D8((__int64)"basic_string::append");
    sub_2241490((unsigned __int64 *)&v15, "__memprof_version_mismatch_check_v", 0x22u);
    sub_2241490((unsigned __int64 *)&v15, v12, v13);
  }
  else
  {
    sub_2485610((__int64 *)&v15, byte_3F871B3, (__int64)byte_3F871B3);
  }
  sub_2A41510(
    (unsigned int)&v11,
    (_DWORD)a3,
    (unsigned int)"memprof.module_ctor",
    19,
    (unsigned int)"__memprof_init",
    14,
    0,
    0,
    0,
    0,
    (__int64)v15,
    v16,
    0);
  v8 = 50;
  if ( HIDWORD(v21) != 37 )
    v8 = 1;
  v26 = v11;
  sub_2A3ED40(a3, v11, v8, 0);
  sub_24876D0((__int64)a3);
  sub_2487800((__int64)a3);
  sub_2487950((__int64)a3);
  if ( v15 != (_QWORD *)v17 )
    j_j___libc_free_0((unsigned __int64)v15);
  if ( v12 != (char *)v14 )
    j_j___libc_free_0((unsigned __int64)v12);
  memset((void *)a1, 0, 0x60u);
  v9 = (_QWORD *)v18[0];
  *(_DWORD *)(a1 + 16) = 2;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_DWORD *)(a1 + 64) = 2;
  *(_BYTE *)(a1 + 76) = 1;
  if ( v9 != (_QWORD *)v19 )
    j_j___libc_free_0((unsigned __int64)v9);
  return a1;
}
