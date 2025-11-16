// Function: sub_315DEA0
// Address: 0x315dea0
//
void __fastcall sub_315DEA0(__int64 *a1, __int64 a2)
{
  const char *v2; // rax
  void *v3; // rdx
  __int64 **v4; // [rsp+8h] [rbp-D8h] BYREF
  _QWORD v5[2]; // [rsp+10h] [rbp-D0h] BYREF
  __int64 v6[2]; // [rsp+20h] [rbp-C0h] BYREF
  _QWORD v7[2]; // [rsp+30h] [rbp-B0h] BYREF
  __int64 *v8; // [rsp+40h] [rbp-A0h] BYREF
  __int64 v9; // [rsp+50h] [rbp-90h] BYREF
  void *v10[4]; // [rsp+60h] [rbp-80h] BYREF
  __int16 v11; // [rsp+80h] [rbp-60h]
  const char *v12; // [rsp+90h] [rbp-50h] BYREF
  char v13; // [rsp+B0h] [rbp-30h]
  char v14; // [rsp+B1h] [rbp-2Fh]

  v5[0] = a1;
  v5[1] = a2;
  v6[0] = (__int64)v7;
  sub_3157F50(v6, byte_3F871B3, (__int64)byte_3F871B3);
  v2 = sub_BD5D20(*a1);
  v14 = 1;
  v10[2] = (void *)v2;
  v11 = 1283;
  v10[0] = "Block Coverage Inference for ";
  v10[3] = v3;
  v12 = "BCI";
  v4 = (__int64 **)v5;
  v13 = 3;
  sub_315D7A0((__int64)&v8, &v4, (void **)&v12, 0, v10, (__int64)v6);
  if ( v8 != &v9 )
    j_j___libc_free_0((unsigned __int64)v8);
  if ( (_QWORD *)v6[0] != v7 )
    j_j___libc_free_0(v6[0]);
}
