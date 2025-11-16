// Function: sub_31DE8D0
// Address: 0x31de8d0
//
__int64 __fastcall sub_31DE8D0(__int64 a1, _DWORD a2, _DWORD a3, _DWORD a4, _DWORD a5, _DWORD a6, char a7)
{
  __int64 v7; // rax
  __int64 v8; // rdi
  __int64 v9; // r12
  const char *v11[4]; // [rsp+0h] [rbp-B0h] BYREF
  __int16 v12; // [rsp+20h] [rbp-90h]
  const char *v13; // [rsp+30h] [rbp-80h] BYREF
  const char *v14; // [rsp+38h] [rbp-78h]
  __int64 v15; // [rsp+40h] [rbp-70h]
  _BYTE v16[104]; // [rsp+48h] [rbp-68h] BYREF

  v13 = v16;
  v14 = 0;
  v15 = 60;
  v7 = sub_31DA930(a1);
  sub_E405D0((__int64)&v13, &a7, v7);
  v8 = *(_QWORD *)(a1 + 216);
  v12 = 261;
  v11[0] = v13;
  v11[1] = v14;
  v9 = sub_E6C460(v8, v11);
  if ( v13 != v16 )
    _libc_free((unsigned __int64)v13);
  return v9;
}
