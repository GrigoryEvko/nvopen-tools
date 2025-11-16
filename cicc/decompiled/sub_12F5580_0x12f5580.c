// Function: sub_12F5580
// Address: 0x12f5580
//
__int64 __fastcall sub_12F5580(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  unsigned __int8 v4; // al
  __int64 v5; // r14
  unsigned int v6; // r12d
  char v8; // [rsp+Ch] [rbp-54h] BYREF
  __int64 v9; // [rsp+18h] [rbp-48h] BYREF
  _QWORD v10[2]; // [rsp+20h] [rbp-40h] BYREF
  __int64 (__fastcall *v11)(_QWORD *, _QWORD *, int); // [rsp+30h] [rbp-30h]
  _BYTE *(__fastcall *v12)(_QWORD *, __int64, __int64); // [rsp+38h] [rbp-28h]

  v10[0] = &v8;
  v8 = a4;
  v12 = sub_12F5420;
  v9 = a2;
  v11 = sub_12F5190;
  v4 = sub_167DAB0(a1, &v9, 3, v10);
  v5 = v9;
  v6 = v4;
  if ( v9 )
  {
    sub_1633490(v9);
    j_j___libc_free_0(v5, 736);
  }
  if ( v11 )
    v11(v10, v10, 3);
  return v6;
}
