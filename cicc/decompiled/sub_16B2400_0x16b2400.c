// Function: sub_16B2400
// Address: 0x16b2400
//
__int64 __fastcall sub_16B2400(__int64 a1, __int64 a2, __int64 a3, double *a4)
{
  const char *v5; // rax
  __int64 v6; // rdx
  double v7; // xmm0_8
  _WORD *v8; // rdi
  unsigned int v9; // r12d
  unsigned __int64 v11; // rdi
  __int64 v12; // rax
  __int64 v13; // r9
  _QWORD v15[3]; // [rsp+10h] [rbp-A0h] BYREF
  char *endptr; // [rsp+28h] [rbp-88h] BYREF
  _QWORD v17[2]; // [rsp+30h] [rbp-80h] BYREF
  __int16 v18; // [rsp+40h] [rbp-70h]
  _WORD *v19; // [rsp+50h] [rbp-60h] BYREF
  __int64 v20; // [rsp+58h] [rbp-58h]
  _WORD v21[40]; // [rsp+60h] [rbp-50h] BYREF

  v15[0] = a2;
  v18 = 261;
  v15[1] = a3;
  v17[0] = v15;
  v19 = v21;
  v20 = 0x2000000000LL;
  v5 = (const char *)sub_16E32E0(v17, &v19);
  v7 = strtod(v5, &endptr);
  if ( *endptr )
  {
    v11 = (unsigned __int64)v19;
    if ( v19 != v21 )
      _libc_free((unsigned __int64)v19);
    v12 = sub_16E8CB0(v11, &endptr, v6);
    v21[0] = 770;
    v18 = 1283;
    v17[0] = "'";
    v17[1] = v15;
    v19 = v17;
    v20 = (__int64)"' value invalid for floating point argument!";
    return (unsigned int)sub_16B1F90(a1, (__int64)&v19, 0, 0, v12, v13);
  }
  else
  {
    v8 = v19;
    v9 = 0;
    *a4 = v7;
    if ( v8 != v21 )
      _libc_free((unsigned __int64)v8);
  }
  return v9;
}
