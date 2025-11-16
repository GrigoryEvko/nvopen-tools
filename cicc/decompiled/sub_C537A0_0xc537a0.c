// Function: sub_C537A0
// Address: 0xc537a0
//
__int64 __fastcall sub_C537A0(__int64 a1, __int64 a2, __int64 a3, double *a4)
{
  const char *v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  double v10; // xmm0_8
  char *v11; // rdi
  unsigned int v12; // r12d
  __int64 v14; // rdi
  __int64 v15; // rax
  char *endptr; // [rsp+18h] [rbp-A8h] BYREF
  _QWORD v19[4]; // [rsp+20h] [rbp-A0h] BYREF
  __int16 v20; // [rsp+40h] [rbp-80h]
  _QWORD v21[2]; // [rsp+50h] [rbp-70h] BYREF
  __int64 v22; // [rsp+60h] [rbp-60h]
  char v23; // [rsp+68h] [rbp-58h] BYREF
  __int16 v24; // [rsp+70h] [rbp-50h]

  v19[0] = a2;
  v20 = 261;
  v19[1] = a3;
  v21[0] = &v23;
  v21[1] = 0;
  v22 = 32;
  v5 = (const char *)sub_CA12A0(v19, v21);
  v10 = strtod(v5, &endptr);
  if ( *endptr )
  {
    v14 = v21[0];
    if ( (char *)v21[0] != &v23 )
      _libc_free(v21[0], &endptr);
    v15 = sub_CEADF0(v14, &endptr, v6, v7, v8, v9);
    v24 = 770;
    v20 = 1283;
    v19[0] = "'";
    v19[3] = a3;
    v19[2] = a2;
    v21[0] = v19;
    v22 = (__int64)"' value invalid for floating point argument!";
    return (unsigned int)sub_C53280(a1, (__int64)v21, 0, 0, v15);
  }
  else
  {
    v11 = (char *)v21[0];
    v12 = 0;
    *a4 = v10;
    if ( v11 != &v23 )
      _libc_free(v11, &endptr);
  }
  return v12;
}
