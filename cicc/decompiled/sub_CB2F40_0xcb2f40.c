// Function: sub_CB2F40
// Address: 0xcb2f40
//
const char *__fastcall sub_CB2F40(const char *a1, const char *a2, __int64 a3, float *a4)
{
  const char *v5; // rax
  float v6; // xmm0_4
  _BYTE *v7; // rdi
  char *endptr; // [rsp+8h] [rbp-88h] BYREF
  const char *v10[4]; // [rsp+10h] [rbp-80h] BYREF
  __int16 v11; // [rsp+30h] [rbp-60h]
  _QWORD v12[3]; // [rsp+40h] [rbp-50h] BYREF
  _BYTE v13[56]; // [rsp+58h] [rbp-38h] BYREF

  v10[0] = a1;
  v10[1] = a2;
  v11 = 261;
  v12[0] = v13;
  v12[1] = 0;
  v12[2] = 32;
  v5 = sub_CA12A0(v10, v12);
  v6 = strtof(v5, &endptr);
  if ( *endptr )
  {
    if ( (_BYTE *)v12[0] != v13 )
      _libc_free(v12[0], &endptr);
    return "invalid floating point number";
  }
  else
  {
    v7 = (_BYTE *)v12[0];
    *a4 = v6;
    if ( v7 != v13 )
      _libc_free(v7, &endptr);
    return 0;
  }
}
