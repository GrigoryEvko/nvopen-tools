// Function: sub_16E5BC0
// Address: 0x16e5bc0
//
const char *__fastcall sub_16E5BC0(__int64 a1, __int64 a2, __int64 a3, float *a4)
{
  const char *v5; // rax
  float v6; // xmm0_4
  _BYTE *v7; // rdi
  _QWORD v9[3]; // [rsp+0h] [rbp-80h] BYREF
  char *endptr; // [rsp+18h] [rbp-68h] BYREF
  const char *v11; // [rsp+20h] [rbp-60h] BYREF
  __int16 v12; // [rsp+30h] [rbp-50h]
  unsigned __int64 v13[2]; // [rsp+40h] [rbp-40h] BYREF
  _BYTE v14[48]; // [rsp+50h] [rbp-30h] BYREF

  v12 = 261;
  v9[0] = a1;
  v9[1] = a2;
  v11 = (const char *)v9;
  v13[0] = (unsigned __int64)v14;
  v13[1] = 0x2000000000LL;
  v5 = sub_16E32E0(&v11, (unsigned int *)v13);
  v6 = strtof(v5, &endptr);
  if ( *endptr )
  {
    if ( (_BYTE *)v13[0] != v14 )
      _libc_free(v13[0]);
    return "invalid floating point number";
  }
  else
  {
    v7 = (_BYTE *)v13[0];
    *a4 = v6;
    if ( v7 != v14 )
      _libc_free((unsigned __int64)v7);
    return 0;
  }
}
