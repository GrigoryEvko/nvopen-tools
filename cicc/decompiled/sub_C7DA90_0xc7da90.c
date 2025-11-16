// Function: sub_C7DA90
// Address: 0xc7da90
//
_QWORD *__fastcall sub_C7DA90(_QWORD *a1, __int64 a2, __int64 a3, const char *a4, const char *a5, __int64 a6)
{
  _QWORD *v7; // rax
  _QWORD *v8; // rbx
  const char *v10[4]; // [rsp+0h] [rbp-60h] BYREF
  __int16 v11; // [rsp+20h] [rbp-40h]

  v11 = 261;
  v10[0] = a4;
  v10[1] = a5;
  v7 = (_QWORD *)sub_C7D7A0(24, v10, a3, (__int64)a4, (__int64)a5, a6);
  v8 = v7;
  if ( v7 )
  {
    *v7 = off_49DC9C8;
    sub_C7DA80((__int64)v7, a2, a2 + a3);
  }
  *a1 = v8;
  return a1;
}
