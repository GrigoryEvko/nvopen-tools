// Function: sub_22267A0
// Address: 0x22267a0
//
_QWORD *__fastcall sub_22267A0(
        __int64 a1,
        _QWORD *a2,
        unsigned int a3,
        _QWORD *a4,
        int a5,
        char a6,
        __int64 a7,
        _DWORD *a8,
        __int64 a9)
{
  _QWORD *v9; // rax
  _QWORD *v10; // r14
  __int64 v12; // [rsp+18h] [rbp-58h] BYREF
  unsigned __int64 v13[2]; // [rsp+20h] [rbp-50h] BYREF
  _BYTE v14[64]; // [rsp+30h] [rbp-40h] BYREF

  v13[0] = (unsigned __int64)v14;
  v13[1] = 0;
  v14[0] = 0;
  if ( a6 )
    v9 = sub_2224080(a1, a2, a3, a4, a5, a7, a8, (__int64)v13);
  else
    v9 = sub_2225410(a1, a2, a3, a4, a5, a7, a8, (__int64)v13);
  v10 = v9;
  v12 = sub_2208E60(a1, v9);
  sub_2254180(v13[0], a9, a8, &v12);
  if ( (_BYTE *)v13[0] != v14 )
    j___libc_free_0(v13[0]);
  return v10;
}
