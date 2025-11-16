// Function: sub_DC7ED0
// Address: 0xdc7ed0
//
_QWORD *__fastcall sub_DC7ED0(__int64 *a1, __int64 a2, __int64 a3, unsigned int a4, unsigned int a5)
{
  _QWORD *v5; // r12
  _QWORD v7[2]; // [rsp+0h] [rbp-30h] BYREF
  _QWORD v8[4]; // [rsp+10h] [rbp-20h] BYREF

  v7[0] = v8;
  v8[0] = a2;
  v8[1] = a3;
  v7[1] = 0x200000002LL;
  v5 = sub_DC7EB0(a1, (__int64)v7, a4, a5);
  if ( (_QWORD *)v7[0] != v8 )
    _libc_free(v7[0], v7);
  return v5;
}
