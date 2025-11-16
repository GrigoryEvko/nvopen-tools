// Function: sub_3892330
// Address: 0x3892330
//
__int64 __fastcall sub_3892330(_QWORD **a1, __int64 *a2, char a3)
{
  unsigned int v4; // r12d
  _QWORD *v6; // [rsp+0h] [rbp-80h] BYREF
  __int64 v7; // [rsp+8h] [rbp-78h]
  _BYTE v8[112]; // [rsp+10h] [rbp-70h] BYREF

  v6 = v8;
  v7 = 0x800000000LL;
  v4 = sub_3892130((__int64)a1, (__int64)&v6);
  if ( !(_BYTE)v4 )
    *a2 = sub_1645600(*a1, v6, (unsigned int)v7, a3);
  if ( v6 != (_QWORD *)v8 )
    _libc_free((unsigned __int64)v6);
  return v4;
}
