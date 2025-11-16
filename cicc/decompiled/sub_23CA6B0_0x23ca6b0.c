// Function: sub_23CA6B0
// Address: 0x23ca6b0
//
_QWORD *__fastcall sub_23CA6B0(_QWORD *a1, char **a2, __int64 *a3)
{
  _QWORD *v4; // rdi
  __int64 v6[2]; // [rsp+0h] [rbp-70h] BYREF
  _BYTE v7[16]; // [rsp+10h] [rbp-60h] BYREF
  unsigned __int64 v8[4]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v9; // [rsp+40h] [rbp-30h]

  v6[0] = (__int64)v7;
  v6[1] = 0;
  v7[0] = 0;
  sub_23CA5F0(v8, a2, a3, v6);
  if ( !v8[0] )
  {
    v9 = 260;
    v8[0] = (unsigned __int64)v6;
    sub_C64D30((__int64)v8, 1u);
  }
  v4 = (_QWORD *)v6[0];
  *a1 = v8[0];
  if ( v4 != (_QWORD *)v7 )
    j_j___libc_free_0((unsigned __int64)v4);
  return a1;
}
