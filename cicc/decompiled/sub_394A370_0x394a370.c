// Function: sub_394A370
// Address: 0x394a370
//
_QWORD *__fastcall sub_394A370(_QWORD *a1, __int64 *a2)
{
  _QWORD *v3; // rdi
  unsigned __int64 *v5; // [rsp+8h] [rbp-48h] BYREF
  __int64 v6[2]; // [rsp+10h] [rbp-40h] BYREF
  _BYTE v7[48]; // [rsp+20h] [rbp-30h] BYREF

  v6[0] = (__int64)v7;
  v6[1] = 0;
  v7[0] = 0;
  sub_394A2C0(&v5, a2, v6);
  if ( !v5 )
    sub_16BD160((__int64)v6, 1u);
  v3 = (_QWORD *)v6[0];
  *a1 = v5;
  if ( v3 != (_QWORD *)v7 )
    j_j___libc_free_0((unsigned __int64)v3);
  return a1;
}
