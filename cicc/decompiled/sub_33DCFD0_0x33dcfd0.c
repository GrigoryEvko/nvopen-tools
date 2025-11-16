// Function: sub_33DCFD0
// Address: 0x33dcfd0
//
_QWORD *__fastcall sub_33DCFD0(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, unsigned int a5)
{
  _QWORD *v6; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v7; // [rsp+18h] [rbp-58h]
  _QWORD *v8; // [rsp+20h] [rbp-50h]
  __int64 v9; // [rsp+28h] [rbp-48h]
  unsigned __int64 v10; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v11; // [rsp+38h] [rbp-38h]
  unsigned __int64 v12; // [rsp+40h] [rbp-30h]
  unsigned int v13; // [rsp+48h] [rbp-28h]
  char v14; // [rsp+50h] [rbp-20h]

  sub_33DC4D0((__int64)&v10, a1, a2, a3, a4, a5);
  if ( v14 )
  {
    sub_AB0910((__int64)&v6, (__int64)&v10);
    LOBYTE(v9) = 1;
    if ( v7 <= 0x40 )
    {
      v8 = v6;
    }
    else
    {
      v8 = (_QWORD *)*v6;
      j_j___libc_free_0_0((unsigned __int64)v6);
    }
    if ( v14 )
    {
      v14 = 0;
      if ( v13 > 0x40 && v12 )
        j_j___libc_free_0_0(v12);
      if ( v11 > 0x40 && v10 )
        j_j___libc_free_0_0(v10);
    }
  }
  else
  {
    LOBYTE(v9) = 0;
  }
  return v8;
}
