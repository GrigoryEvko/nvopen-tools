// Function: sub_B48740
// Address: 0xb48740
//
_QWORD *__fastcall sub_B48740(__int64 a1, __int64 a2)
{
  unsigned int v2; // eax
  _QWORD *v4; // rdx
  char v5; // [rsp+1Fh] [rbp-41h] BYREF
  __int64 v6; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v7; // [rsp+28h] [rbp-38h]
  __int64 v8; // [rsp+30h] [rbp-30h] BYREF
  unsigned int v9; // [rsp+38h] [rbp-28h]
  _QWORD *v10; // [rsp+40h] [rbp-20h] BYREF
  unsigned int v11; // [rsp+48h] [rbp-18h]
  _QWORD *v12; // [rsp+50h] [rbp-10h]
  __int64 v13; // [rsp+58h] [rbp-8h]

  v6 = a1;
  v8 = a2;
  v7 = 64;
  v9 = 64;
  sub_C49BE0(&v10, &v6, &v8, &v5);
  if ( v5 )
  {
    LOBYTE(v13) = 0;
    v2 = v11;
  }
  else
  {
    v2 = v11;
    v4 = v10;
    if ( v11 > 0x40 )
      v4 = (_QWORD *)*v10;
    v12 = v4;
    LOBYTE(v13) = 1;
  }
  if ( v2 > 0x40 && v10 )
    j_j___libc_free_0_0(v10);
  if ( v9 > 0x40 && v8 )
    j_j___libc_free_0_0(v8);
  if ( v7 > 0x40 && v6 )
    j_j___libc_free_0_0(v6);
  return v12;
}
