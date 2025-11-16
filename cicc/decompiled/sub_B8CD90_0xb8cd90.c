// Function: sub_B8CD90
// Address: 0xb8cd90
//
__int64 __fastcall sub_B8CD90(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rdx
  __int64 v5; // r12
  __int64 v7; // r13
  __int64 v8; // rax
  unsigned __int64 v9; // rdx
  _QWORD *v10; // [rsp+0h] [rbp-50h] BYREF
  __int64 v11; // [rsp+8h] [rbp-48h]
  _QWORD v12[8]; // [rsp+10h] [rbp-40h] BYREF

  v10 = v12;
  v11 = 0x300000001LL;
  v12[0] = 0;
  if ( a4 )
  {
    v12[1] = a4;
    LODWORD(v11) = 2;
  }
  if ( a3 )
  {
    v7 = sub_B8C130(a1, a2, a3);
    v8 = (unsigned int)v11;
    v9 = (unsigned int)v11 + 1LL;
    if ( v9 > HIDWORD(v11) )
    {
      sub_C8D5F0(&v10, v12, v9, 8);
      v8 = (unsigned int)v11;
    }
    v10[v8] = v7;
    v4 = (unsigned int)(v11 + 1);
    LODWORD(v11) = v11 + 1;
  }
  else
  {
    v4 = (unsigned int)v11;
  }
  v5 = sub_B9C770(*a1, v10, v4, 1, 1);
  sub_BA6610(v5, 0, v5);
  if ( v10 != v12 )
    _libc_free(v10, 0);
  return v5;
}
