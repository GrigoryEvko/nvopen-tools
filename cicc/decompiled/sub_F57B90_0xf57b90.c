// Function: sub_F57B90
// Address: 0xf57b90
//
__int64 __fastcall sub_F57B90(__int64 a1)
{
  _QWORD **v2; // rbx
  _QWORD **v3; // r14
  _QWORD *v4; // rdi
  _BYTE *v5; // rdi
  __int64 result; // rax
  _QWORD **v7; // r14
  _QWORD **v8; // rbx
  _QWORD *v9; // rdi
  _BYTE *v10; // [rsp+0h] [rbp-60h] BYREF
  __int64 v11; // [rsp+8h] [rbp-58h]
  _BYTE v12[16]; // [rsp+10h] [rbp-50h] BYREF
  _BYTE *v13; // [rsp+20h] [rbp-40h] BYREF
  __int64 v14; // [rsp+28h] [rbp-38h]
  _BYTE v15[48]; // [rsp+30h] [rbp-30h] BYREF

  v11 = 0x100000000LL;
  v14 = 0x100000000LL;
  v10 = v12;
  v13 = v15;
  sub_AE7A50((__int64)&v10, a1, (__int64)&v13);
  v2 = (_QWORD **)v10;
  v3 = (_QWORD **)&v10[8 * (unsigned int)v11];
  if ( v3 != (_QWORD **)v10 )
  {
    do
    {
      v4 = *v2++;
      sub_B43D60(v4);
    }
    while ( v3 != v2 );
  }
  v5 = v13;
  result = (unsigned int)v14;
  v7 = (_QWORD **)&v13[8 * (unsigned int)v14];
  v8 = (_QWORD **)v13;
  if ( v7 != (_QWORD **)v13 )
  {
    do
    {
      v9 = *v8++;
      result = sub_B14290(v9);
    }
    while ( v7 != v8 );
    v5 = v13;
  }
  if ( v5 != v15 )
    result = _libc_free(v5, a1);
  if ( v10 != v12 )
    return _libc_free(v10, a1);
  return result;
}
