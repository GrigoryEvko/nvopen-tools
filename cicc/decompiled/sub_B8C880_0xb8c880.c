// Function: sub_B8C880
// Address: 0xb8c880
//
__int64 __fastcall sub_B8C880(_QWORD *a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 *v4; // r13
  __int64 *v5; // rbx
  __int64 v6; // rax
  __int64 v7; // rdx
  _BYTE *v8; // rsi
  __int64 v9; // r12
  __int64 v11; // [rsp+8h] [rbp-68h]
  _BYTE *v12; // [rsp+10h] [rbp-60h] BYREF
  __int64 v13; // [rsp+18h] [rbp-58h]
  _BYTE v14[80]; // [rsp+20h] [rbp-50h] BYREF

  v4 = &a2[a3];
  v12 = v14;
  v13 = 0x400000000LL;
  if ( v4 == a2 )
  {
    a3 = 0;
    v8 = v14;
  }
  else
  {
    v5 = a2;
    do
    {
      v6 = sub_B8C140((__int64)a1, *v5, a3, a4);
      v7 = (unsigned int)v13;
      if ( (unsigned __int64)(unsigned int)v13 + 1 > HIDWORD(v13) )
      {
        v11 = v6;
        sub_C8D5F0(&v12, v14, (unsigned int)v13 + 1LL, 8);
        v7 = (unsigned int)v13;
        v6 = v11;
      }
      a4 = (__int64)v12;
      ++v5;
      *(_QWORD *)&v12[8 * v7] = v6;
      a3 = (unsigned int)(v13 + 1);
      LODWORD(v13) = v13 + 1;
    }
    while ( v4 != v5 );
    v8 = v12;
  }
  v9 = sub_B9C770(*a1, v8, a3, 0, 1);
  if ( v12 != v14 )
    _libc_free(v12, v8);
  return v9;
}
