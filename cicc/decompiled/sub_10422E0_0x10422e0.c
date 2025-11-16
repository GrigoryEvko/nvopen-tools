// Function: sub_10422E0
// Address: 0x10422e0
//
__int64 __fastcall sub_10422E0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  _QWORD *v4; // rsi
  _BYTE *v5; // rdi
  __int64 result; // rax
  _QWORD *v7; // r13
  _QWORD *v8; // rbx
  _QWORD v9[2]; // [rsp+0h] [rbp-160h] BYREF
  char v10; // [rsp+10h] [rbp-150h]
  __int64 v11; // [rsp+20h] [rbp-140h]
  _BYTE *v12; // [rsp+30h] [rbp-130h] BYREF
  __int64 v13; // [rsp+38h] [rbp-128h]
  _BYTE v14[288]; // [rsp+40h] [rbp-120h] BYREF

  v3 = *(_QWORD *)(a1 + 8);
  v11 = a2;
  v4 = &v12;
  v10 = 0;
  v9[0] = v3;
  v13 = 0x2000000000LL;
  v9[1] = 0;
  v12 = v14;
  sub_D6A180((__int64)v9, (__int64)&v12);
  v5 = v12;
  result = (unsigned int)v13;
  v7 = &v12[8 * (unsigned int)v13];
  if ( v7 != (_QWORD *)v12 )
  {
    v8 = v12;
    do
    {
      v4 = (_QWORD *)*v8++;
      result = sub_10420D0(a1, (__int64)v4);
    }
    while ( v7 != v8 );
    v5 = v12;
  }
  if ( v5 != v14 )
    return _libc_free(v5, v4);
  return result;
}
