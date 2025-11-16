// Function: sub_C0C0F0
// Address: 0xc0c0f0
//
__int64 __fastcall sub_C0C0F0(__int64 a1, __int64 a2)
{
  _DWORD *v3; // rsi
  __int64 v4; // r14
  _BYTE *v5; // rdx
  _BYTE *v6; // rax
  _BYTE *v7; // rsi
  __int64 result; // rax
  _BYTE *v9; // [rsp+0h] [rbp-40h] BYREF
  __int64 v10; // [rsp+8h] [rbp-38h]
  __int64 v11; // [rsp+10h] [rbp-30h]
  _BYTE v12[40]; // [rsp+18h] [rbp-28h] BYREF

  v3 = v12;
  v4 = *(_QWORD *)(a1 + 32);
  v9 = v12;
  v10 = 0;
  v11 = 0;
  if ( v4 )
  {
    sub_C8D290(&v9, v12, v4, 1);
    v3 = v9;
    v5 = &v9[v4];
    v6 = &v9[v10];
    if ( &v9[v10] != &v9[v4] )
    {
      do
      {
        if ( v6 )
          *v6 = 0;
        ++v6;
      }
      while ( v5 != v6 );
      v3 = v9;
    }
    v10 = v4;
  }
  sub_C0BFF0(a1, v3);
  v7 = v9;
  result = sub_CB6200(a2, v9, v10);
  if ( v9 != v12 )
    return _libc_free(v9, v7);
  return result;
}
