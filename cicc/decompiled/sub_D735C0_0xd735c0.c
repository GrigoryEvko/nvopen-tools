// Function: sub_D735C0
// Address: 0xd735c0
//
__int64 __fastcall sub_D735C0(__int64 *a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rsi
  unsigned __int64 v4; // rax
  __int64 v5; // rsi
  unsigned __int64 v6; // r12
  _QWORD *v7; // rbx
  _QWORD *v8; // r13
  __int64 v9; // [rsp+0h] [rbp-40h] BYREF
  _QWORD *v10; // [rsp+8h] [rbp-38h]
  __int64 v11; // [rsp+10h] [rbp-30h]
  unsigned int v12; // [rsp+18h] [rbp-28h]

  result = sub_D69420(a1, a2);
  if ( !result )
  {
    v3 = *(_QWORD *)(a2 + 64);
    v9 = 0;
    v10 = 0;
    v11 = 0;
    v12 = 0;
    v4 = sub_D72D40(a1, v3, (__int64)&v9);
    v5 = v12;
    v6 = v4;
    if ( v12 )
    {
      v7 = v10;
      v8 = &v10[4 * v12];
      do
      {
        if ( *v7 != -8192 && *v7 != -4096 )
          sub_D68D70(v7 + 1);
        v7 += 4;
      }
      while ( v8 != v7 );
      v5 = v12;
    }
    sub_C7D6A0((__int64)v10, 32 * v5, 8);
    return v6;
  }
  return result;
}
