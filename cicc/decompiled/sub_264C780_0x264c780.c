// Function: sub_264C780
// Address: 0x264c780
//
__int64 *__fastcall sub_264C780(_QWORD *a1)
{
  __int64 *v1; // rax
  __int64 v2; // rsi
  __int64 *v3; // rbx
  __int64 *v4; // rdx
  __int64 *v5; // r14
  __int64 v6; // rax
  __int64 *result; // rax
  __int64 v8; // rsi
  __int64 *v9; // rbx
  __int64 *v10; // rdx
  __int64 *v11; // r13
  __int64 v12; // rax
  __int64 v13; // [rsp+0h] [rbp-40h] BYREF
  __int64 v14; // [rsp+8h] [rbp-38h]
  __int64 v15; // [rsp+10h] [rbp-30h]
  __int64 v16; // [rsp+18h] [rbp-28h]

  v1 = (__int64 *)a1[9];
  if ( (__int64 *)a1[10] != v1 )
  {
    v2 = *v1;
    v13 = 0;
    v14 = 0;
    v15 = 0;
    v16 = 0;
    sub_264A680((__int64)&v13, v2 + 24);
    v3 = (__int64 *)sub_263EC80(a1 + 9);
    v5 = v4;
    while ( v5 != v3 )
    {
      v6 = *v3;
      v3 += 2;
      sub_264C6F0((__int64)&v13, v6 + 24);
    }
    sub_C7D6A0(v14, 4LL * (unsigned int)v16, 4);
  }
  result = (__int64 *)a1[6];
  if ( (__int64 *)a1[7] != result )
  {
    v8 = *result;
    v13 = 0;
    v14 = 0;
    v15 = 0;
    v16 = 0;
    sub_264A680((__int64)&v13, v8 + 24);
    v9 = (__int64 *)sub_263EC80(a1 + 6);
    v11 = v10;
    while ( v11 != v9 )
    {
      v12 = *v9;
      v9 += 2;
      sub_264C6F0((__int64)&v13, v12 + 24);
    }
    return (__int64 *)sub_C7D6A0(v14, 4LL * (unsigned int)v16, 4);
  }
  return result;
}
