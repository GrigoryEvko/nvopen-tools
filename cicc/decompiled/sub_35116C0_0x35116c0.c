// Function: sub_35116C0
// Address: 0x35116c0
//
__int64 *__fastcall sub_35116C0(__int64 *a1, __int64 a2, __int64 *a3, __int64 a4)
{
  __int64 v4; // rsi
  __int64 v5; // rbx
  __int64 v7; // r13
  __int64 *v8; // r14
  __int64 v9; // r12
  __int64 *v12; // [rsp+10h] [rbp-40h]
  unsigned __int64 v13; // [rsp+18h] [rbp-38h]

  v4 = a2 - (_QWORD)a1;
  v5 = v4 >> 3;
  if ( v4 <= 0 )
    return a1;
  v12 = a1;
  do
  {
    while ( 1 )
    {
      v7 = v5 >> 1;
      v8 = &v12[v5 >> 1];
      v9 = *v8;
      v13 = sub_2F06CB0(*(_QWORD *)(a4 + 536), *a3);
      if ( v13 >= sub_2F06CB0(*(_QWORD *)(a4 + 536), v9) )
        break;
      v5 = v5 - v7 - 1;
      v12 = v8 + 1;
      if ( v5 <= 0 )
        return v12;
    }
    v5 >>= 1;
  }
  while ( v7 > 0 );
  return v12;
}
