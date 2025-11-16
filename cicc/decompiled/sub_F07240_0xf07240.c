// Function: sub_F07240
// Address: 0xf07240
//
__int64 *__fastcall sub_F07240(__int64 *a1, __int64 a2, __int64 *a3)
{
  __int64 v3; // rsi
  __int64 v4; // rbx
  __int64 v5; // r15
  __int64 *v6; // r14
  __int64 v7; // r13
  __int64 v8; // r12
  __int64 v9; // rax
  __int64 *v12; // [rsp+8h] [rbp-38h]

  v3 = a2 - (_QWORD)a1;
  v4 = v3 >> 3;
  v12 = a1;
  if ( v3 > 0 )
  {
    do
    {
      while ( 1 )
      {
        v5 = v4 >> 1;
        v6 = &v12[v4 >> 1];
        v7 = *v6;
        v8 = sub_B140A0(*a3);
        v9 = sub_B140A0(v7);
        if ( !sub_B445A0(v8, v9) )
          break;
        v4 = v4 - v5 - 1;
        v12 = v6 + 1;
        if ( v4 <= 0 )
          return v12;
      }
      v4 >>= 1;
    }
    while ( v5 > 0 );
  }
  return v12;
}
