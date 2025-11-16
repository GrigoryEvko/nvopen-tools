// Function: sub_2BB8FF0
// Address: 0x2bb8ff0
//
__int64 __fastcall sub_2BB8FF0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rbx
  __int64 v8; // r12
  __int64 v9; // r14
  __int64 v10; // r15
  __int64 v11; // r13
  __int64 v12; // rdi
  __int64 v13; // rdx
  __int64 v14; // r14
  __int64 v15; // r13
  __int64 i; // r15
  __int64 v18; // rbx
  __int64 v19; // rsi
  __int64 v20; // [rsp+8h] [rbp-48h]
  __int64 v21; // [rsp+10h] [rbp-40h]
  __int64 v22; // [rsp+18h] [rbp-38h]

  v21 = a1;
  if ( a1 == a2 )
    return a3;
  if ( a2 == a3 )
    return a1;
  v7 = (a2 - a1) >> 6;
  v20 = a1 + a3 - a2;
  v22 = (a3 - a1) >> 6;
  if ( v7 == v22 - v7 )
  {
    v20 = a1;
    v18 = a2;
    do
    {
      v19 = v18;
      v18 += 64;
      sub_2BB8DF0(v20, v19, a3, a4, a5, a6);
      v20 += 64;
    }
    while ( a2 != v20 );
  }
  else
  {
    v8 = v22 - v7;
    if ( v7 >= v22 - v7 )
      goto LABEL_12;
    while ( 1 )
    {
      v9 = v21;
      v10 = v21 + (v7 << 6);
      if ( v8 > 0 )
      {
        v11 = 0;
        do
        {
          v12 = v9;
          ++v11;
          v9 += 64;
          sub_2BB8DF0(v12, v10, a3, a4, a5, a6);
          v10 += 64;
        }
        while ( v8 != v11 );
        v21 += v8 << 6;
      }
      a3 = v22 % v7;
      if ( !(v22 % v7) )
        break;
      v8 = v7;
      v7 -= a3;
      while ( 1 )
      {
        v22 = v8;
        v8 -= v7;
        if ( v7 < v8 )
          break;
LABEL_12:
        v13 = v8 << 6;
        v14 = (v22 << 6) + v21;
        v21 = v14 - (v8 << 6);
        if ( v7 > 0 )
        {
          v15 = v14 - (v8 << 6);
          for ( i = 0; i != v7; ++i )
          {
            v15 -= 64;
            v14 -= 64;
            sub_2BB8DF0(v15, v14, v13, a4, a5, a6);
          }
          v21 -= v7 << 6;
        }
        a3 = v22 % v8;
        v7 = v22 % v8;
        if ( !(v22 % v8) )
          return v20;
      }
    }
  }
  return v20;
}
