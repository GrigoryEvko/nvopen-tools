// Function: sub_2634FC0
// Address: 0x2634fc0
//
__int64 __fastcall sub_2634FC0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v5; // rbx
  __int64 v6; // r12
  __int64 v7; // r14
  __int64 v8; // r15
  __int64 v9; // r13
  __int64 v10; // rsi
  __int64 v11; // rax
  __int64 v12; // r14
  __int64 v13; // r13
  __int64 v14; // r15
  __int64 v15; // rdi
  __int64 v16; // rbx
  __int64 v17; // r13
  __int64 v18; // rsi
  __int64 v19; // rdi
  __int64 v20; // [rsp+8h] [rbp-48h]
  __int64 v21; // [rsp+10h] [rbp-40h]
  signed __int64 v22; // [rsp+18h] [rbp-38h]

  result = a3;
  v21 = a1;
  if ( a1 != a2 )
  {
    result = a1;
    if ( a2 != a3 )
    {
      v20 = a1 + a3 - a2;
      v5 = 0xAAAAAAAAAAAAAAABLL * ((a2 - a1) >> 4);
      v22 = 0xAAAAAAAAAAAAAAABLL * ((a3 - a1) >> 4);
      if ( v5 == v22 - v5 )
      {
        v16 = a1;
        v17 = a2;
        do
        {
          v18 = v17;
          v19 = v16;
          v16 += 48;
          v17 += 48;
          sub_1888690(v19, v18);
        }
        while ( a2 != v16 );
        return a2;
      }
      else
      {
        v6 = v22 - v5;
        if ( v5 >= v22 - v5 )
          goto LABEL_12;
        while ( 1 )
        {
          v7 = v21;
          if ( v6 > 0 )
          {
            v8 = v21 + 48 * v5;
            v9 = 0;
            do
            {
              v10 = v8;
              ++v9;
              v8 += 48;
              sub_1888690(v7, v10);
              v7 += 48;
            }
            while ( v6 != v9 );
            v21 += 48 * v6;
          }
          if ( !(v22 % v5) )
            break;
          v6 = v5;
          v5 -= v22 % v5;
          while ( 1 )
          {
            v22 = v6;
            v6 -= v5;
            if ( v5 < v6 )
              break;
LABEL_12:
            v11 = v21 + 48 * v22;
            v21 = v11 - 48 * v6;
            if ( v5 > 0 )
            {
              v12 = v11 - 48 * v6 - 48;
              v13 = v11 - 48;
              v14 = 0;
              do
              {
                v15 = v12;
                ++v14;
                v12 -= 48;
                sub_1888690(v15, v13);
                v13 -= 48;
              }
              while ( v5 != v14 );
              v21 += -48 * v5;
            }
            v5 = v22 % v6;
            if ( !(v22 % v6) )
              return v20;
          }
        }
        return v20;
      }
    }
  }
  return result;
}
