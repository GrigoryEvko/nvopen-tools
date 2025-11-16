// Function: sub_17E20D0
// Address: 0x17e20d0
//
__int64 __fastcall sub_17E20D0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rbx
  __int64 result; // rax
  __int64 v7; // rsi
  __int64 v8; // r9
  __int64 v9; // rax
  __int64 v10; // rdi
  __int64 v11; // r8
  __int64 v12; // rdx
  __int64 i; // rcx
  __int64 v14; // r9
  __int64 v15; // r9
  __int64 v16; // rcx
  __int64 v17; // rdx
  __int64 v18; // r10
  __int64 v19; // rax
  __int64 v20; // rdx

  v4 = a3;
  if ( a1 == a2 )
    return v4;
  result = a1;
  if ( a2 == a3 )
    return result;
  v7 = a1;
  v8 = a2 - a1;
  v4 = a1 + a3 - a2;
  v9 = (a3 - a1) >> 3;
  v10 = (a2 - a1) >> 3;
  if ( v10 != v9 - v10 )
  {
    v11 = v9 - v10;
    if ( v10 >= v9 - v10 )
      goto LABEL_12;
    while ( 1 )
    {
      v12 = v7 + 8 * v10;
      if ( v11 > 0 )
      {
        for ( i = 0; i != v11; ++i )
        {
          v14 = *(_QWORD *)(v7 + 8 * i);
          *(_QWORD *)(v7 + 8 * i) = *(_QWORD *)(v12 + 8 * i);
          *(_QWORD *)(v12 + 8 * i) = v14;
        }
        v7 += 8 * v11;
      }
      if ( !(v9 % v10) )
        break;
      v11 = v10;
      v10 -= v9 % v10;
      while ( 1 )
      {
        v9 = v11;
        v11 -= v10;
        if ( v10 < v11 )
          break;
LABEL_12:
        v15 = v7 + 8 * v9;
        v7 = v15 - 8 * v11;
        if ( v10 > 0 )
        {
          v16 = -8;
          v17 = 0;
          do
          {
            v18 = *(_QWORD *)(v7 + v16);
            ++v17;
            *(_QWORD *)(v7 + v16) = *(_QWORD *)(v15 + v16);
            *(_QWORD *)(v15 + v16) = v18;
            v16 -= 8;
          }
          while ( v10 != v17 );
          v7 -= 8 * v10;
        }
        v10 = v9 % v11;
        if ( !(v9 % v11) )
          return v4;
      }
    }
    return v4;
  }
  v19 = 0;
  do
  {
    v20 = *(_QWORD *)(a1 + v19);
    *(_QWORD *)(a1 + v19) = *(_QWORD *)(a2 + v19);
    *(_QWORD *)(a2 + v19) = v20;
    v19 += 8;
  }
  while ( v19 != v8 );
  return a2;
}
