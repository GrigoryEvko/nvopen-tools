// Function: sub_1E3CCB0
// Address: 0x1e3ccb0
//
__int64 __fastcall sub_1E3CCB0(__int64 *a1, __int64 *a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v4; // r14
  __int64 v7; // r13
  __int64 v8; // rcx
  __int64 v9; // rdx
  __int64 v10; // rdi
  _QWORD *v11; // rsi
  _QWORD *v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rsi

  result = 0x7FFFFFFFFFFFFFFLL;
  *a1 = a3;
  a1[1] = 0;
  a1[2] = 0;
  if ( a3 <= 0x7FFFFFFFFFFFFFFLL )
    result = a3;
  if ( a3 > 0 )
  {
    v4 = result;
    while ( 1 )
    {
      v7 = 16 * v4;
      result = sub_2207800(16 * v4, &unk_435FF63);
      v8 = result;
      if ( result )
        break;
      v4 >>= 1;
      if ( !v4 )
        return result;
    }
    v9 = *a2;
    v10 = a2[1];
    v11 = (_QWORD *)(result + v7);
    v12 = (_QWORD *)(result + 16);
    a2[1] = 0;
    *(v12 - 2) = v9;
    *(v12 - 1) = v10;
    *a2 = 0;
    if ( v11 == v12 )
    {
      result = v8;
    }
    else
    {
      while ( 1 )
      {
        *v12 = v9;
        v13 = *(v12 - 1);
        v12 += 2;
        *(v12 - 3) = 0;
        *(v12 - 1) = v13;
        *(v12 - 4) = 0;
        if ( v11 == v12 )
          break;
        v9 = *(v12 - 2);
      }
      v14 = v8 + v7 - 32;
      result = v8 + v7 - 16;
      v9 = *(_QWORD *)(v14 + 16);
      v10 = *(_QWORD *)(v14 + 24);
    }
    *(_QWORD *)(result + 8) = 0;
    *(_QWORD *)result = 0;
    *a2 = v9;
    a2[1] = v10;
    a1[2] = v8;
    a1[1] = v4;
  }
  return result;
}
