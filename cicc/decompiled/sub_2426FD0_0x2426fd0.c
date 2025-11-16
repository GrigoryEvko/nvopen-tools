// Function: sub_2426FD0
// Address: 0x2426fd0
//
void __fastcall sub_2426FD0(__int64 *a1, __int64 *a2, __int64 a3)
{
  __int64 v4; // r12
  __int64 v5; // r14
  __int64 v6; // rax
  _QWORD *v7; // rcx
  __int64 v8; // rdx
  _QWORD *v9; // rsi
  _QWORD *v10; // rax
  _QWORD *v11; // rax

  *a1 = a3;
  a1[1] = 0;
  a1[2] = 0;
  if ( a3 > 0 )
  {
    v4 = a3;
    while ( 1 )
    {
      v5 = v4;
      v6 = sub_2207800(8 * v4);
      v7 = (_QWORD *)v6;
      if ( v6 )
        break;
      v4 >>= 1;
      if ( !v4 )
        return;
    }
    v8 = *a2;
    v9 = (_QWORD *)(v6 + v5 * 8);
    v10 = (_QWORD *)(v6 + 8);
    *a2 = 0;
    *(v10 - 1) = v8;
    if ( v9 == v10 )
    {
      v11 = v7;
    }
    else
    {
      while ( 1 )
      {
        *v10++ = v8;
        *(v10 - 2) = 0;
        if ( v9 == v10 )
          break;
        v8 = *(v10 - 1);
      }
      v8 = v7[v5 - 1];
      v11 = &v7[v5 - 1];
    }
    *v11 = 0;
    *a2 = v8;
    a1[2] = (__int64)v7;
    a1[1] = v4;
  }
}
