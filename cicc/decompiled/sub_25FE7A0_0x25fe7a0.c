// Function: sub_25FE7A0
// Address: 0x25fe7a0
//
void __fastcall sub_25FE7A0(__int64 *a1, __int64 *a2, __int64 a3)
{
  __int64 v3; // r14
  __int64 v6; // r12
  __int64 v7; // rax
  _QWORD *v8; // rdx
  __int64 v9; // rdi
  __int64 v10; // rsi
  _QWORD *v11; // r8
  _QWORD *v12; // rax
  __int64 v13; // rcx
  unsigned __int64 v14; // r12
  _QWORD *v15; // rcx
  _QWORD *v16; // rax

  v3 = 0x555555555555555LL;
  if ( a3 <= 0x555555555555555LL )
    v3 = a3;
  *a1 = a3;
  a1[1] = 0;
  a1[2] = 0;
  if ( a3 > 0 )
  {
    while ( 1 )
    {
      v6 = 24 * v3;
      v7 = sub_2207800(24 * v3);
      v8 = (_QWORD *)v7;
      if ( v7 )
        break;
      v3 >>= 1;
      if ( !v3 )
        return;
    }
    v9 = *a2;
    v10 = a2[1];
    v11 = (_QWORD *)(v7 + v6);
    v12 = (_QWORD *)(v7 + 24);
    v13 = a2[2];
    a2[1] = 0;
    *(v12 - 3) = v9;
    *(v12 - 2) = v10;
    *(v12 - 1) = v13;
    a2[2] = 0;
    *a2 = 0;
    if ( v11 == v12 )
    {
      v16 = v8;
    }
    else
    {
      while ( 1 )
      {
        *v12 = v9;
        v12 += 3;
        *(v12 - 2) = v10;
        *(v12 - 1) = v13;
        *(v12 - 4) = 0;
        *(v12 - 5) = 0;
        *(v12 - 6) = 0;
        if ( v11 == v12 )
          break;
        v9 = *(v12 - 3);
        v10 = *(v12 - 2);
        v13 = *(v12 - 1);
      }
      v14 = (0xAAAAAAAAAAAAAABLL * ((unsigned __int64)(v6 - 48) >> 3)) & 0x1FFFFFFFFFFFFFFFLL;
      v15 = &v8[3 * v14];
      v16 = &v8[3 * v14 + 3];
      v9 = v15[3];
      v10 = v15[4];
      v13 = v15[5];
    }
    *a2 = v9;
    a2[1] = v10;
    a2[2] = v13;
    *v16 = 0;
    v16[1] = 0;
    v16[2] = 0;
    a1[2] = (__int64)v8;
    a1[1] = v3;
  }
}
