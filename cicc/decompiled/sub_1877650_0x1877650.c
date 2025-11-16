// Function: sub_1877650
// Address: 0x1877650
//
void __fastcall sub_1877650(__int64 a1, char *a2, __int64 a3)
{
  __int64 v3; // rbx
  char *v4; // r12
  __int64 v5; // rdi
  __int64 v6; // r12
  __int64 v7; // rbx
  __int64 v8; // r8
  __int64 v9; // r15
  __int64 v10; // r14
  __int64 v11; // r12
  __int64 v12; // rdi
  signed __int64 v13; // rax
  __int64 v14; // r8
  __int64 v15; // r14
  __int64 v16; // r15
  __int64 v17; // rbx
  __int64 v18; // rdi
  signed __int64 v19; // rax
  __int64 v20; // [rsp+0h] [rbp-60h]
  signed __int64 v21; // [rsp+0h] [rbp-60h]
  __int64 v24; // [rsp+18h] [rbp-48h]
  __int64 v25; // [rsp+20h] [rbp-40h]

  v3 = (__int64)&a2[-a1];
  v25 = (__int64)&a2[a3 - a1];
  v24 = 0xAAAAAAAAAAAAAAABLL * ((__int64)&a2[-a1] >> 4);
  if ( (__int64)&a2[-a1] <= 288 )
  {
    sub_18772A0(a1, a2);
  }
  else
  {
    v4 = (char *)a1;
    do
    {
      v5 = (__int64)v4;
      v4 += 336;
      sub_18772A0(v5, v4);
    }
    while ( a2 - v4 > 288 );
    sub_18772A0((__int64)v4, a2);
    if ( v3 > 336 )
    {
      v6 = 7;
      while ( 1 )
      {
        v7 = 2 * v6;
        if ( v24 < 2 * v6 )
        {
          v8 = a3;
          v13 = v24;
          v9 = a1;
        }
        else
        {
          v8 = a3;
          v9 = a1;
          v20 = v6;
          v10 = 96 * v6;
          v11 = 48 * v6;
          do
          {
            v12 = v9;
            v9 += v10;
            v8 = sub_1876380(v12, v9 + v11 - v10, v9 + v11 - v10, v9, v8);
            v13 = 0xAAAAAAAAAAAAAAABLL * ((__int64)&a2[-v9] >> 4);
          }
          while ( v7 <= v13 );
          v6 = v20;
        }
        if ( v6 <= v13 )
          v13 = v6;
        v6 *= 4;
        sub_1876380(v9, v9 + 48 * v13, v9 + 48 * v13, (__int64)a2, v8);
        v14 = a1;
        if ( v24 < v6 )
          break;
        v21 = v7;
        v15 = a3;
        v16 = 48 * v6;
        v17 = 48 * v7;
        do
        {
          v18 = v15;
          v15 += v16;
          v14 = sub_1876700(v18, v15 + v17 - v16, v15 + v17 - v16, v15, v14);
          v19 = 0xAAAAAAAAAAAAAAABLL * ((v25 - v15) >> 4);
        }
        while ( v6 <= v19 );
        if ( v19 > v21 )
          v19 = v21;
        sub_1876700(v15, v15 + 48 * v19, v15 + 48 * v19, v25, v14);
        if ( v24 <= v6 )
          return;
      }
      if ( v24 <= v7 )
        v7 = v24;
      sub_1876700(a3, a3 + 48 * v7, a3 + 48 * v7, v25, a1);
    }
  }
}
