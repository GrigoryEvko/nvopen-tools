// Function: sub_153D060
// Address: 0x153d060
//
void __fastcall sub_153D060(__int64 a1, __int64 **a2, __int64 **a3, __int64 a4)
{
  __int64 v5; // rbx
  __int64 **v6; // r12
  __int64 v7; // rdi
  __int64 v8; // rbx
  __int64 v9; // r13
  __int64 v10; // r12
  __int64 v11; // r8
  __int64 **v12; // r14
  __int64 v13; // rax
  __int64 v14; // r15
  __int64 v15; // rbx
  __int64 v16; // r13
  __int64 v17; // r15
  __int64 **v18; // rdi
  __int64 v19; // rax
  __int64 v20; // r8
  __int64 **v21; // r14
  __int64 **v22; // rdi
  __int64 v23; // rax
  __int64 v24; // rsi
  __int64 v25; // rsi
  __int64 v26; // [rsp+0h] [rbp-60h]
  __int64 v29; // [rsp+18h] [rbp-48h]
  __int64 **v31; // [rsp+28h] [rbp-38h]

  v5 = (__int64)a2 - a1;
  v29 = ((__int64)a2 - a1) >> 4;
  v31 = (__int64 **)((char *)a2 + (_QWORD)a3 - a1);
  if ( (__int64)a2 - a1 <= 96 )
  {
    sub_153CD00(a1, a2, a4);
  }
  else
  {
    v6 = (__int64 **)a1;
    do
    {
      v7 = (__int64)v6;
      v6 += 14;
      sub_153CD00(v7, v6, a4);
    }
    while ( (char *)a2 - (char *)v6 > 96 );
    sub_153CD00((__int64)v6, a2, a4);
    if ( v5 > 112 )
    {
      v8 = 7;
      v9 = a4;
      while ( 1 )
      {
        v10 = 2 * v8;
        if ( v29 < 2 * v8 )
        {
          v11 = (__int64)a3;
          v19 = v29;
          v12 = (__int64 **)a1;
        }
        else
        {
          v11 = (__int64)a3;
          v12 = (__int64 **)a1;
          v13 = 32 * v8;
          v14 = 16 * v8;
          v26 = v8;
          v15 = v9;
          v16 = v14 - v13;
          v17 = v13;
          do
          {
            v18 = v12;
            v12 = (__int64 **)((char *)v12 + v17);
            v11 = sub_153CBB0(v18, (__int64 **)((char *)v12 + v16), (__int64 **)((char *)v12 + v16), v12, v11, v15);
            v19 = ((char *)a2 - (char *)v12) >> 4;
          }
          while ( v10 <= v19 );
          v9 = v15;
          v8 = v26;
        }
        if ( v8 <= v19 )
          v19 = v8;
        v8 *= 4;
        sub_153CBB0(v12, &v12[2 * v19], &v12[2 * v19], a2, v11, v9);
        v20 = a1;
        if ( v29 < v8 )
          break;
        v21 = a3;
        do
        {
          v22 = v21;
          v21 += 2 * v8;
          v20 = sub_153CF40(
                  v22,
                  (__int64 **)((char *)v21 + 16 * v10 - 16 * v8),
                  (__int64 **)((char *)v21 + 16 * v10 - 16 * v8),
                  v21,
                  v20,
                  v9);
          v23 = ((char *)v31 - (char *)v21) >> 4;
        }
        while ( v8 <= v23 );
        v24 = v10;
        if ( v23 <= v10 )
          v24 = ((char *)v31 - (char *)v21) >> 4;
        sub_153CF40(v21, &v21[2 * v24], &v21[2 * v24], v31, v20, v9);
        if ( v29 <= v8 )
          return;
      }
      v25 = v10;
      if ( v29 <= v10 )
        v25 = v29;
      sub_153CF40(a3, &a3[2 * v25], &a3[2 * v25], v31, a1, v9);
    }
  }
}
