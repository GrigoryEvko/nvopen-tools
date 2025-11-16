// Function: sub_2664D00
// Address: 0x2664d00
//
void __fastcall sub_2664D00(char *a1, char *a2, char *a3)
{
  char *v3; // r14
  char *v4; // r13
  char *v5; // r12
  char *v6; // rdi
  __int64 v7; // r11
  __int64 v8; // rbx
  char *v9; // r8
  char *v10; // r14
  char *v11; // rdi
  __int64 v12; // rax
  __int64 v13; // r11
  __int64 v14; // r11
  _QWORD *v15; // r8
  char *v16; // rax
  char *v17; // r13
  char *v18; // r14
  __int64 v19; // r11
  char *v20; // rdi
  __int64 v21; // rax
  char *v22; // rcx
  char *v23; // r14
  __int64 v24; // rsi
  __int64 v25; // [rsp+8h] [rbp-58h]
  char *v26; // [rsp+10h] [rbp-50h]
  __int64 v29; // [rsp+28h] [rbp-38h]

  v3 = &a3[a2 - a1];
  v4 = a2;
  v29 = (a2 - a1) >> 4;
  if ( a2 - a1 <= 96 )
  {
    sub_2664AB0(a1, a2);
  }
  else
  {
    v5 = a1;
    do
    {
      v6 = v5;
      v5 += 112;
      sub_2664AB0(v6, v5);
    }
    while ( a2 - v5 > 96 );
    sub_2664AB0(v5, a2);
    if ( v7 > 112 )
    {
      v26 = v3;
      v8 = 7;
      while ( 1 )
      {
        if ( v29 < 2 * v8 )
        {
          v9 = a3;
          v12 = v29;
          v10 = a1;
        }
        else
        {
          v9 = a3;
          v10 = a1;
          do
          {
            v11 = v10;
            v10 += 32 * v8;
            v9 = sub_2664840(v11, &v10[-16 * v8], &v10[-16 * v8], v10, v9);
            v12 = (v4 - v10) >> 4;
          }
          while ( v13 <= v12 );
        }
        if ( v8 <= v12 )
          v12 = v8;
        v8 *= 4;
        sub_2664840(v10, &v10[16 * v12], &v10[16 * v12], v4, v9);
        v15 = a1;
        if ( v29 < v8 )
          break;
        v16 = v4;
        v25 = v14;
        v17 = a3;
        v18 = v16;
        v19 = 16 * v14 - 16 * v8;
        do
        {
          v20 = v17;
          v17 += 16 * v8;
          v15 = sub_2664C20(v20, &v17[v19], &v17[v19], v17, v15);
          v21 = (v26 - v17) >> 4;
        }
        while ( v8 <= v21 );
        v22 = v18;
        v23 = v17;
        v4 = v22;
        v24 = v25;
        if ( v21 <= v25 )
          v24 = v21;
        sub_2664C20(v23, &v23[16 * v24], &v23[16 * v24], v26, v15);
        if ( v29 <= v8 )
          return;
      }
      if ( v29 <= v14 )
        v14 = v29;
      sub_2664C20(a3, &a3[16 * v14], &a3[16 * v14], v26, a1);
    }
  }
}
