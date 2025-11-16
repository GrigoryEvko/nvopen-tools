// Function: sub_29BF380
// Address: 0x29bf380
//
void __fastcall sub_29BF380(char *src, char *a2, char *a3, _QWORD *a4)
{
  __int64 v5; // rbx
  char *v6; // r12
  char *v7; // rdi
  __int64 v8; // rbx
  _QWORD *v9; // r13
  __int64 v10; // r12
  char *v11; // r8
  char *v12; // r14
  __int64 v13; // rax
  __int64 v14; // r15
  _QWORD *v15; // rbx
  __int64 v16; // r13
  __int64 v17; // r15
  char *v18; // rdi
  __int64 v19; // rax
  char *v20; // r8
  char *v21; // r14
  char *v22; // rdi
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // [rsp+0h] [rbp-60h]
  __int64 v28; // [rsp+18h] [rbp-48h]
  char *v29; // [rsp+28h] [rbp-38h]

  v5 = a2 - src;
  v28 = (a2 - src) >> 3;
  v29 = &a3[a2 - src];
  if ( a2 - src <= 48 )
  {
    sub_29BF2A0(src, a2, a4);
  }
  else
  {
    v6 = src;
    do
    {
      v7 = v6;
      v6 += 56;
      sub_29BF2A0(v7, v6, a4);
    }
    while ( a2 - v6 > 48 );
    sub_29BF2A0(v6, a2, a4);
    if ( v5 > 56 )
    {
      v8 = 7;
      v9 = a4;
      while ( 1 )
      {
        v10 = 2 * v8;
        if ( v28 < 2 * v8 )
        {
          v11 = a3;
          v19 = v28;
          v12 = src;
        }
        else
        {
          v11 = a3;
          v12 = src;
          v25 = v8;
          v13 = 16 * v8;
          v14 = 8 * v8;
          v15 = v9;
          v16 = v14 - v13;
          v17 = v13;
          do
          {
            v18 = v12;
            v12 += v17;
            v11 = sub_29BF110(v18, &v12[v16], &v12[v16], v12, v11, v15);
            v19 = (a2 - v12) >> 3;
          }
          while ( v10 <= v19 );
          v9 = v15;
          v8 = v25;
        }
        if ( v8 <= v19 )
          v19 = v8;
        v8 *= 4;
        sub_29BF110(v12, &v12[8 * v19], &v12[8 * v19], a2, v11, v9);
        v20 = src;
        if ( v28 < v8 )
          break;
        v21 = a3;
        do
        {
          v22 = v21;
          v21 += 8 * v8;
          v20 = sub_29BF1E0(v22, &v21[8 * v10 - 8 * v8], &v21[8 * v10 - 8 * v8], v21, v20, v9);
          v23 = (v29 - v21) >> 3;
        }
        while ( v8 <= v23 );
        if ( v23 > v10 )
          v23 = v10;
        sub_29BF1E0(v21, &v21[8 * v23], &v21[8 * v23], v29, v20, v9);
        if ( v28 <= v8 )
          return;
      }
      v24 = v10;
      if ( v28 <= v10 )
        v24 = v28;
      sub_29BF1E0(a3, &a3[8 * v24], &a3[8 * v24], v29, src, v9);
    }
  }
}
