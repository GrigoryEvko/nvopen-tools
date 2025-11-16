// Function: sub_2652B00
// Address: 0x2652b00
//
void __fastcall sub_2652B00(__int64 a1, char *a2, char *a3, __int64 a4)
{
  __int64 v4; // r15
  __int64 v5; // rbx
  __int64 v6; // r12
  __int64 v7; // rdi
  __int64 v8; // r12
  __int64 v9; // rdx
  __int64 v10; // r8
  char *v11; // r15
  __int64 v12; // rbx
  __int64 v13; // r14
  __int64 v14; // r12
  char *v15; // rdi
  signed __int64 v16; // rax
  __int64 v17; // rdx
  char *v18; // rbx
  __int64 v19; // r8
  char *v20; // r14
  __int64 v21; // rbx
  __int64 v22; // r15
  char *v23; // rdi
  signed __int64 v24; // rax
  __int64 v25; // rbx
  __int64 v28; // [rsp+18h] [rbp-58h]
  __int64 v29; // [rsp+20h] [rbp-50h]
  char *v30; // [rsp+28h] [rbp-48h]
  __int64 v31; // [rsp+38h] [rbp-38h]

  v4 = a4;
  v5 = (__int64)&a2[-a1];
  v30 = &a2[(_QWORD)a3 - a1];
  v28 = 0x8E38E38E38E38E39LL * ((__int64)&a2[-a1] >> 3);
  if ( (__int64)&a2[-a1] <= 432 )
  {
    sub_2651040(a1, (__int64)a2, a4);
  }
  else
  {
    v6 = a1;
    do
    {
      v7 = v6;
      v6 += 504;
      sub_2651040(v7, v6, v4);
    }
    while ( (__int64)&a2[-v6] > 432 );
    sub_2651040(v6, (__int64)a2, v4);
    if ( v5 > 504 )
    {
      v8 = 7;
      while ( 1 )
      {
        v31 = 2 * v8;
        if ( v28 < 2 * v8 )
        {
          v10 = (__int64)a3;
          v16 = v28;
          v18 = (char *)a1;
        }
        else
        {
          v9 = v4;
          v10 = (__int64)a3;
          v29 = v8;
          v11 = (char *)a1;
          v12 = v9;
          v13 = 144 * v8;
          v14 = 72 * v8;
          do
          {
            v15 = v11;
            v11 += v13;
            v10 = sub_2652050(v15, &v11[v14 - v13], &v11[v14 - v13], v11, v10, v12);
            v16 = 0x8E38E38E38E38E39LL * ((a2 - v11) >> 3);
          }
          while ( v31 <= v16 );
          v17 = v12;
          v8 = v29;
          v18 = v11;
          v4 = v17;
        }
        if ( v8 <= v16 )
          v16 = v8;
        v8 *= 4;
        sub_2652050(v18, &v18[72 * v16], &v18[72 * v16], a2, v10, v4);
        v19 = a1;
        if ( v28 < v8 )
          break;
        v20 = a3;
        v21 = v4;
        v22 = 72 * v8;
        do
        {
          v23 = v20;
          v20 += v22;
          v19 = sub_26525C0(v23, &v20[72 * v31 - v22], &v20[72 * v31 - v22], v20, v19, v21);
          v24 = 0x8E38E38E38E38E39LL * ((v30 - v20) >> 3);
        }
        while ( v8 <= v24 );
        v4 = v21;
        if ( v24 > v31 )
          v24 = v31;
        sub_26525C0(v20, &v20[72 * v24], &v20[72 * v24], v30, v19, v21);
        if ( v28 <= v8 )
          return;
      }
      v25 = v31;
      if ( v28 <= v31 )
        v25 = v28;
      sub_26525C0(a3, &a3[72 * v25], &a3[72 * v25], v30, a1, v4);
    }
  }
}
