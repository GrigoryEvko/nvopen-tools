// Function: sub_22ADDC0
// Address: 0x22addc0
//
void __fastcall sub_22ADDC0(char *a1, char *a2, unsigned int *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rbx
  __int64 v9; // rdx
  __int64 v10; // rcx
  char *v11; // r12
  char *v12; // rdi
  __int64 v13; // r9
  __int64 v14; // r12
  __int64 v15; // rbx
  char *v16; // r8
  char *v17; // r15
  __int64 v18; // r10
  __int64 v19; // r12
  __int64 v20; // r14
  char *v21; // rdi
  signed __int64 v22; // rax
  __int64 v23; // r9
  _DWORD *v24; // r8
  unsigned int *v25; // r14
  __int64 v26; // r15
  __int64 v27; // rbx
  unsigned int *v28; // rdi
  signed __int64 v29; // rax
  __int64 v30; // [rsp+0h] [rbp-60h]
  signed __int64 v31; // [rsp+0h] [rbp-60h]
  __int64 v34; // [rsp+18h] [rbp-48h]
  __int64 v35; // [rsp+20h] [rbp-40h]
  __int64 v36; // [rsp+28h] [rbp-38h]

  v8 = a2 - a1;
  v9 = (a2 - a1) >> 3;
  v10 = 0x8E38E38E38E38E39LL * v9;
  v36 = (__int64)a3 + a2 - a1;
  v34 = 0x8E38E38E38E38E39LL * v9;
  if ( a2 - a1 <= 432 )
  {
    sub_22AD930(a1, a2, v9, v10, a5, a6);
  }
  else
  {
    v11 = a1;
    do
    {
      v12 = v11;
      v11 += 504;
      sub_22AD930(v12, v11, v9, v10, a5, a6);
    }
    while ( a2 - v11 > 432 );
    sub_22AD930(v11, a2, v9, v10, a5, a6);
    if ( v8 > 504 )
    {
      v35 = (__int64)a2;
      v14 = 7;
      while ( 1 )
      {
        v15 = 2 * v14;
        if ( v34 < 2 * v14 )
        {
          v16 = (char *)a3;
          v22 = v34;
          v17 = a1;
        }
        else
        {
          v16 = (char *)a3;
          v17 = a1;
          v30 = v14;
          v18 = 18 * v14;
          v19 = 72 * v14;
          v20 = 8 * v18;
          do
          {
            v21 = v17;
            v17 += v20;
            v16 = sub_22ADC50(v21, &v17[v19 - v20], (__int64)&v17[v19 - v20], (__int64)v17, (__int64)v16, v13);
            v22 = 0x8E38E38E38E38E39LL * ((v35 - (__int64)v17) >> 3);
          }
          while ( v15 <= v22 );
          v14 = v30;
        }
        if ( v14 <= v22 )
          v22 = v14;
        v14 *= 4;
        sub_22ADC50(v17, &v17[72 * v22], (__int64)&v17[72 * v22], v35, (__int64)v16, v13);
        v24 = a1;
        if ( v34 < v14 )
          break;
        v31 = v15;
        v25 = a3;
        v26 = 72 * v14;
        v27 = 72 * v15;
        do
        {
          v28 = v25;
          v25 = (unsigned int *)((char *)v25 + v26);
          v24 = sub_22ADAF0(
                  v28,
                  (unsigned int *)((char *)v25 + v27 - v26),
                  (unsigned int *)((char *)v25 + v27 - v26),
                  (__int64)v25,
                  (__int64)v24,
                  v23);
          v29 = 0x8E38E38E38E38E39LL * ((v36 - (__int64)v25) >> 3);
        }
        while ( v14 <= v29 );
        if ( v29 > v31 )
          v29 = v31;
        sub_22ADAF0(v25, &v25[18 * v29], &v25[18 * v29], v36, (__int64)v24, v23);
        if ( v34 <= v14 )
          return;
      }
      if ( v34 <= v15 )
        v15 = v34;
      sub_22ADAF0(a3, &a3[18 * v15], &a3[18 * v15], v36, (__int64)a1, v23);
    }
  }
}
