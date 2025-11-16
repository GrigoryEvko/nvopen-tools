// Function: sub_2B80AE0
// Address: 0x2b80ae0
//
__int64 __fastcall sub_2B80AE0(__int64 a1, char **a2, char **a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  char **v7; // r14
  __int64 v8; // r13
  __int64 v9; // r12
  char **v10; // rbx
  __int64 v12; // rcx
  __int64 v13; // r15
  __int64 v14; // rdi
  __int64 v15; // rax
  __int64 v16; // r15
  __int64 v17; // r14
  __int64 v18; // r13
  char **v19; // rsi
  __int64 v20; // rdi
  __int64 v21; // r13
  __int64 v22; // rcx
  __int64 v23; // r15
  __int64 v24; // rdi
  __int64 v25; // rax
  __int64 v26; // r15
  char **v27; // rsi
  __int64 v28; // rdi
  __int64 v29; // r14
  __int64 v30; // r12
  __int64 v31; // r12
  __int64 v32; // [rsp+8h] [rbp-58h]
  __int64 v33; // [rsp+8h] [rbp-58h]
  __int64 v34; // [rsp+10h] [rbp-50h]
  __int64 v35; // [rsp+10h] [rbp-50h]
  __int64 v36; // [rsp+18h] [rbp-48h]
  __int64 v37; // [rsp+18h] [rbp-48h]
  __int64 v38; // [rsp+20h] [rbp-40h]
  __int64 v39; // [rsp+20h] [rbp-40h]
  char **v40; // [rsp+28h] [rbp-38h]
  __int64 v41; // [rsp+28h] [rbp-38h]
  char **v42; // [rsp+28h] [rbp-38h]
  __int64 v43; // [rsp+28h] [rbp-38h]

  v7 = a2;
  v8 = (__int64)a3;
  v9 = a1;
  v10 = (char **)a6;
  if ( a4 <= a5 || a5 > a7 )
  {
    if ( a4 > a7 )
      return sub_2B11A30(a1, a2, a3, a4, a5, a6);
    if ( !a4 )
      return v8;
    v33 = (__int64)a2 - a1;
    v37 = ((__int64)a2 - a1) >> 6;
    v35 = (char *)a3 - (char *)a2;
    v22 = ((char *)a3 - (char *)a2) >> 6;
    v39 = v22;
    if ( (__int64)a2 - a1 <= 0 )
    {
      if ( v35 <= 0 )
        return v8;
      v43 = 0;
      v26 = 0;
    }
    else
    {
      a3 = (char **)a1;
      v23 = a6;
      do
      {
        v24 = v23;
        v42 = a3;
        v23 += 64;
        sub_2B0F6D0(v24, a3, (__int64)a3, v22, a5, a6);
        a3 = v42 + 8;
        --v37;
      }
      while ( v37 );
      v22 = v33;
      v25 = 64;
      if ( v33 > 0 )
        v25 = v33;
      v10 = (char **)((char *)v10 + v25);
      v43 = v25;
      v26 = v25 >> 6;
      if ( v35 <= 0 )
      {
LABEL_29:
        if ( v43 > 0 )
        {
          v29 = v8;
          v30 = v26;
          do
          {
            v10 -= 8;
            v29 -= 64;
            sub_2B0F6D0(v29, v10, (__int64)a3, v22, a5, a6);
            --v30;
          }
          while ( v30 );
          v31 = -64;
          if ( v26 > 0 )
            v31 = -64 * v26;
          return v8 + v31;
        }
        return v8;
      }
    }
    do
    {
      v27 = v7;
      v28 = v9;
      v7 += 8;
      v9 += 64;
      sub_2B0F6D0(v28, v27, (__int64)a3, v22, a5, a6);
      --v39;
    }
    while ( v39 );
    goto LABEL_29;
  }
  if ( !a5 )
    return v9;
  v32 = (char *)a3 - (char *)a2;
  v36 = ((char *)a3 - (char *)a2) >> 6;
  v34 = (__int64)a2 - a1;
  v12 = ((__int64)a2 - a1) >> 6;
  v38 = v12;
  if ( (char *)a3 - (char *)a2 <= 0 )
  {
    if ( v34 <= 0 )
      return v9;
    v41 = 0;
    v16 = 0;
  }
  else
  {
    a3 = a2;
    v13 = a6;
    do
    {
      v14 = v13;
      v40 = a3;
      v13 += 64;
      sub_2B0F6D0(v14, a3, (__int64)a3, v12, a5, a6);
      a3 = v40 + 8;
      --v36;
    }
    while ( v36 );
    v12 = v32;
    v15 = 64;
    if ( v32 > 0 )
      v15 = v32;
    v41 = v15;
    v16 = v15 >> 6;
    if ( v34 <= 0 )
      goto LABEL_15;
  }
  do
  {
    v7 -= 8;
    v8 -= 64;
    sub_2B0F6D0(v8, v7, (__int64)a3, v12, a5, a6);
    --v38;
  }
  while ( v38 );
LABEL_15:
  if ( v41 <= 0 )
    return v9;
  v17 = v9;
  v18 = v16;
  do
  {
    v19 = v10;
    v20 = v17;
    v10 += 8;
    v17 += 64;
    sub_2B0F6D0(v20, v19, (__int64)a3, v12, a5, a6);
    --v18;
  }
  while ( v18 );
  v21 = v16 << 6;
  if ( v16 <= 0 )
    v21 = 64;
  return v9 + v21;
}
