// Function: sub_2C71050
// Address: 0x2c71050
//
signed __int64 __fastcall sub_2C71050(__int64 **a1, __int64 **a2, __int64 a3)
{
  signed __int64 result; // rax
  __int64 *v4; // rax
  __int64 *v5; // r15
  const char *v6; // rax
  size_t v7; // rdx
  size_t v8; // rbx
  const char *v9; // rax
  size_t v10; // rdx
  size_t v11; // r15
  bool v12; // cc
  size_t v13; // rdx
  int v14; // eax
  __int64 **v15; // r15
  __int64 *v16; // rax
  __int64 *v17; // r13
  const char *v18; // rax
  size_t v19; // rdx
  size_t v20; // rbx
  const char *v21; // r14
  const char *v22; // rax
  size_t v23; // rdx
  size_t v24; // r13
  int v25; // eax
  __int64 v26; // r13
  __int64 i; // rbx
  __int64 **v28; // r15
  __int64 *v29; // rcx
  __int64 v30; // rbx
  __int64 **v31; // [rsp+10h] [rbp-60h]
  __int64 v32; // [rsp+18h] [rbp-58h]
  __int64 **v33; // [rsp+20h] [rbp-50h]
  const char *s2; // [rsp+28h] [rbp-48h]
  __int64 **v35; // [rsp+30h] [rbp-40h]
  __int64 **v36; // [rsp+38h] [rbp-38h]

  result = (char *)a2 - (char *)a1;
  v32 = a3;
  v31 = a2;
  if ( (char *)a2 - (char *)a1 <= 128 )
    return result;
  if ( !a3 )
  {
    v33 = a2;
    goto LABEL_28;
  }
  while ( 2 )
  {
    --v32;
    sub_2C6F8B0(
      a1,
      a1 + 1,
      &a1[(__int64)(v31 - a1 + ((unsigned __int64)((char *)v31 - (char *)a1) >> 63)) >> 1],
      v31 - 1);
    v35 = a1 + 1;
    v36 = v31;
    while ( 1 )
    {
      v5 = *v35;
      v33 = v35;
      v6 = sub_BD5D20(**a1);
      v8 = v7;
      s2 = v6;
      v9 = sub_BD5D20(*v5);
      v11 = v10;
      v12 = v10 <= v8;
      v13 = v8;
      if ( v12 )
        v13 = v11;
      if ( !v13 )
        break;
      v14 = memcmp(v9, s2, v13);
      if ( !v14 )
        break;
      if ( v14 >= 0 )
        goto LABEL_15;
LABEL_8:
      ++v35;
    }
    if ( v11 != v8 && v11 < v8 )
      goto LABEL_8;
LABEL_15:
    v15 = v36;
    do
    {
      while ( 1 )
      {
        v16 = *(v15 - 1);
        v17 = *a1;
        v36 = --v15;
        v18 = sub_BD5D20(*v16);
        v20 = v19;
        v21 = v18;
        v22 = sub_BD5D20(*v17);
        v24 = v23;
        if ( v20 <= v23 )
          v23 = v20;
        if ( v23 )
        {
          v25 = memcmp(v22, v21, v23);
          if ( v25 )
            break;
        }
        if ( v20 == v24 || v20 <= v24 )
        {
          if ( v35 >= v15 )
            goto LABEL_22;
LABEL_7:
          v4 = *v35;
          *v35 = *v15;
          *v15 = v4;
          goto LABEL_8;
        }
      }
    }
    while ( v25 < 0 );
    if ( v35 < v15 )
      goto LABEL_7;
LABEL_22:
    sub_2C71050(v35, v31, v32);
    result = (char *)v35 - (char *)a1;
    if ( (char *)v35 - (char *)a1 > 128 )
    {
      if ( v32 )
      {
        v31 = v35;
        continue;
      }
LABEL_28:
      v26 = result >> 3;
      for ( i = ((result >> 3) - 2) >> 1; ; --i )
      {
        sub_2C70E10((__int64)a1, i, v26, a1[i]);
        if ( !i )
          break;
      }
      v28 = v33 - 1;
      do
      {
        v29 = *v28;
        v30 = (char *)v28-- - (char *)a1;
        v28[1] = *a1;
        result = (signed __int64)sub_2C70E10((__int64)a1, 0, v30 >> 3, v29);
      }
      while ( v30 > 8 );
    }
    return result;
  }
}
