// Function: sub_ED6EC0
// Address: 0xed6ec0
//
char *__fastcall sub_ED6EC0(_QWORD *a1, char *a2, char *a3, char *a4)
{
  char *v6; // r12
  size_t v7; // r9
  char *result; // rax
  __int64 v9; // rdx
  size_t v10; // rcx
  char *v11; // r8
  char *v12; // r10
  size_t v13; // r11
  char *v14; // r13
  size_t v15; // rbx
  size_t v16; // rdx
  size_t v17; // rbx
  void *v18; // rdi
  size_t v19; // r15
  size_t v20; // r15
  size_t v21; // [rsp+0h] [rbp-50h]
  size_t v22; // [rsp+8h] [rbp-48h]
  char *v23; // [rsp+8h] [rbp-48h]
  size_t v24; // [rsp+10h] [rbp-40h]
  char *v25; // [rsp+10h] [rbp-40h]
  char *v26; // [rsp+10h] [rbp-40h]
  char *v27; // [rsp+18h] [rbp-38h]
  size_t v28; // [rsp+18h] [rbp-38h]
  char *v29; // [rsp+18h] [rbp-38h]
  size_t v30; // [rsp+18h] [rbp-38h]
  size_t v31; // [rsp+18h] [rbp-38h]
  char *v32; // [rsp+18h] [rbp-38h]
  char *v33; // [rsp+18h] [rbp-38h]

  v6 = a3;
  v7 = a4 - a3;
  result = (char *)*a1;
  v9 = a1[1];
  v10 = a1[2];
  v11 = (char *)(*a1 + v9);
  v12 = &a2[-*a1];
  v13 = v9 + v7;
  if ( a2 == v11 )
  {
    if ( v13 > v10 )
    {
      v30 = v7;
      result = (char *)sub_C8D290((__int64)a1, a1 + 3, v9 + v7, 1u, (__int64)v11, v7);
      v9 = a1[1];
      v7 = v30;
      v11 = (char *)(v9 + *a1);
    }
    if ( v6 != a4 )
    {
      v28 = v7;
      result = (char *)memcpy(v11, v6, v7);
      v9 = a1[1];
      v7 = v28;
    }
    a1[1] = v7 + v9;
  }
  else
  {
    v14 = a2;
    if ( v13 > v10 )
    {
      v24 = v7;
      v29 = &a2[-*a1];
      sub_C8D290((__int64)a1, a1 + 3, v9 + v7, 1u, (__int64)v11, v7);
      v9 = a1[1];
      v12 = v29;
      result = (char *)*a1;
      v7 = v24;
      v15 = v9 - (_QWORD)v29;
      v14 = &v29[*a1];
      v11 = (char *)(*a1 + v9);
      if ( v9 - (__int64)v29 < v24 )
        goto LABEL_4;
    }
    else
    {
      v15 = v9 - (_QWORD)v12;
      if ( v9 - (__int64)v12 < v7 )
      {
LABEL_4:
        v16 = v7 + v9;
        a1[1] = v16;
        if ( v11 != v14 )
        {
          v27 = v11;
          result = (char *)memcpy(&result[v16 - v15], v14, v15);
          v11 = v27;
        }
        if ( v15 )
        {
          for ( result = 0; result != (char *)v15; ++result )
            result[(_QWORD)v14] = result[(_QWORD)v6];
          v6 += v15;
        }
        if ( a4 != v6 )
          return (char *)memcpy(v11, v6, a4 - v6);
        return result;
      }
    }
    v17 = v7;
    v18 = v11;
    v19 = v9 - v7;
    if ( v9 + v7 > a1[2] )
    {
      v21 = v7;
      v23 = v11;
      v26 = v12;
      v33 = result;
      sub_C8D290((__int64)a1, a1 + 3, v9 + v7, 1u, (__int64)v11, v7);
      v9 = a1[1];
      v7 = v21;
      v11 = v23;
      v12 = v26;
      result = v33;
      v18 = (void *)(v9 + *a1);
    }
    if ( v17 )
    {
      v22 = v7;
      v25 = v11;
      v32 = v12;
      result = (char *)memmove(v18, &result[v19], v17);
      v9 = a1[1];
      v7 = v22;
      v11 = v25;
      v12 = v32;
    }
    a1[1] = v17 + v9;
    v20 = v19 - (_QWORD)v12;
    if ( v20 )
    {
      v31 = v7;
      result = (char *)memmove(&v11[-v20], v14, v20);
      v7 = v31;
    }
    if ( v7 )
      return (char *)memmove(v14, v6, v7);
  }
  return result;
}
