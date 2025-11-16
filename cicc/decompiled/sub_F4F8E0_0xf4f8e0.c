// Function: sub_F4F8E0
// Address: 0xf4f8e0
//
char *__fastcall sub_F4F8E0(char **a1, char *a2, size_t a3, unsigned __int8 a4, __int64 a5)
{
  char *v8; // rdx
  char *result; // rax
  unsigned __int64 v10; // rcx
  char *v11; // r15
  unsigned __int64 v12; // r9
  char *v13; // r13
  __int64 v14; // r8
  size_t v15; // r9
  char *v16; // rdx
  size_t v17; // r9
  char *v18; // rdi
  size_t v19; // r10
  size_t v20; // r10
  char *v21; // [rsp+8h] [rbp-48h]
  __int64 v22; // [rsp+8h] [rbp-48h]
  __int64 v23; // [rsp+10h] [rbp-40h]
  size_t v24; // [rsp+10h] [rbp-40h]
  size_t v25; // [rsp+18h] [rbp-38h]
  size_t v26; // [rsp+18h] [rbp-38h]
  __int64 v27; // [rsp+18h] [rbp-38h]
  char *v28; // [rsp+18h] [rbp-38h]
  size_t v29; // [rsp+18h] [rbp-38h]

  v8 = a1[1];
  result = *a1;
  v10 = (unsigned __int64)a1[2];
  v11 = &v8[(_QWORD)*a1];
  v12 = (unsigned __int64)&v8[a3];
  if ( a2 == v11 )
  {
    if ( v12 > v10 )
    {
      result = (char *)sub_C8D290((__int64)a1, a1 + 3, (__int64)&v8[a3], 1u, a5, v12);
      v8 = a1[1];
      v11 = &(*a1)[(_QWORD)v8];
    }
    if ( a3 )
    {
      result = (char *)memset(v11, a4, a3);
      v8 = a1[1];
    }
    a1[1] = &v8[a3];
  }
  else
  {
    v13 = a2;
    v14 = a2 - result;
    if ( v12 > v10 )
    {
      v27 = a2 - result;
      sub_C8D290((__int64)a1, a1 + 3, (__int64)&v8[a3], 1u, v14, v12);
      result = *a1;
      v14 = v27;
      v8 = a1[1];
      v13 = &(*a1)[v27];
      v11 = &v8[(_QWORD)*a1];
    }
    v15 = (size_t)&v8[-v14];
    if ( a3 <= (unsigned __int64)&v8[-v14] )
    {
      v17 = a3;
      v18 = v11;
      v19 = (size_t)&v8[-a3];
      if ( &v8[a3] > a1[2] )
      {
        v21 = &v8[-a3];
        v23 = v14;
        v28 = result;
        sub_C8D290((__int64)a1, a1 + 3, (__int64)&v8[a3], 1u, v14, a3);
        v8 = a1[1];
        v17 = a3;
        v19 = (size_t)v21;
        v14 = v23;
        result = v28;
        v18 = &(*a1)[(_QWORD)v8];
      }
      if ( v17 )
      {
        v22 = v14;
        v24 = v19;
        v29 = v17;
        result = (char *)memmove(v18, &result[v19], v17);
        v8 = a1[1];
        v14 = v22;
        v19 = v24;
        v17 = v29;
      }
      a1[1] = &v8[v17];
      v20 = v19 - v14;
      if ( v20 )
        result = (char *)memmove(&v11[-v20], v13, v20);
      if ( a3 )
        return (char *)memset(v13, a4, a3);
    }
    else
    {
      v16 = &v8[a3];
      a1[1] = v16;
      if ( v13 != v11 )
      {
        v25 = v15;
        memcpy(&v16[(_QWORD)result - v15], v13, v15);
        v15 = v25;
      }
      if ( v15 )
      {
        v26 = v15;
        memset(v13, a4, v15);
        v15 = v26;
      }
      return (char *)memset(v11, a4, a3 - v15);
    }
  }
  return result;
}
