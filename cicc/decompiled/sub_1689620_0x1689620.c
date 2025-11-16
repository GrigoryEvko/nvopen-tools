// Function: sub_1689620
// Address: 0x1689620
//
char *__fastcall sub_1689620(char **a1, char *a2, char *a3, char *a4)
{
  char *v7; // r13
  size_t v8; // r9
  char *v9; // r12
  char *result; // rax
  __int64 v11; // rcx
  size_t v12; // rsi
  char *v13; // r8
  char *v14; // r10
  __int64 v15; // rdx
  size_t v16; // rbx
  unsigned int v17; // edx
  size_t v18; // rbx
  char *v19; // rdi
  size_t v20; // r15
  size_t v21; // r15
  size_t v22; // [rsp+0h] [rbp-50h]
  size_t v23; // [rsp+8h] [rbp-48h]
  char *v24; // [rsp+8h] [rbp-48h]
  size_t v25; // [rsp+10h] [rbp-40h]
  char *v26; // [rsp+10h] [rbp-40h]
  char *v27; // [rsp+10h] [rbp-40h]
  char *v28; // [rsp+18h] [rbp-38h]
  int v29; // [rsp+18h] [rbp-38h]
  char *v30; // [rsp+18h] [rbp-38h]
  size_t v31; // [rsp+18h] [rbp-38h]
  size_t v32; // [rsp+18h] [rbp-38h]
  char *v33; // [rsp+18h] [rbp-38h]
  char *v34; // [rsp+18h] [rbp-38h]

  v7 = a2;
  v8 = a4 - a3;
  v9 = a3;
  result = *a1;
  v11 = *((unsigned int *)a1 + 2);
  v12 = *((unsigned int *)a1 + 3);
  v13 = &(*a1)[v11];
  v14 = (char *)(a2 - *a1);
  if ( v7 == v13 )
  {
    if ( v12 - v11 < v8 )
    {
      v31 = v8;
      result = (char *)sub_16CD150(a1, a1 + 2, v11 + v8, 1);
      v11 = *((unsigned int *)a1 + 2);
      v8 = v31;
      v13 = &(*a1)[v11];
    }
    if ( v9 != a4 )
    {
      v29 = v8;
      result = (char *)memcpy(v13, v9, v8);
      LODWORD(v11) = *((_DWORD *)a1 + 2);
      LODWORD(v8) = v29;
    }
    *((_DWORD *)a1 + 2) = v8 + v11;
  }
  else
  {
    v15 = *((unsigned int *)a1 + 2);
    if ( v11 + v8 > v12 )
    {
      v25 = v8;
      v30 = v14;
      sub_16CD150(a1, a1 + 2, v11 + v8, 1);
      v15 = *((unsigned int *)a1 + 2);
      v14 = v30;
      result = *a1;
      v8 = v25;
      LODWORD(v11) = *((_DWORD *)a1 + 2);
      v16 = v15 - (_QWORD)v30;
      v7 = &v30[(_QWORD)*a1];
      v13 = &(*a1)[v15];
      if ( v15 - (__int64)v30 < v25 )
        goto LABEL_4;
    }
    else
    {
      v16 = v11 - (_QWORD)v14;
      if ( v11 - (__int64)v14 < v8 )
      {
LABEL_4:
        v17 = v8 + v15;
        *((_DWORD *)a1 + 2) = v17;
        if ( v13 != v7 )
        {
          v28 = v13;
          result = (char *)memcpy(&result[v17 - v16], v7, v16);
          v13 = v28;
        }
        if ( v16 )
        {
          for ( result = 0; result != (char *)v16; ++result )
            result[(_QWORD)v7] = result[(_QWORD)v9];
          v9 += v16;
        }
        if ( a4 != v9 )
          return (char *)memcpy(v13, v9, a4 - v9);
        return result;
      }
    }
    v18 = v8;
    v19 = v13;
    v20 = v15 - v8;
    if ( v8 > (unsigned __int64)*((unsigned int *)a1 + 3) - v15 )
    {
      v22 = v8;
      v24 = v13;
      v27 = v14;
      v34 = result;
      sub_16CD150(a1, a1 + 2, v8 + v15, 1);
      v11 = *((unsigned int *)a1 + 2);
      v8 = v22;
      v13 = v24;
      v14 = v27;
      result = v34;
      v19 = &(*a1)[v11];
    }
    if ( v18 )
    {
      v23 = v8;
      v26 = v13;
      v33 = v14;
      result = (char *)memmove(v19, &result[v20], v18);
      LODWORD(v11) = *((_DWORD *)a1 + 2);
      v8 = v23;
      v13 = v26;
      v14 = v33;
    }
    *((_DWORD *)a1 + 2) = v18 + v11;
    v21 = v20 - (_QWORD)v14;
    if ( v21 )
    {
      v32 = v8;
      result = (char *)memmove(&v13[-v21], v7, v21);
      v8 = v32;
    }
    if ( v8 )
      return (char *)memmove(v7, v9, v8);
  }
  return result;
}
