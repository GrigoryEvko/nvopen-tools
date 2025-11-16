// Function: sub_3724910
// Address: 0x3724910
//
char *__fastcall sub_3724910(char *a1, char *a2, __int64 a3, __int64 a4, __int64 a5, char *a6, char *a7)
{
  __int64 v7; // r14
  char *v8; // r13
  char *v9; // r12
  char *result; // rax
  char *v11; // rcx
  signed __int64 v12; // rbx
  char *v13; // rsi
  char *v14; // rdx
  __int64 v15; // rbx
  size_t v16; // rbx
  char *v17; // r15
  _QWORD *i; // r14
  __int64 v19; // rax
  __int64 v20; // r10
  __int64 v21; // r11
  __int64 v22; // r8
  char *v23; // r15
  __int64 v24; // [rsp+10h] [rbp-60h]
  __int64 v25; // [rsp+18h] [rbp-58h]
  __int64 v26; // [rsp+20h] [rbp-50h]
  __int64 v27; // [rsp+20h] [rbp-50h]
  char *src; // [rsp+28h] [rbp-48h]
  char *v29; // [rsp+30h] [rbp-40h]
  __int64 v30; // [rsp+38h] [rbp-38h]

  while ( 1 )
  {
    v7 = a3;
    v8 = a6;
    v9 = a1;
    result = a7;
    if ( a5 <= (__int64)a7 )
      result = (char *)a5;
    if ( (__int64)result >= a4 )
    {
      v11 = a2;
      v12 = a2 - a1;
      if ( a2 != a1 )
      {
        result = (char *)memmove(a6, a1, a2 - a1);
        v11 = a2;
      }
      v13 = &v8[v12];
      if ( v8 != &v8[v12] )
      {
        while ( (char *)v7 != v11 )
        {
          v14 = *(char **)v8;
          result = *(char **)v11;
          if ( *(_DWORD *)(*(_QWORD *)v11 + 8LL) < *(_DWORD *)(*(_QWORD *)v8 + 8LL) )
          {
            v11 += 8;
          }
          else
          {
            v8 += 8;
            result = v14;
          }
          *(_QWORD *)v9 = result;
          v9 += 8;
          if ( v13 == v8 )
            return result;
        }
        return (char *)memmove(v9, v8, v13 - v8);
      }
      return result;
    }
    v15 = a5;
    if ( a5 <= (__int64)a7 )
      break;
    if ( a5 < a4 )
    {
      v30 = a4 / 2;
      src = &a1[8 * (a4 / 2)];
      v29 = (char *)sub_37222B0((__int64)a2, a3, (__int64)src);
      v22 = (v29 - a2) >> 3;
    }
    else
    {
      v26 = a5 / 2;
      v29 = &a2[8 * (a5 / 2)];
      v19 = sub_3722300((__int64)a1, (__int64)a2, (__int64)v29);
      v22 = v26;
      src = (char *)v19;
      v30 = (v19 - (__int64)a1) >> 3;
    }
    v24 = v20 - v30;
    v25 = v21;
    v27 = v22;
    v23 = sub_37247F0(src, a2, v29, v20 - v30, v22, v8, v21);
    sub_3724910((_DWORD)a1, (_DWORD)src, (_DWORD)v23, v30, v27, (_DWORD)v8, v25);
    a6 = v8;
    a2 = v29;
    a1 = v23;
    a7 = (char *)v25;
    a5 = v15 - v27;
    a3 = v7;
    a4 = v24;
  }
  v16 = a3 - (_QWORD)a2;
  if ( (char *)a3 != a2 )
    memmove(a6, a2, a3 - (_QWORD)a2);
  result = &v8[v16];
  if ( a2 != a1 )
  {
    if ( v8 == result )
      return result;
    v17 = a2 - 8;
    result -= 8;
    for ( i = (_QWORD *)(v7 - 8); ; --i )
    {
      if ( *(_DWORD *)(*(_QWORD *)result + 8LL) < *(_DWORD *)(*(_QWORD *)v17 + 8LL) )
      {
        *i = *(_QWORD *)v17;
        if ( a1 == v17 )
        {
          if ( v8 != result + 8 )
            return (char *)memmove((char *)i - (result + 8 - v8), v8, result + 8 - v8);
          return result;
        }
        v17 -= 8;
      }
      else
      {
        *i = *(_QWORD *)result;
        if ( v8 == result )
          return result;
        result -= 8;
      }
    }
  }
  if ( v8 != result )
    return (char *)memmove((void *)(v7 - v16), v8, v16);
  return result;
}
