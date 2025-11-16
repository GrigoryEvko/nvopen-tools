// Function: sub_2609450
// Address: 0x2609450
//
__int64 __fastcall sub_2609450(char *a1, char *a2, __int64 a3, __int64 a4, __int64 a5, char *a6, __int64 a7)
{
  unsigned int *v7; // r14
  char *v8; // r13
  char *v9; // r12
  __int64 result; // rax
  unsigned int *v11; // rcx
  signed __int64 v12; // rbx
  char *v13; // rsi
  unsigned int v14; // edx
  char *v15; // r15
  __int64 v16; // rbx
  __int64 v17; // rbx
  char *v18; // rsi
  unsigned int *i; // rdx
  char *v20; // r15
  char *v21; // rsi
  unsigned int *v22; // rax
  __int64 v23; // r10
  __int64 v24; // r11
  __int64 v25; // r8
  char *v26; // r15
  __int64 v27; // [rsp+10h] [rbp-60h]
  __int64 v28; // [rsp+18h] [rbp-58h]
  __int64 v29; // [rsp+20h] [rbp-50h]
  __int64 v30; // [rsp+20h] [rbp-50h]
  unsigned int *src; // [rsp+28h] [rbp-48h]
  unsigned int *v32; // [rsp+30h] [rbp-40h]
  __int64 v33; // [rsp+38h] [rbp-38h]

  while ( 1 )
  {
    v7 = (unsigned int *)a3;
    v8 = a6;
    v9 = a1;
    result = a7;
    if ( a5 <= a7 )
      result = a5;
    if ( result >= a4 )
    {
      v11 = (unsigned int *)a2;
      v12 = a2 - a1;
      if ( a2 != a1 )
      {
        result = (__int64)memmove(a6, a1, a2 - a1);
        v11 = (unsigned int *)a2;
      }
      v13 = &v8[v12];
      if ( v8 != &v8[v12] )
      {
        while ( v7 != v11 )
        {
          result = *v11;
          v14 = *(_DWORD *)v8;
          if ( (unsigned int)result < *(_DWORD *)v8 )
          {
            ++v11;
          }
          else
          {
            v8 += 4;
            result = v14;
          }
          *(_DWORD *)v9 = result;
          v9 += 4;
          if ( v13 == v8 )
            return result;
        }
        return (__int64)memmove(v9, v8, v13 - v8);
      }
      return result;
    }
    v15 = a2;
    v16 = a5;
    if ( a5 <= a7 )
      break;
    if ( a5 < a4 )
    {
      v33 = a4 / 2;
      src = (unsigned int *)&a1[4 * (a4 / 2)];
      v32 = sub_25F68D0(a2, a3, src);
      v25 = ((char *)v32 - a2) >> 2;
    }
    else
    {
      v29 = a5 / 2;
      v32 = (unsigned int *)&a2[4 * (a5 / 2)];
      v22 = sub_25F6880(a1, (__int64)a2, v32);
      v25 = v29;
      src = v22;
      v33 = ((char *)v22 - a1) >> 2;
    }
    v27 = v23 - v33;
    v28 = v24;
    v30 = v25;
    v26 = sub_26036F0((char *)src, a2, (char *)v32, v23 - v33, v25, v8, v24);
    sub_2609450((_DWORD)a1, (_DWORD)src, (_DWORD)v26, v33, v30, (_DWORD)v8, v28);
    a6 = v8;
    a2 = (char *)v32;
    a1 = v26;
    a7 = v28;
    a5 = v16 - v30;
    a3 = (__int64)v7;
    a4 = v27;
  }
  v17 = a3 - (_QWORD)a2;
  if ( (char *)a3 != a2 )
    result = (__int64)memmove(a6, a2, a3 - (_QWORD)a2);
  v18 = &v8[v17];
  i = v7;
  if ( v15 == a1 )
    return (__int64)sub_2608E90(v8, v18, (__int64)i);
  if ( v8 != v18 )
  {
    v20 = v15 - 4;
    v21 = v18 - 4;
    for ( i = v7 - 1; ; --i )
    {
      result = *(unsigned int *)v21;
      if ( (unsigned int)result < *(_DWORD *)v20 )
      {
        *i = *(_DWORD *)v20;
        if ( a1 == v20 )
        {
          v18 = v21 + 4;
          return (__int64)sub_2608E90(v8, v18, (__int64)i);
        }
        v20 -= 4;
      }
      else
      {
        *i = result;
        if ( v8 == v21 )
          return result;
        v21 -= 4;
      }
    }
  }
  return result;
}
