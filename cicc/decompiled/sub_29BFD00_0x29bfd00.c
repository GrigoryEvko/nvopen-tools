// Function: sub_29BFD00
// Address: 0x29bfd00
//
char *__fastcall sub_29BFD00(char *a1, char *a2, __int64 a3, __int64 a4, __int64 a5, char *a6, char *a7, __int64 *a8)
{
  _QWORD *v8; // r14
  char *v9; // r12
  char *result; // rax
  char *v11; // rcx
  signed __int64 v12; // r13
  char *v13; // r11
  char *v14; // rsi
  __int64 v15; // r15
  _QWORD *v16; // r15
  size_t v17; // r8
  char *v18; // rdx
  char *v19; // r13
  char *v20; // rax
  __int64 v21; // r10
  __int64 v22; // r11
  char *v23; // r9
  __int64 v24; // r8
  char *v25; // r13
  char *v26; // rax
  char *v27; // rdx
  __int64 v28; // [rsp+8h] [rbp-68h]
  __int64 v29; // [rsp+10h] [rbp-60h]
  char *dest; // [rsp+18h] [rbp-58h]
  char *desta; // [rsp+18h] [rbp-58h]
  __int64 v32; // [rsp+20h] [rbp-50h]
  __int64 v33; // [rsp+20h] [rbp-50h]
  char *v34; // [rsp+20h] [rbp-50h]
  char *src; // [rsp+28h] [rbp-48h]
  char *v36; // [rsp+30h] [rbp-40h]
  size_t v37; // [rsp+38h] [rbp-38h]
  __int64 v38; // [rsp+38h] [rbp-38h]

  while ( 1 )
  {
    v8 = (_QWORD *)a3;
    v9 = a1;
    result = a7;
    if ( a5 <= (__int64)a7 )
      result = (char *)a5;
    if ( a4 <= (__int64)result )
    {
      v11 = a2;
      v12 = a2 - a1;
      if ( a2 != a1 )
      {
        result = (char *)memmove(a6, a1, a2 - a1);
        v11 = a2;
        a6 = result;
      }
      v13 = &a6[v12];
      if ( a6 != &a6[v12] )
      {
        while ( v8 != (_QWORD *)v11 )
        {
          v14 = *(char **)a6;
          result = *(char **)v11;
          if ( *(_DWORD *)(*a8 + 16LL * *(_QWORD *)v11) < *(_DWORD *)(*a8 + 16LL * *(_QWORD *)a6) )
          {
            v11 += 8;
          }
          else
          {
            a6 += 8;
            result = v14;
          }
          *(_QWORD *)v9 = result;
          v9 += 8;
          if ( v13 == a6 )
            return result;
        }
        return (char *)memmove(v9, a6, v13 - a6);
      }
      return result;
    }
    v15 = a5;
    if ( a5 <= (__int64)a7 )
      break;
    if ( a4 > a5 )
    {
      v34 = a6;
      v38 = a4 / 2;
      src = &a1[8 * (a4 / 2)];
      v26 = (char *)sub_29BF5B0(a2, a3, src, a8);
      v23 = v34;
      v36 = v26;
      v24 = (v26 - a2) >> 3;
    }
    else
    {
      dest = a6;
      v32 = a5 / 2;
      v36 = &a2[8 * (a5 / 2)];
      v20 = (char *)sub_29BF610(a1, (__int64)a2, v36, a8);
      v23 = dest;
      v24 = v32;
      src = v20;
      v38 = (v20 - a1) >> 3;
    }
    v28 = v21 - v38;
    v29 = v22;
    desta = v23;
    v33 = v24;
    v25 = sub_29BFBE0(src, a2, v36, v21 - v38, v24, v23, v22);
    sub_29BFD00((_DWORD)a1, (_DWORD)src, (_DWORD)v25, v38, v33, (_DWORD)desta, v29, (__int64)a8);
    a3 = (__int64)v8;
    a1 = v25;
    a6 = desta;
    a2 = v36;
    a7 = (char *)v29;
    a5 = v15 - v33;
    a4 = v28;
  }
  v16 = (_QWORD *)a3;
  v17 = a3 - (_QWORD)a2;
  if ( (char *)a3 != a2 )
  {
    v37 = a3 - (_QWORD)a2;
    result = (char *)memmove(a6, a2, a3 - (_QWORD)a2);
    v17 = v37;
    a6 = result;
  }
  v18 = &a6[v17];
  if ( a2 == a1 )
  {
    if ( a6 != v18 )
      return (char *)memmove((char *)v8 - v17, a6, v17);
  }
  else if ( a6 != v18 )
  {
    v19 = a2 - 8;
    while ( 2 )
    {
      v18 -= 8;
      while ( 1 )
      {
        --v16;
        result = (char *)*a8;
        if ( *(_DWORD *)(*a8 + 16LL * *(_QWORD *)v18) >= *(_DWORD *)(*a8 + 16LL * *(_QWORD *)v19) )
          break;
        *v16 = *(_QWORD *)v19;
        if ( a1 == v19 )
        {
          v27 = v18 + 8;
          if ( a6 != v27 )
            return (char *)memmove((char *)v16 - (v27 - a6), a6, v27 - a6);
          return result;
        }
        v19 -= 8;
      }
      *v16 = *(_QWORD *)v18;
      if ( a6 != v18 )
        continue;
      break;
    }
  }
  return result;
}
