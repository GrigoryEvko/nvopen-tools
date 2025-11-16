// Function: sub_18950C0
// Address: 0x18950c0
//
char *__fastcall sub_18950C0(char *a1, unsigned __int64 *a2, __int64 a3, __int64 a4, __int64 a5, char *a6, char *a7)
{
  unsigned __int64 *v7; // r13
  char *v8; // r12
  char *v9; // rbx
  char *result; // rax
  unsigned __int64 *v11; // rcx
  char *v12; // rdx
  __int64 v13; // rdi
  __int64 v14; // rsi
  unsigned __int64 v15; // r8
  char *v16; // rsi
  unsigned __int64 *v17; // r10
  __int64 v18; // r14
  __int64 v19; // r9
  __int64 v20; // r8
  __int64 v21; // rcx
  char *v22; // rdx
  unsigned __int64 *v23; // rax
  unsigned __int64 v24; // rsi
  char *v25; // r10
  unsigned __int64 *i; // r13
  unsigned __int64 *v27; // rsi
  unsigned __int64 *v28; // rax
  char *v29; // r10
  __int64 v30; // r11
  __int64 v31; // r8
  char *v32; // rax
  int v33; // ecx
  __int64 v34; // rsi
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // rdx
  unsigned __int64 v38; // rcx
  __int64 v39; // rdx
  __int64 v40; // [rsp+18h] [rbp-58h]
  __int64 v41; // [rsp+20h] [rbp-50h]
  __int64 v42; // [rsp+20h] [rbp-50h]
  unsigned __int64 *v43; // [rsp+28h] [rbp-48h]
  unsigned __int64 *v44; // [rsp+30h] [rbp-40h]
  __int64 v45; // [rsp+38h] [rbp-38h]
  char *v46; // [rsp+38h] [rbp-38h]

  while ( 1 )
  {
    v7 = (unsigned __int64 *)a3;
    v8 = a1;
    v9 = a6;
    result = a7;
    if ( a5 <= (__int64)a7 )
      result = (char *)a5;
    if ( (__int64)result >= a4 )
      break;
    v17 = a2;
    v18 = a5;
    if ( a5 <= (__int64)a7 )
    {
      v19 = a3;
      v20 = a3 - (_QWORD)a2;
      v21 = (a3 - (__int64)a2) >> 4;
      if ( a3 - (__int64)a2 <= 0 )
        return result;
      v22 = v9;
      v23 = a2;
      do
      {
        v24 = *v23;
        v22 += 16;
        v23 += 2;
        *((_QWORD *)v22 - 2) = v24;
        *((_QWORD *)v22 - 1) = *(v23 - 1);
        --v21;
      }
      while ( v21 );
      if ( v20 <= 0 )
        v20 = 16;
      result = &v9[v20];
      if ( v17 == (unsigned __int64 *)a1 )
      {
        v39 = v20 >> 4;
        while ( 1 )
        {
          result -= 16;
          *(_QWORD *)(v19 - 16) = v24;
          v19 -= 16;
          *(_QWORD *)(v19 + 8) = *((_QWORD *)result + 1);
          if ( !--v39 )
            break;
          v24 = *((_QWORD *)result - 2);
        }
        return result;
      }
      if ( v9 == result )
        return result;
      v25 = (char *)(v17 - 2);
      result -= 16;
      for ( i = v7 - 2; ; i -= 2 )
      {
        v27 = i;
        if ( *(_QWORD *)result < *(_QWORD *)v25 )
        {
          *i = *(_QWORD *)v25;
          i[1] = *((_QWORD *)v25 + 1);
          if ( v25 == a1 )
          {
            result += 16;
            v37 = (result - v9) >> 4;
            if ( result - v9 > 0 )
            {
              do
              {
                v38 = *((_QWORD *)result - 2);
                result -= 16;
                v27 -= 2;
                *v27 = v38;
                v27[1] = *((_QWORD *)result + 1);
                --v37;
              }
              while ( v37 );
            }
            return result;
          }
          v25 -= 16;
        }
        else
        {
          *i = *(_QWORD *)result;
          i[1] = *((_QWORD *)result + 1);
          if ( v9 == result )
            return result;
          result -= 16;
        }
      }
    }
    if ( a5 < a4 )
    {
      v45 = a4 / 2;
      v43 = (unsigned __int64 *)&a1[16 * (a4 / 2)];
      v44 = sub_18906F0(a2, a3, v43);
      v31 = ((char *)v44 - v29) >> 4;
    }
    else
    {
      v41 = a5 / 2;
      v44 = &a2[2 * (a5 / 2)];
      v28 = sub_1890750(a1, (__int64)a2, v44);
      v31 = v41;
      v43 = v28;
      v45 = ((char *)v28 - a1) >> 4;
    }
    v40 = v30 - v45;
    v42 = v31;
    v32 = sub_1894EA0((char *)v43, v29, (char *)v44, v30 - v45, v31, v9, (__int64)a7);
    v33 = v45;
    v46 = v32;
    sub_18950C0((_DWORD)a1, (_DWORD)v43, (_DWORD)v32, v33, v42, (_DWORD)v9, (__int64)a7);
    a6 = v9;
    a2 = v44;
    a4 = v40;
    a5 = v18 - v42;
    a3 = (__int64)v7;
    a1 = v46;
  }
  v11 = a2;
  v12 = a6;
  result = a1;
  v13 = (char *)a2 - a1;
  v14 = ((char *)a2 - v8) >> 4;
  if ( v13 > 0 )
  {
    do
    {
      v15 = *(_QWORD *)result;
      v12 += 16;
      result += 16;
      *((_QWORD *)v12 - 2) = v15;
      *((_QWORD *)v12 - 1) = *((_QWORD *)result - 1);
      --v14;
    }
    while ( v14 );
    v16 = &a6[v13];
    if ( a6 != &a6[v13] )
    {
      while ( 1 )
      {
        result = v8;
        if ( v7 == v11 )
          break;
        if ( *v11 < *(_QWORD *)v9 )
        {
          *(_QWORD *)v8 = *v11;
          result = (char *)v11[1];
          v8 += 16;
          v11 += 2;
          *((_QWORD *)v8 - 1) = result;
          if ( v9 == v16 )
            return result;
        }
        else
        {
          *(_QWORD *)v8 = *(_QWORD *)v9;
          result = (char *)*((_QWORD *)v9 + 1);
          v9 += 16;
          v8 += 16;
          *((_QWORD *)v8 - 1) = result;
          if ( v9 == v16 )
            return result;
        }
      }
      v34 = v16 - v9;
      v35 = v34 >> 4;
      if ( v34 > 0 )
      {
        do
        {
          v36 = *(_QWORD *)v9;
          result += 16;
          v9 += 16;
          *((_QWORD *)result - 2) = v36;
          *((_QWORD *)result - 1) = *((_QWORD *)v9 - 1);
          --v35;
        }
        while ( v35 );
      }
    }
  }
  return result;
}
