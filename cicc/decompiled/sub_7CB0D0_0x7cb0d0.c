// Function: sub_7CB0D0
// Address: 0x7cb0d0
//
_BYTE *__fastcall sub_7CB0D0(__int64 *a1, __int64 a2, const char *a3, _BYTE *a4, __int64 a5)
{
  const char *v5; // r13
  size_t v7; // rbx
  __int64 v8; // r14
  __int64 v9; // r14
  size_t v10; // rcx
  _BYTE *v11; // r8
  char *v12; // rax
  char *v13; // r8
  __int64 v14; // rbx
  __int64 v15; // rsi
  __int64 v16; // r14
  _BYTE *v17; // r13
  _BYTE *v18; // rax
  __int64 v19; // r12
  _BYTE *result; // rax
  _BYTE *v21; // rax
  _BYTE *v22; // r9
  _BYTE *v23; // r8
  _BYTE *v24; // r11
  _BYTE *v25; // rsi
  _BYTE *v26; // rax
  _BYTE *v27; // r8
  _BYTE *v28; // r10
  _BYTE *v29; // rdx
  _BYTE *v30; // rcx
  __int64 v31; // [rsp+0h] [rbp-50h]
  _BYTE *v32; // [rsp+8h] [rbp-48h]
  _BYTE *v33; // [rsp+8h] [rbp-48h]
  _BYTE *v35; // [rsp+18h] [rbp-38h]
  _BYTE *v36; // [rsp+18h] [rbp-38h]

  v5 = a3;
  v7 = strlen(a3);
  v8 = sub_823970(a5 + v7 + 1);
  sub_823A00(*a1, a1[1]);
  *a1 = v8;
  a1[2] = 0;
  a1[1] = a5 + v7 + 1;
  sub_823A00(0, 0);
  v9 = a1[2];
  v10 = v7 + v9;
  if ( (__int64)(v7 + v9) > a1[1] )
  {
    v31 = a1[1];
    v32 = (_BYTE *)*a1;
    v21 = (_BYTE *)sub_823970(v7 + v9);
    v22 = v32;
    v23 = v21;
    v24 = &v21[v9];
    v25 = v32;
    if ( v9 > 0 )
    {
      do
      {
        if ( v21 )
          *v21 = *v25;
        ++v21;
        ++v25;
      }
      while ( v24 != v21 );
    }
    v33 = v23;
    sub_823A00(v22, v31);
    v11 = v33;
    v10 = v7 + v9;
    *a1 = (__int64)v33;
    a1[1] = v7 + v9;
  }
  else
  {
    v11 = (_BYTE *)*a1;
  }
  if ( v7 )
  {
    v12 = &v11[v9];
    v13 = &v11[v10];
    do
    {
      if ( v12 )
        *v12 = *v5;
      ++v12;
      ++v5;
    }
    while ( v13 != v12 );
    v11 = (_BYTE *)*a1;
  }
  v14 = a1[2] + v7;
  v15 = a1[1];
  v16 = a5 + v14;
  a1[2] = v14;
  v17 = a4;
  if ( a5 + v14 > v15 )
  {
    v35 = v11;
    v26 = (_BYTE *)sub_823970(a5 + v14);
    v27 = v35;
    v28 = v26;
    v29 = &v26[v14];
    v30 = v35;
    if ( v14 > 0 )
    {
      do
      {
        if ( v26 )
          *v26 = *v30;
        ++v26;
        ++v30;
      }
      while ( v29 != v26 );
    }
    v36 = v28;
    sub_823A00(v27, v15);
    a1[1] = v16;
    *a1 = (__int64)v36;
    v11 = v36;
  }
  v18 = &v11[v14];
  if ( a5 > 0 )
  {
    do
    {
      if ( v18 )
        *v18 = *v17;
      ++v18;
      ++v17;
    }
    while ( v18 != &v11[v16] );
  }
  v19 = a1[2] + a5;
  a1[2] = v19;
  if ( a1[1] == v19 )
    sub_7CB020(a1);
  result = (_BYTE *)(*a1 + v19);
  if ( result )
    *result = 0;
  a1[2] = v19 + 1;
  return result;
}
