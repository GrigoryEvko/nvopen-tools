// Function: sub_84DAC0
// Address: 0x84dac0
//
_BYTE *__fastcall sub_84DAC0(__int64 *a1, __int64 a2, const char *a3, const char *a4)
{
  const char *v5; // r12
  size_t v7; // r13
  size_t v8; // r15
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 *v12; // r9
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 *v15; // r9
  char *v16; // rax
  char *v17; // rcx
  size_t v18; // r13
  __int64 v19; // rdx
  char *v20; // rax
  char *v21; // r13
  size_t v22; // r15
  _BYTE *result; // rax
  __int64 v24; // [rsp+0h] [rbp-40h]
  __int64 v25; // [rsp+8h] [rbp-38h]
  __int64 v26; // [rsp+8h] [rbp-38h]

  v5 = a3;
  v7 = strlen(a3);
  v8 = strlen(a4);
  v24 = v7 + v8 + 1;
  v25 = sub_823970(v24);
  sub_823A00(*a1, a1[1], v9, v10, v11, v12);
  a1[2] = 0;
  *a1 = v25;
  a1[1] = v24;
  sub_823A00(0, 0, v24, v13, v14, v15);
  v26 = a1[2];
  sub_84D980(a1, v7 + a1[2]);
  if ( v7 )
  {
    v16 = (char *)(v26 + *a1);
    v17 = &v16[v7];
    do
    {
      if ( v16 )
        *v16 = *v5;
      ++v16;
      ++v5;
    }
    while ( v17 != v16 );
  }
  v18 = a1[2] + v7;
  a1[2] = v18;
  sub_84D980(a1, v8 + v18);
  v19 = *a1;
  if ( v8 )
  {
    v20 = (char *)(v19 + v18);
    v21 = (char *)(v8 + v19 + v18);
    do
    {
      if ( v20 )
        *v20 = *a4;
      ++v20;
      ++a4;
    }
    while ( v20 != v21 );
  }
  v22 = a1[2] + v8;
  a1[2] = v22;
  if ( a1[1] == v22 )
    sub_7CB020(a1);
  result = (_BYTE *)(*a1 + v22);
  if ( result )
    *result = 0;
  a1[2] = v22 + 1;
  return result;
}
