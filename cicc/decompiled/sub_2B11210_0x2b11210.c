// Function: sub_2B11210
// Address: 0x2b11210
//
__int64 __fastcall sub_2B11210(__int64 *a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r15
  __int64 v8; // r12
  __int64 *v9; // rbx
  __int64 v10; // rax
  char **v11; // rsi
  __int64 v12; // rdi
  __int64 v13; // rax
  char **v14; // rsi
  unsigned __int64 v15; // r13
  __int64 v16; // r14
  char **v17; // rbx
  __int64 v18; // rcx
  char **v19; // rsi
  __int64 v20; // rdi
  __int64 v21; // rdx
  unsigned __int64 v22; // r13
  __int64 v23; // r14
  char **v24; // rbx
  __int64 v25; // rax
  char **v26; // rsi
  __int64 v27; // rdi
  __int64 v29; // [rsp+0h] [rbp-40h]
  __int64 v30; // [rsp+8h] [rbp-38h]
  __int64 v31; // [rsp+8h] [rbp-38h]

  v6 = a3;
  v8 = a5;
  v9 = a1;
  v30 = a4;
  if ( a2 != a1 && a4 != a3 )
  {
    do
    {
      v12 = v8 + 8;
      if ( *(_DWORD *)(v6 + 16) > *((_DWORD *)v9 + 4) )
      {
        v10 = *(_QWORD *)v6;
        v11 = (char **)(v6 + 8);
        v8 += 72;
        v6 += 72;
        *(_QWORD *)(v8 - 72) = v10;
        sub_2B0D090(v12, v11, a3, a4, a5, a6);
        if ( a2 == v9 )
          break;
      }
      else
      {
        v13 = *v9;
        v14 = (char **)(v9 + 1);
        v9 += 9;
        v8 += 72;
        *(_QWORD *)(v8 - 72) = v13;
        sub_2B0D090(v12, v14, a3, a4, a5, a6);
        if ( a2 == v9 )
          break;
      }
    }
    while ( v30 != v6 );
  }
  v29 = (char *)a2 - (char *)v9;
  v15 = 0x8E38E38E38E38E39LL * (a2 - v9);
  if ( (char *)a2 - (char *)v9 > 0 )
  {
    v16 = v8 + 8;
    v17 = (char **)(v9 + 1);
    do
    {
      v18 = (__int64)*(v17 - 1);
      v19 = v17;
      v20 = v16;
      v17 += 9;
      v16 += 72;
      *(_QWORD *)(v16 - 80) = v18;
      sub_2B0D090(v20, v19, a3, v18, a5, a6);
      --v15;
    }
    while ( v15 );
    a4 = v29;
    if ( v29 <= 0 )
      a4 = 72;
    v8 += a4;
  }
  v31 = v30 - v6;
  v21 = v31;
  v22 = 0x8E38E38E38E38E39LL * (v31 >> 3);
  if ( v31 <= 0 )
    return v8;
  v23 = v8 + 8;
  v24 = (char **)(v6 + 8);
  do
  {
    v25 = (__int64)*(v24 - 1);
    v26 = v24;
    v27 = v23;
    v24 += 9;
    v23 += 72;
    *(_QWORD *)(v23 - 80) = v25;
    sub_2B0D090(v27, v26, v21, a4, a5, a6);
    --v22;
  }
  while ( v22 );
  return v8 + v31;
}
