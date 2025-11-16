// Function: sub_2B0FFB0
// Address: 0x2b0ffb0
//
__int64 __fastcall sub_2B0FFB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r15
  __int64 v8; // r12
  __int64 v9; // rbx
  char **v10; // rsi
  __int64 v11; // rdi
  char **v12; // rsi
  __int64 v13; // rdi
  __int64 v14; // r13
  __int64 v15; // r14
  char **v16; // rsi
  __int64 v17; // rdi
  __int64 v18; // r14
  __int64 v19; // rbx
  __int64 v20; // r13
  char **v21; // rsi
  __int64 v22; // rdi
  __int64 v23; // rax
  __int64 v25; // [rsp+0h] [rbp-40h]

  v6 = a3;
  v8 = a5;
  v9 = a1;
  if ( a2 != a1 && a4 != a3 )
  {
    do
    {
      if ( *(_DWORD *)(v9 + 8) < *(_DWORD *)(v6 + 8) )
      {
        v10 = (char **)v6;
        v11 = v8;
        v6 += 64;
        v8 += 64;
        sub_2B0F6D0(v11, v10, a3, a4, a5, a6);
        if ( a2 == v9 )
          break;
      }
      else
      {
        v12 = (char **)v9;
        v13 = v8;
        v9 += 64;
        v8 += 64;
        sub_2B0F6D0(v13, v12, a3, a4, a5, a6);
        if ( a2 == v9 )
          break;
      }
    }
    while ( a4 != v6 );
  }
  v25 = a2 - v9;
  v14 = (a2 - v9) >> 6;
  if ( a2 - v9 > 0 )
  {
    v15 = v8;
    do
    {
      v16 = (char **)v9;
      v17 = v15;
      v9 += 64;
      v15 += 64;
      sub_2B0F6D0(v17, v16, a3, a4, a5, a6);
      --v14;
    }
    while ( v14 );
    a3 = v25;
    if ( v25 <= 0 )
      a3 = 64;
    v8 += a3;
  }
  v18 = a4 - v6;
  v19 = (a4 - v6) >> 6;
  if ( a4 - v6 > 0 )
  {
    v20 = v8;
    do
    {
      v21 = (char **)v6;
      v22 = v20;
      v6 += 64;
      v20 += 64;
      sub_2B0F6D0(v22, v21, a3, a4, a5, a6);
      --v19;
    }
    while ( v19 );
    v23 = 64;
    if ( v18 > 0 )
      v23 = v18;
    v8 += v23;
  }
  return v8;
}
