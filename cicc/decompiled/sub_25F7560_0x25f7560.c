// Function: sub_25F7560
// Address: 0x25f7560
//
__int64 __fastcall sub_25F7560(char *a1, char *a2, char *a3, char *a4, __int64 a5)
{
  char *v5; // r15
  char *v8; // rbx
  __int64 v9; // rsi
  __int64 v10; // rdi
  __int64 v11; // rsi
  __int64 v12; // rdi
  unsigned __int64 v13; // r13
  __int64 v14; // r14
  __int64 v15; // rsi
  __int64 v16; // rdi
  __int64 v17; // rcx
  __int64 v18; // r14
  unsigned __int64 v19; // rbx
  __int64 v20; // r13
  __int64 v21; // rsi
  __int64 v22; // rdi
  __int64 v23; // rax
  __int64 v25; // [rsp+0h] [rbp-40h]

  v5 = a3;
  v8 = a1;
  if ( a2 != a1 && a4 != a3 )
  {
    do
    {
      if ( *(_DWORD *)v5 < *(_DWORD *)v8 )
      {
        v9 = (__int64)v5;
        v10 = a5;
        v5 += 152;
        a5 += 152;
        sub_25F6310(v10, v9);
        if ( a2 == v8 )
          break;
      }
      else
      {
        v11 = (__int64)v8;
        v12 = a5;
        v8 += 152;
        a5 += 152;
        sub_25F6310(v12, v11);
        if ( a2 == v8 )
          break;
      }
    }
    while ( a4 != v5 );
  }
  v25 = a2 - v8;
  v13 = 0x86BCA1AF286BCA1BLL * ((a2 - v8) >> 3);
  if ( a2 - v8 > 0 )
  {
    v14 = a5;
    do
    {
      v15 = (__int64)v8;
      v16 = v14;
      v8 += 152;
      v14 += 152;
      sub_25F6310(v16, v15);
      --v13;
    }
    while ( v13 );
    v17 = v25;
    if ( v25 <= 0 )
      v17 = 152;
    a5 += v17;
  }
  v18 = a4 - v5;
  v19 = 0x86BCA1AF286BCA1BLL * ((a4 - v5) >> 3);
  if ( a4 - v5 > 0 )
  {
    v20 = a5;
    do
    {
      v21 = (__int64)v5;
      v22 = v20;
      v5 += 152;
      v20 += 152;
      sub_25F6310(v22, v21);
      --v19;
    }
    while ( v19 );
    v23 = 152;
    if ( v18 > 0 )
      v23 = v18;
    a5 += v23;
  }
  return a5;
}
