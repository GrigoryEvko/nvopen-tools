// Function: sub_27ACC40
// Address: 0x27acc40
//
__int64 __fastcall sub_27ACC40(char *a1, char *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char *v6; // r15
  __int64 v7; // r14
  char *v9; // rbx
  int v10; // eax
  char **v11; // rsi
  __int64 v12; // rdi
  int v13; // eax
  char **v14; // rsi
  unsigned __int64 v15; // r12
  __int64 v16; // r13
  char **v17; // rbx
  int v18; // edx
  char **v19; // rsi
  __int64 v20; // rdi
  __int64 v21; // rdx
  __int64 v22; // rbx
  unsigned __int64 v23; // r12
  __int64 v24; // r13
  char **v25; // r15
  int v26; // eax
  char **v27; // rsi
  __int64 v28; // rdi
  __int64 v29; // rax
  __int64 v31; // [rsp+0h] [rbp-40h]
  char *v32; // [rsp+8h] [rbp-38h]

  v6 = (char *)a3;
  v7 = a5;
  v9 = a1;
  v32 = (char *)a4;
  if ( a2 != a1 && a4 != a3 )
  {
    do
    {
      v12 = v7 + 24;
      if ( *((_DWORD *)v6 + 4) > *((_DWORD *)v9 + 4) )
      {
        v10 = *(_DWORD *)v6;
        v11 = (char **)(v6 + 24);
        v7 += 72;
        v6 += 72;
        *(_DWORD *)(v7 - 72) = v10;
        *(_DWORD *)(v7 - 68) = *((_DWORD *)v6 - 17);
        *(_DWORD *)(v7 - 64) = *((_DWORD *)v6 - 16);
        *(_DWORD *)(v7 - 60) = *((_DWORD *)v6 - 15);
        *(_DWORD *)(v7 - 56) = *((_DWORD *)v6 - 14);
        sub_27AC070(v12, v11, a3, a4, a5, a6);
        if ( a2 == v9 )
          break;
      }
      else
      {
        v13 = *(_DWORD *)v9;
        v14 = (char **)(v9 + 24);
        v9 += 72;
        v7 += 72;
        *(_DWORD *)(v7 - 72) = v13;
        *(_DWORD *)(v7 - 68) = *((_DWORD *)v9 - 17);
        *(_DWORD *)(v7 - 64) = *((_DWORD *)v9 - 16);
        *(_DWORD *)(v7 - 60) = *((_DWORD *)v9 - 15);
        *(_DWORD *)(v7 - 56) = *((_DWORD *)v9 - 14);
        sub_27AC070(v12, v14, a3, a4, a5, a6);
        if ( a2 == v9 )
          break;
      }
    }
    while ( v32 != v6 );
  }
  v31 = a2 - v9;
  v15 = 0x8E38E38E38E38E39LL * ((a2 - v9) >> 3);
  if ( a2 - v9 > 0 )
  {
    v16 = v7 + 24;
    v17 = (char **)(v9 + 24);
    do
    {
      v18 = *((_DWORD *)v17 - 6);
      v19 = v17;
      v20 = v16;
      v17 += 9;
      v16 += 72;
      *(_DWORD *)(v16 - 96) = v18;
      *(_DWORD *)(v16 - 92) = *((_DWORD *)v17 - 23);
      *(_DWORD *)(v16 - 88) = *((_DWORD *)v17 - 22);
      *(_DWORD *)(v16 - 84) = *((_DWORD *)v17 - 21);
      v21 = *((unsigned int *)v17 - 20);
      *(_DWORD *)(v16 - 80) = v21;
      sub_27AC070(v20, v19, v21, a4, a5, a6);
      --v15;
    }
    while ( v15 );
    a4 = v31;
    if ( v31 <= 0 )
      a4 = 72;
    v7 += a4;
  }
  v22 = v32 - v6;
  v23 = 0x8E38E38E38E38E39LL * ((v32 - v6) >> 3);
  if ( v32 - v6 <= 0 )
    return v7;
  v24 = v7 + 24;
  v25 = (char **)(v6 + 24);
  do
  {
    v26 = *((_DWORD *)v25 - 6);
    v27 = v25;
    v28 = v24;
    v25 += 9;
    v24 += 72;
    *(_DWORD *)(v24 - 96) = v26;
    *(_DWORD *)(v24 - 92) = *((_DWORD *)v25 - 23);
    *(_DWORD *)(v24 - 88) = *((_DWORD *)v25 - 22);
    *(_DWORD *)(v24 - 84) = *((_DWORD *)v25 - 21);
    *(_DWORD *)(v24 - 80) = *((_DWORD *)v25 - 20);
    sub_27AC070(v28, v27, a3, a4, a5, a6);
    --v23;
  }
  while ( v23 );
  v29 = 72;
  if ( v22 > 0 )
    v29 = v22;
  return v7 + v29;
}
