// Function: sub_1F21920
// Address: 0x1f21920
//
char *__fastcall sub_1F21920(char *src, char *a2, char *a3)
{
  char *v3; // r12
  int *v4; // rbx
  __int64 v5; // rax
  __int64 v6; // rdi
  __int64 v7; // r8
  int *v8; // rdx
  __int64 i; // rcx
  int v10; // esi
  __int64 v11; // rdx
  __int64 v12; // rcx
  int *v13; // rcx
  int *v14; // rsi
  __int64 v15; // rdx
  int v16; // r9d
  int v17; // r10d
  char *v19; // rax
  int v20; // ecx
  int v21; // edx
  __int64 v22; // r13
  int v23; // r14d
  int *v24; // rax
  int v25; // r13d

  v3 = a3;
  if ( src == a2 )
    return v3;
  v4 = (int *)src;
  if ( a2 == a3 )
    return src;
  v3 = &src[a3 - a2];
  v5 = (a3 - src) >> 2;
  v6 = (a2 - src) >> 2;
  if ( v6 == v5 - v6 )
  {
    v19 = a2;
    do
    {
      v20 = *(_DWORD *)v19;
      v21 = *v4++;
      v19 += 4;
      *(v4 - 1) = v20;
      *((_DWORD *)v19 - 1) = v21;
    }
    while ( a2 != (char *)v4 );
    return a2;
  }
  else
  {
    while ( 1 )
    {
      v7 = v5 - v6;
      if ( v6 < v5 - v6 )
        break;
LABEL_12:
      v12 = v5;
      if ( v7 == 1 )
      {
        v24 = &v4[v12 - 1];
        v25 = *v24;
        if ( v4 != v24 )
          memmove(v4 + 1, v4, v12 * 4 - 4);
        *v4 = v25;
        return v3;
      }
      v13 = &v4[v12];
      v4 = &v13[-v7];
      if ( v6 > 0 )
      {
        v14 = &v13[-v7];
        v15 = 0;
        do
        {
          v16 = *(v14 - 1);
          v17 = *(v13 - 1);
          ++v15;
          --v14;
          --v13;
          *v14 = v17;
          *v13 = v16;
        }
        while ( v6 != v15 );
        v4 -= v6;
      }
      v6 = v5 % v7;
      if ( !(v5 % v7) )
        return v3;
      v5 = v7;
    }
    while ( v6 != 1 )
    {
      v8 = &v4[v6];
      if ( v7 > 0 )
      {
        for ( i = 0; i != v7; ++i )
        {
          v10 = v4[i];
          v4[i] = v8[i];
          v8[i] = v10;
        }
        v4 += v7;
      }
      v11 = v5 % v6;
      if ( !(v5 % v6) )
        return v3;
      v5 = v6;
      v6 -= v11;
      v7 = v5 - v6;
      if ( v6 >= v5 - v6 )
        goto LABEL_12;
    }
    v22 = v5;
    v23 = *v4;
    if ( v4 + 1 != &v4[v5] )
      memmove(v4, v4 + 1, v22 * 4 - 4);
    v4[v22 - 1] = v23;
    return v3;
  }
}
