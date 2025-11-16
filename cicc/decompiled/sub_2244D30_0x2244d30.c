// Function: sub_2244D30
// Address: 0x2244d30
//
_DWORD *__fastcall sub_2244D30(_DWORD *a1, int a2, char *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // r10
  unsigned __int64 v9; // r12
  unsigned __int64 v10; // r11
  __int64 v11; // rcx
  char *v12; // rbx
  __int64 v13; // r12
  __int64 v14; // r10
  unsigned __int64 v15; // rax
  _DWORD *v16; // r8
  __int64 v17; // rax
  __int64 v18; // rdx
  bool v19; // cf
  __int64 v20; // rax
  __int64 v21; // rdx

  v8 = *a3;
  if ( (a6 - a5) >> 2 <= v8 || (v9 = a4 - 1, v10 = 0, v11 = 0, (unsigned __int8)(v8 - 1) > 0x7Du) )
  {
    if ( a6 == a5 )
      return a1;
    v12 = a3;
    v14 = -1;
    v11 = 0;
    v10 = 0;
    v13 = -1;
    goto LABEL_10;
  }
  do
  {
    a6 -= 4 * v8;
    if ( v9 > v10 )
      ++v10;
    else
      ++v11;
    v12 = &a3[v10];
    v8 = a3[v10];
  }
  while ( (a6 - a5) >> 2 > v8 && (unsigned __int8)(v8 - 1) <= 0x7Du );
  v13 = v11 - 1;
  v14 = v10 - 1;
  if ( a5 != a6 )
  {
LABEL_10:
    v15 = 0;
    do
    {
      a1[v15 / 4] = *(_DWORD *)(a5 + v15);
      v15 += 4LL;
    }
    while ( v15 != a6 - a5 );
    v16 = &a1[v15 / 4];
    goto LABEL_13;
  }
  v16 = a1;
LABEL_13:
  if ( v11 )
  {
    do
    {
      *v16 = a2;
      v17 = 0;
      v18 = *v12;
      if ( (char)v18 <= 0 )
      {
        ++v16;
      }
      else
      {
        do
        {
          v16[v17 + 1] = *(_DWORD *)(a6 + v17 * 4);
          ++v17;
        }
        while ( v17 != v18 );
        a6 += v17 * 4;
        v16 = (_DWORD *)((char *)v16 + v17 * 4 + 4);
      }
      v19 = v13-- == 0;
    }
    while ( !v19 );
  }
  if ( v10 )
  {
    do
    {
      *v16 = a2;
      v20 = 0;
      v21 = a3[v14];
      if ( (char)v21 <= 0 )
      {
        ++v16;
      }
      else
      {
        do
        {
          v16[v20 + 1] = *(_DWORD *)(a6 + v20 * 4);
          ++v20;
        }
        while ( v20 != v21 );
        a6 += v20 * 4;
        v16 = (_DWORD *)((char *)v16 + v20 * 4 + 4);
      }
      v19 = v14-- == 0;
    }
    while ( !v19 );
  }
  return v16;
}
