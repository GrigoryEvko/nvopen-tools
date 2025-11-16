// Function: sub_2231480
// Address: 0x2231480
//
_BYTE *__fastcall sub_2231480(_BYTE *a1, char a2, char *a3, __int64 a4, __int64 a5, __int64 a6)
{
  char v8; // al
  __int64 v10; // rdx
  unsigned __int64 v11; // r12
  unsigned __int64 v12; // r10
  __int64 v13; // rcx
  char *v14; // rbp
  __int64 v15; // r12
  __int64 v16; // rdi
  __int64 v17; // rax
  _BYTE *v18; // r8
  __int64 v19; // rcx
  __int64 v20; // rax
  bool v21; // cf
  __int64 v22; // rcx
  __int64 v23; // rax

  v8 = *a3;
  if ( (unsigned __int8)(*a3 - 1) > 0x7Du || (v10 = v8, v11 = a4 - 1, v12 = 0, v13 = 0, a6 - a5 <= v8) )
  {
    if ( a6 == a5 )
      return a1;
    v14 = a3;
    v16 = -1;
    v13 = 0;
    v12 = 0;
    v15 = -1;
    goto LABEL_10;
  }
  do
  {
    a6 -= v10;
    if ( v11 > v12 )
      ++v12;
    else
      ++v13;
    v14 = &a3[v12];
    v10 = a3[v12];
  }
  while ( a6 - a5 > v10 && (unsigned __int8)(v10 - 1) <= 0x7Du );
  v15 = v13 - 1;
  v16 = v12 - 1;
  if ( a5 != a6 )
  {
LABEL_10:
    v17 = 0;
    do
    {
      a1[v17] = *(_BYTE *)(a5 + v17);
      ++v17;
    }
    while ( v17 != a6 - a5 );
    v18 = &a1[v17];
    goto LABEL_13;
  }
  v18 = a1;
LABEL_13:
  if ( v13 )
  {
    do
    {
      *v18 = a2;
      v19 = *v14;
      v20 = 0;
      if ( (char)v19 <= 0 )
      {
        ++v18;
      }
      else
      {
        do
        {
          v18[v20 + 1] = *(_BYTE *)(a6 + v20);
          ++v20;
        }
        while ( v20 != v19 );
        a6 += v20;
        v18 += v20 + 1;
      }
      v21 = v15-- == 0;
    }
    while ( !v21 );
  }
  if ( v12 )
  {
    do
    {
      *v18 = a2;
      v22 = a3[v16];
      v23 = 0;
      if ( (char)v22 <= 0 )
      {
        ++v18;
      }
      else
      {
        do
        {
          v18[v23 + 1] = *(_BYTE *)(a6 + v23);
          ++v23;
        }
        while ( v23 != v22 );
        a6 += v23;
        v18 += v23 + 1;
      }
      v21 = v16-- == 0;
    }
    while ( !v21 );
  }
  return v18;
}
