// Function: sub_2F41CE0
// Address: 0x2f41ce0
//
__int64 __fastcall sub_2F41CE0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        unsigned int a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int64 a8)
{
  __int64 v8; // r14
  __int64 i; // r15
  __int64 v10; // r12
  unsigned int *v11; // rbx
  __int64 v12; // r14
  unsigned int *v13; // r15
  unsigned int v15; // eax
  __int64 v17; // [rsp+8h] [rbp-68h]
  __m128i v19; // [rsp+20h] [rbp-50h] BYREF
  __int64 v20; // [rsp+30h] [rbp-40h]

  v8 = (a3 - 1) / 2;
  v17 = a3 & 1;
  if ( a2 >= v8 )
  {
    v11 = (unsigned int *)(a1 + 4 * a2);
    if ( (a3 & 1) != 0 )
      goto LABEL_13;
    v10 = a2;
    goto LABEL_16;
  }
  for ( i = a2; ; i = v10 )
  {
    v10 = 2 * (i + 1);
    v11 = (unsigned int *)(a1 + 8 * (i + 1));
    if ( sub_2F41AD0(&a7, *v11, *(v11 - 1)) )
    {
      --v10;
      v11 = (unsigned int *)(a1 + 4 * v10);
    }
    *(_DWORD *)(a1 + 4 * i) = *v11;
    if ( v10 >= v8 )
      break;
  }
  if ( !v17 )
  {
LABEL_16:
    if ( (a3 - 2) / 2 == v10 )
    {
      v15 = *(_DWORD *)(a1 + 4 * (2 * v10 + 2) - 4);
      v10 = 2 * v10 + 1;
      *v11 = v15;
      v11 = (unsigned int *)(a1 + 4 * v10);
    }
  }
  v20 = a8;
  v19 = _mm_loadu_si128((const __m128i *)&a7);
  v12 = (v10 - 1) / 2;
  if ( v10 > a2 )
  {
    while ( 1 )
    {
      v13 = (unsigned int *)(a1 + 4 * v12);
      v11 = (unsigned int *)(a1 + 4 * v10);
      if ( !sub_2F41AD0(&v19, *v13, a4) )
        break;
      v10 = v12;
      *v11 = *v13;
      if ( a2 >= v12 )
      {
        v11 = (unsigned int *)(a1 + 4 * v12);
        break;
      }
      v12 = (v12 - 1) / 2;
    }
  }
LABEL_13:
  *v11 = a4;
  return a4;
}
