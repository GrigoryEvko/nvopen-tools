// Function: sub_18E53D0
// Address: 0x18e53d0
//
__int64 __fastcall sub_18E53D0(__int64 *a1, __m128i a2, __m128i a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  unsigned int v8; // r12d
  __int64 v9; // r15
  unsigned int v10; // ebx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // r13
  __int64 v14; // rdx
  unsigned int v15; // edx
  __int64 i; // [rsp+0h] [rbp-40h]
  __int64 v18; // [rsp+8h] [rbp-38h]

  *a1 = a6;
  a1[1] = a7;
  if ( !*(_BYTE *)(a5 + 184) )
    sub_14CDF70(a5);
  v8 = 0;
  v18 = *(_QWORD *)(a5 + 8);
  for ( i = v18 + 32LL * *(unsigned int *)(a5 + 16); i != v18; v18 += 32 )
  {
    while ( 1 )
    {
      v9 = *(_QWORD *)(v18 + 16);
      if ( v9 )
        break;
LABEL_5:
      v18 += 32;
      if ( i == v18 )
        return v8;
    }
    v10 = 0;
    while ( *(char *)(v9 + 23) < 0 )
    {
      v11 = sub_1648A40(v9);
      v13 = v11 + v12;
      v14 = 0;
      if ( *(char *)(v9 + 23) < 0 )
        v14 = sub_1648A40(v9);
      if ( v10 >= (unsigned int)((v13 - v14) >> 4) )
        goto LABEL_5;
      v15 = v10++;
      v8 |= sub_18E4C60(a1, v9, v15, a2, a3);
    }
  }
  return v8;
}
