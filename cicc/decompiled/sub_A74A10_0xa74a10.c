// Function: sub_A74A10
// Address: 0xa74a10
//
__int64 __fastcall sub_A74A10(__int64 a1, __int64 a2)
{
  __int64 *v2; // rbx
  __int64 v3; // r13
  __int64 *v4; // r14
  __int64 v5; // rax
  __int64 v6; // r13
  __int64 *v7; // r13
  __int64 *i; // r13
  __int64 v9; // rax
  __int64 v10; // r12
  __int64 v11; // rbx

  v2 = *(__int64 **)(a1 + 8);
  v3 = 8LL * *(unsigned int *)(a1 + 16);
  v4 = &v2[(unsigned __int64)v3 / 8];
  v5 = v3 >> 3;
  v6 = v3 >> 5;
  if ( v6 )
  {
    v7 = &v2[4 * v6];
    while ( !(unsigned __int8)sub_A71FF0(a2, *v2) )
    {
      if ( (unsigned __int8)sub_A71FF0(a2, v2[1]) )
      {
        ++v2;
        goto LABEL_8;
      }
      if ( (unsigned __int8)sub_A71FF0(a2, v2[2]) )
      {
        v2 += 2;
        goto LABEL_8;
      }
      if ( (unsigned __int8)sub_A71FF0(a2, v2[3]) )
      {
        v2 += 3;
        goto LABEL_8;
      }
      v2 += 4;
      if ( v7 == v2 )
      {
        v5 = v4 - v2;
        goto LABEL_17;
      }
    }
    goto LABEL_8;
  }
LABEL_17:
  if ( v5 == 2 )
  {
LABEL_23:
    if ( (unsigned __int8)sub_A71FF0(a2, *v2) )
      goto LABEL_8;
    ++v2;
    goto LABEL_25;
  }
  if ( v5 == 3 )
  {
    if ( (unsigned __int8)sub_A71FF0(a2, *v2) )
      goto LABEL_8;
    ++v2;
    goto LABEL_23;
  }
  if ( v5 != 1 )
  {
LABEL_20:
    v2 = v4;
    goto LABEL_13;
  }
LABEL_25:
  if ( !(unsigned __int8)sub_A71FF0(a2, *v2) )
    goto LABEL_20;
LABEL_8:
  if ( v4 != v2 )
  {
    for ( i = v2 + 1; v4 != i; ++i )
    {
      if ( !(unsigned __int8)sub_A71FF0(a2, *i) )
        *v2++ = *i;
    }
  }
LABEL_13:
  v9 = *(_QWORD *)(a1 + 8);
  v10 = v9 + 8LL * *(unsigned int *)(a1 + 16) - (_QWORD)v4;
  if ( v4 != (__int64 *)(v9 + 8LL * *(unsigned int *)(a1 + 16)) )
  {
    memmove(v2, v4, v9 + 8LL * *(unsigned int *)(a1 + 16) - (_QWORD)v4);
    v9 = *(_QWORD *)(a1 + 8);
  }
  v11 = (__int64)v2 + v10 - v9;
  *(_DWORD *)(a1 + 16) = v11 >> 3;
  return a1;
}
