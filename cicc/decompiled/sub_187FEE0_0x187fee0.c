// Function: sub_187FEE0
// Address: 0x187fee0
//
__int64 __fastcall sub_187FEE0(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // r14
  const void *v4; // r15
  size_t v5; // r13
  size_t v6; // rbx
  size_t v7; // rdx
  int v8; // eax
  __int64 v9; // rbx
  __int64 v11; // [rsp+8h] [rbp-38h]

  v2 = a1 + 8;
  v3 = *(_QWORD *)(a1 + 16);
  v11 = a1 + 8;
  if ( !v3 )
    return v11;
  v4 = *(const void **)a2;
  v5 = *(_QWORD *)(a2 + 8);
  do
  {
    while ( 1 )
    {
      v6 = *(_QWORD *)(v3 + 40);
      v7 = v5;
      if ( v6 <= v5 )
        v7 = *(_QWORD *)(v3 + 40);
      if ( v7 )
      {
        v8 = memcmp(*(const void **)(v3 + 32), v4, v7);
        if ( v8 )
          break;
      }
      v9 = v6 - v5;
      if ( v9 >= 0x80000000LL )
        goto LABEL_12;
      if ( v9 > (__int64)0xFFFFFFFF7FFFFFFFLL )
      {
        v8 = v9;
        break;
      }
LABEL_3:
      v3 = *(_QWORD *)(v3 + 24);
      if ( !v3 )
        goto LABEL_13;
    }
    if ( v8 < 0 )
      goto LABEL_3;
LABEL_12:
    v2 = v3;
    v3 = *(_QWORD *)(v3 + 16);
  }
  while ( v3 );
LABEL_13:
  if ( v11 != v2 )
  {
    if ( sub_1872D20(v4, v5, *(const void **)(v2 + 32), *(_QWORD *)(v2 + 40)) < 0 )
      return a1 + 8;
    return v2;
  }
  return v11;
}
