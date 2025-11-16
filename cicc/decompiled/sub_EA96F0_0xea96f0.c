// Function: sub_EA96F0
// Address: 0xea96f0
//
__int64 __fastcall sub_EA96F0(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  const void *v3; // r15
  size_t v4; // r13
  __int64 v5; // r14
  size_t v6; // r12
  size_t v7; // rdx
  int v8; // eax
  size_t v9; // rbx
  size_t v10; // rdx
  int v11; // eax

  v2 = *(_QWORD *)(a1 + 16);
  if ( !v2 )
    return a1 + 8;
  v3 = *(const void **)a2;
  v4 = *(_QWORD *)(a2 + 8);
  v5 = a1 + 8;
  do
  {
    while ( 1 )
    {
      v6 = *(_QWORD *)(v2 + 40);
      v7 = v4;
      if ( v6 <= v4 )
        v7 = *(_QWORD *)(v2 + 40);
      if ( v7 )
      {
        v8 = memcmp(*(const void **)(v2 + 32), v3, v7);
        if ( v8 )
          break;
      }
      if ( v6 != v4 && v6 < v4 )
      {
        v2 = *(_QWORD *)(v2 + 24);
        goto LABEL_11;
      }
LABEL_3:
      v5 = v2;
      v2 = *(_QWORD *)(v2 + 16);
      if ( !v2 )
        goto LABEL_12;
    }
    if ( v8 >= 0 )
      goto LABEL_3;
    v2 = *(_QWORD *)(v2 + 24);
LABEL_11:
    ;
  }
  while ( v2 );
LABEL_12:
  if ( a1 + 8 != v5 )
  {
    v9 = *(_QWORD *)(v5 + 40);
    v10 = v9;
    if ( v4 <= v9 )
      v10 = v4;
    if ( v10 && (v11 = memcmp(v3, *(const void **)(v5 + 32), v10)) != 0 )
    {
      if ( v11 >= 0 )
        return v5;
    }
    else if ( v4 == v9 || v4 >= v9 )
    {
      return v5;
    }
  }
  return a1 + 8;
}
