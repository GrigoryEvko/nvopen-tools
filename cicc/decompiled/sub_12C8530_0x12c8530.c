// Function: sub_12C8530
// Address: 0x12c8530
//
__int64 __fastcall sub_12C8530(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // r13
  const void *v4; // r14
  size_t v5; // r15
  size_t v6; // rbx
  size_t v7; // rdx
  int v8; // eax
  __int64 v9; // rbx
  size_t v10; // rbx
  size_t v11; // rdx
  int v12; // eax

  v2 = a1 + 8;
  v3 = *(_QWORD *)(a1 + 16);
  if ( !v3 )
    return a1 + 8;
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
  if ( a1 + 8 != v2 )
  {
    v10 = *(_QWORD *)(v2 + 40);
    v11 = v5;
    if ( v10 <= v5 )
      v11 = *(_QWORD *)(v2 + 40);
    if ( v11 && (v12 = memcmp(v4, *(const void **)(v2 + 32), v11)) != 0 )
    {
LABEL_21:
      if ( v12 < 0 )
        return a1 + 8;
    }
    else if ( (__int64)(v5 - v10) <= 0x7FFFFFFF )
    {
      if ( (__int64)(v5 - v10) >= (__int64)0xFFFFFFFF80000000LL )
      {
        v12 = v5 - v10;
        goto LABEL_21;
      }
      return a1 + 8;
    }
  }
  return v2;
}
