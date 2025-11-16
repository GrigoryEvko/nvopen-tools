// Function: sub_BAF930
// Address: 0xbaf930
//
__int64 __fastcall sub_BAF930(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  const void *v3; // r14
  size_t v4; // r13
  __int64 v5; // rax
  char v6; // cl
  size_t v7; // r12
  const void *v8; // r15
  size_t v9; // rdx
  int v10; // eax
  int v11; // eax
  __int64 v13; // rax
  size_t n; // [rsp+18h] [rbp-38h]

  v2 = *(_QWORD *)(a1 + 16);
  if ( !v2 )
  {
    v2 = a1 + 8;
    goto LABEL_22;
  }
  v3 = *(const void **)a2;
  v4 = *(_QWORD *)(a2 + 8);
  while ( 1 )
  {
    v7 = *(_QWORD *)(v2 + 40);
    v8 = *(const void **)(v2 + 32);
    v9 = v7;
    if ( v4 <= v7 )
      v9 = v4;
    if ( v9 )
    {
      n = v9;
      v10 = memcmp(v3, *(const void **)(v2 + 32), v9);
      v9 = n;
      if ( v10 )
        break;
    }
    if ( v4 == v7 || v4 >= v7 )
    {
      v5 = *(_QWORD *)(v2 + 24);
      v6 = 0;
      goto LABEL_12;
    }
LABEL_3:
    v5 = *(_QWORD *)(v2 + 16);
    v6 = 1;
    if ( !v5 )
      goto LABEL_13;
LABEL_4:
    v2 = v5;
  }
  if ( v10 < 0 )
    goto LABEL_3;
  v5 = *(_QWORD *)(v2 + 24);
  v6 = 0;
LABEL_12:
  if ( v5 )
    goto LABEL_4;
LABEL_13:
  if ( !v6 )
    goto LABEL_14;
LABEL_22:
  if ( *(_QWORD *)(a1 + 24) == v2 )
    return 0;
  v13 = sub_220EF80(v2);
  v7 = *(_QWORD *)(v13 + 40);
  v8 = *(const void **)(v13 + 32);
  v2 = v13;
  v4 = *(_QWORD *)(a2 + 8);
  v3 = *(const void **)a2;
  v9 = v4;
  if ( v7 <= v4 )
    v9 = *(_QWORD *)(v13 + 40);
LABEL_14:
  if ( v9 && (v11 = memcmp(v8, v3, v9)) != 0 )
  {
    if ( v11 < 0 )
      return 0;
  }
  else if ( v7 != v4 && v7 < v4 )
  {
    return 0;
  }
  return v2;
}
