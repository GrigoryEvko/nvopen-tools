// Function: sub_391AB80
// Address: 0x391ab80
//
__int64 __fastcall sub_391AB80(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  const void *v3; // r14
  size_t v4; // r13
  int v5; // eax
  __int64 v6; // rax
  char v7; // dl
  size_t v8; // r12
  const void *v9; // r15
  int v10; // eax
  __int64 v12; // rax

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
    v8 = *(_QWORD *)(v2 + 40);
    v9 = *(const void **)(v2 + 32);
    if ( v8 < v4 )
      break;
    if ( v4 )
    {
      v5 = memcmp(v3, *(const void **)(v2 + 32), v4);
      if ( v5 )
        goto LABEL_12;
    }
    if ( v8 == v4 )
      goto LABEL_13;
LABEL_6:
    if ( v8 <= v4 )
      goto LABEL_13;
LABEL_7:
    v6 = *(_QWORD *)(v2 + 16);
    v7 = 1;
    if ( !v6 )
      goto LABEL_14;
LABEL_8:
    v2 = v6;
  }
  if ( !v8 )
    goto LABEL_13;
  v5 = memcmp(v3, *(const void **)(v2 + 32), *(_QWORD *)(v2 + 40));
  if ( !v5 )
    goto LABEL_6;
LABEL_12:
  if ( v5 < 0 )
    goto LABEL_7;
LABEL_13:
  v6 = *(_QWORD *)(v2 + 24);
  v7 = 0;
  if ( v6 )
    goto LABEL_8;
LABEL_14:
  if ( v7 )
  {
LABEL_22:
    if ( *(_QWORD *)(a1 + 24) == v2 )
      return 0;
    v12 = sub_220EF80(v2);
    v8 = *(_QWORD *)(v12 + 40);
    v9 = *(const void **)(v12 + 32);
    v4 = *(_QWORD *)(a2 + 8);
    v3 = *(const void **)a2;
    v2 = v12;
    if ( v4 >= v8 )
      goto LABEL_16;
LABEL_24:
    if ( !v4 )
      return v2;
    v10 = memcmp(v9, v3, v4);
    if ( !v10 )
    {
LABEL_19:
      if ( v4 <= v8 )
        return v2;
      return 0;
    }
LABEL_26:
    if ( v10 >= 0 )
      return v2;
    return 0;
  }
  if ( v4 < v8 )
    goto LABEL_24;
LABEL_16:
  if ( v8 )
  {
    v10 = memcmp(v9, v3, v8);
    if ( v10 )
      goto LABEL_26;
  }
  if ( v4 != v8 )
    goto LABEL_19;
  return v2;
}
