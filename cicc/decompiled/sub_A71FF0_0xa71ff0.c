// Function: sub_A71FF0
// Address: 0xa71ff0
//
__int64 __fastcall sub_A71FF0(__int64 a1, __int64 a2)
{
  unsigned int v2; // r13d
  unsigned int v3; // eax
  __int64 v4; // rax
  __int64 v5; // r12
  const void *v6; // r14
  size_t v7; // rdx
  size_t v8; // r15
  size_t v9; // rbx
  size_t v10; // rdx
  int v11; // eax
  __int64 v12; // r12
  size_t v13; // rbx
  size_t v14; // rdx
  int v15; // eax
  unsigned int v16; // eax
  __int64 v18; // [rsp+8h] [rbp-48h]
  __int64 v19; // [rsp+10h] [rbp-40h]
  __int64 v20[7]; // [rsp+18h] [rbp-38h] BYREF

  v20[0] = a2;
  LOBYTE(v3) = sub_A71840((__int64)v20);
  if ( !(_BYTE)v3 )
  {
    v16 = sub_A71AE0(v20);
    LOBYTE(v2) = (*(_QWORD *)(a1 + 8 * ((unsigned __int64)v16 >> 6)) & (1LL << v16)) != 0;
    return v2;
  }
  v2 = v3;
  v4 = sub_A71FD0(v20);
  v5 = *(_QWORD *)(a1 + 32);
  v6 = (const void *)v4;
  v8 = v7;
  v18 = a1 + 24;
  if ( !v5 )
    return 0;
  v19 = a1 + 24;
  do
  {
    while ( 1 )
    {
      v9 = *(_QWORD *)(v5 + 40);
      v10 = v8;
      if ( v9 <= v8 )
        v10 = *(_QWORD *)(v5 + 40);
      if ( v10 )
      {
        v11 = memcmp(*(const void **)(v5 + 32), v6, v10);
        if ( v11 )
          break;
      }
      if ( v9 == v8 || v9 >= v8 )
      {
        v19 = v5;
        v5 = *(_QWORD *)(v5 + 16);
        goto LABEL_12;
      }
LABEL_4:
      v5 = *(_QWORD *)(v5 + 24);
      if ( !v5 )
        goto LABEL_13;
    }
    if ( v11 < 0 )
      goto LABEL_4;
    v19 = v5;
    v5 = *(_QWORD *)(v5 + 16);
LABEL_12:
    ;
  }
  while ( v5 );
LABEL_13:
  if ( v18 == v19 )
    return 0;
  v12 = v19;
  while ( 1 )
  {
    v13 = *(_QWORD *)(v12 + 40);
    v14 = v8;
    if ( v13 <= v8 )
      v14 = *(_QWORD *)(v12 + 40);
    if ( !v14 )
      break;
    v15 = memcmp(v6, *(const void **)(v12 + 32), v14);
    if ( !v15 )
      break;
    if ( v15 < 0 )
      goto LABEL_21;
LABEL_29:
    v12 = sub_220EF30(v12);
    if ( v12 == v18 )
      return v2;
  }
  if ( v13 == v8 || v13 <= v8 )
    goto LABEL_29;
LABEL_21:
  if ( v19 == v12 )
    return 0;
  return v2;
}
