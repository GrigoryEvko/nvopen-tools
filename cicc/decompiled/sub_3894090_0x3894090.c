// Function: sub_3894090
// Address: 0x3894090
//
__int64 __fastcall sub_3894090(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  const void *v3; // r12
  size_t v4; // r13
  int v5; // eax
  size_t v6; // rdx
  signed __int64 v7; // rax
  signed __int64 v8; // rax
  size_t v9; // r15
  const void *v10; // r14
  __int64 v12; // r14
  __int64 v13; // r15
  size_t v14; // r10
  size_t v15; // rdx
  signed __int64 v16; // rax
  size_t v17; // r14
  size_t v18; // rdx
  int v19; // eax
  __int64 v20; // [rsp+0h] [rbp-40h]
  size_t n; // [rsp+8h] [rbp-38h]
  size_t na; // [rsp+8h] [rbp-38h]

  v2 = *(_QWORD *)(a1 + 16);
  v20 = a1 + 8;
  if ( !v2 )
    return v20;
  v3 = *(const void **)a2;
  v4 = *(_QWORD *)(a2 + 8);
  while ( 1 )
  {
    while ( 1 )
    {
      v9 = *(_QWORD *)(v2 + 40);
      v6 = v4;
      v10 = *(const void **)(v2 + 32);
      if ( v9 <= v4 )
        v6 = *(_QWORD *)(v2 + 40);
      if ( !v6 )
        break;
      n = v6;
      v5 = memcmp(*(const void **)(v2 + 32), v3, v6);
      v6 = n;
      if ( v5 )
      {
        if ( v5 < 0 )
        {
          v2 = *(_QWORD *)(v2 + 24);
          goto LABEL_19;
        }
LABEL_8:
        LODWORD(v8) = memcmp(v3, v10, v6);
        if ( (_DWORD)v8 )
          goto LABEL_11;
        goto LABEL_9;
      }
      v7 = v9 - v4;
      if ( (__int64)(v9 - v4) >= 0x80000000LL )
        goto LABEL_8;
      if ( v7 > (__int64)0xFFFFFFFF7FFFFFFFLL )
        goto LABEL_6;
LABEL_18:
      v2 = *(_QWORD *)(v2 + 24);
LABEL_19:
      if ( !v2 )
        return v20;
    }
    v7 = v9 - v4;
    if ( (__int64)(v9 - v4) >= 0x80000000LL )
      goto LABEL_9;
    if ( v7 <= (__int64)0xFFFFFFFF7FFFFFFFLL )
      goto LABEL_18;
LABEL_6:
    if ( (int)v7 < 0 )
      goto LABEL_18;
    if ( v6 )
      goto LABEL_8;
LABEL_9:
    v8 = v4 - v9;
    if ( (__int64)(v4 - v9) >= 0x80000000LL )
      break;
    if ( v8 <= (__int64)0xFFFFFFFF7FFFFFFFLL )
      goto LABEL_12;
LABEL_11:
    if ( (int)v8 >= 0 )
      break;
LABEL_12:
    v20 = v2;
    v2 = *(_QWORD *)(v2 + 16);
    if ( !v2 )
      return v20;
  }
  v12 = *(_QWORD *)(v2 + 24);
  v13 = *(_QWORD *)(v2 + 16);
  while ( v12 )
  {
    while ( 1 )
    {
      v14 = *(_QWORD *)(v12 + 40);
      v15 = v14;
      if ( v4 <= v14 )
        v15 = v4;
      if ( !v15 )
        break;
      na = *(_QWORD *)(v12 + 40);
      LODWORD(v16) = memcmp(v3, *(const void **)(v12 + 32), v15);
      v14 = na;
      if ( !(_DWORD)v16 )
        break;
LABEL_32:
      if ( (int)v16 >= 0 )
        goto LABEL_33;
LABEL_25:
      v12 = *(_QWORD *)(v12 + 16);
      if ( !v12 )
        goto LABEL_34;
    }
    v16 = v4 - v14;
    if ( (__int64)(v4 - v14) < 0x80000000LL )
    {
      if ( v16 <= (__int64)0xFFFFFFFF7FFFFFFFLL )
        goto LABEL_25;
      goto LABEL_32;
    }
LABEL_33:
    v12 = *(_QWORD *)(v12 + 24);
  }
LABEL_34:
  while ( v13 )
  {
    v17 = *(_QWORD *)(v13 + 40);
    v18 = v17;
    if ( v4 <= v17 )
      v18 = v4;
    if ( !v18 || (v19 = memcmp(*(const void **)(v13 + 32), v3, v18)) == 0 )
    {
      if ( (__int64)(v17 - v4) >= 0x80000000LL )
        goto LABEL_45;
      if ( (__int64)(v17 - v4) <= (__int64)0xFFFFFFFF7FFFFFFFLL )
        goto LABEL_43;
      v19 = v17 - v4;
    }
    if ( v19 >= 0 )
    {
LABEL_45:
      v2 = v13;
      v13 = *(_QWORD *)(v13 + 16);
      continue;
    }
LABEL_43:
    v13 = *(_QWORD *)(v13 + 24);
  }
  return v2;
}
