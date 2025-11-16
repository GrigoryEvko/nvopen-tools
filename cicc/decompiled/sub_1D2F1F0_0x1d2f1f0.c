// Function: sub_1D2F1F0
// Address: 0x1d2f1f0
//
__int64 __fastcall sub_1D2F1F0(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  const void *v3; // r12
  size_t v4; // r15
  int v5; // eax
  void *v6; // r9
  signed __int64 v7; // rax
  signed __int64 v8; // rax
  __int64 v9; // rax
  char v10; // si
  size_t v11; // r14
  size_t v12; // r13
  int v13; // eax
  signed __int64 v14; // rax
  int v15; // eax
  __int64 v17; // rax
  void *s2; // [rsp+8h] [rbp-48h]
  void *s1; // [rsp+18h] [rbp-38h]
  void *s1a; // [rsp+18h] [rbp-38h]

  v2 = *(_QWORD *)(a1 + 16);
  if ( !v2 )
  {
    v2 = a1 + 8;
    goto LABEL_38;
  }
  v3 = *(const void **)a2;
  v4 = *(_QWORD *)(a2 + 8);
  while ( 1 )
  {
    v11 = *(_QWORD *)(v2 + 40);
    v6 = *(void **)(v2 + 32);
    v12 = v11;
    if ( v4 <= v11 )
      v12 = v4;
    if ( !v12 )
    {
      v7 = v4 - v11;
      if ( (__int64)(v4 - v11) >= 0x80000000LL )
        goto LABEL_9;
      goto LABEL_5;
    }
    s1 = *(void **)(v2 + 32);
    v5 = memcmp(v3, s1, v12);
    v6 = s1;
    if ( !v5 )
      break;
    if ( v5 >= 0 )
      goto LABEL_8;
LABEL_13:
    v9 = *(_QWORD *)(v2 + 16);
    v10 = 1;
    if ( !v9 )
      goto LABEL_21;
LABEL_14:
    v2 = v9;
  }
  v7 = v4 - v11;
  if ( (__int64)(v4 - v11) >= 0x80000000LL )
    goto LABEL_8;
LABEL_5:
  if ( v7 <= (__int64)0xFFFFFFFF7FFFFFFFLL || (int)v7 < 0 )
    goto LABEL_13;
  if ( !v12 )
    goto LABEL_9;
LABEL_8:
  s1a = v6;
  LODWORD(v8) = memcmp(v6, v3, v12);
  v6 = s1a;
  if ( (_DWORD)v8 )
    goto LABEL_11;
LABEL_9:
  v8 = v11 - v4;
  if ( (__int64)(v11 - v4) >= 0x80000000LL )
    goto LABEL_12;
  if ( v8 <= (__int64)0xFFFFFFFF7FFFFFFFLL )
    goto LABEL_20;
LABEL_11:
  if ( (int)v8 < 0 )
    goto LABEL_20;
LABEL_12:
  if ( *(_BYTE *)(a2 + 32) < *(_BYTE *)(v2 + 64) )
    goto LABEL_13;
LABEL_20:
  v9 = *(_QWORD *)(v2 + 24);
  v10 = 0;
  if ( v9 )
    goto LABEL_14;
LABEL_21:
  if ( v10 )
  {
LABEL_38:
    if ( v2 == *(_QWORD *)(a1 + 24) )
      return 0;
    v17 = sub_220EF80(v2);
    v11 = *(_QWORD *)(v17 + 40);
    v6 = *(void **)(v17 + 32);
    v2 = v17;
    v4 = *(_QWORD *)(a2 + 8);
    v3 = *(const void **)a2;
    v12 = v4;
    if ( v11 <= v4 )
      v12 = *(_QWORD *)(v17 + 40);
    if ( !v12 )
      goto LABEL_42;
LABEL_23:
    s2 = v6;
    v13 = memcmp(v6, v3, v12);
    v6 = s2;
    if ( v13 )
    {
      if ( v13 < 0 )
        return 0;
    }
    else
    {
      v14 = v11 - v4;
      if ( (__int64)(v11 - v4) <= 0x7FFFFFFF )
      {
LABEL_25:
        if ( v14 < (__int64)0xFFFFFFFF80000000LL || (int)v14 < 0 )
          return 0;
        if ( !v12 )
          goto LABEL_29;
      }
    }
    v15 = memcmp(v3, v6, v12);
    if ( !v15 )
      goto LABEL_29;
LABEL_32:
    if ( v15 >= 0 )
      goto LABEL_33;
    return v2;
  }
  if ( v12 )
    goto LABEL_23;
LABEL_42:
  v14 = v11 - v4;
  if ( (__int64)(v11 - v4) <= 0x7FFFFFFF )
    goto LABEL_25;
LABEL_29:
  if ( (__int64)(v4 - v11) <= 0x7FFFFFFF )
  {
    if ( (__int64)(v4 - v11) >= (__int64)0xFFFFFFFF80000000LL )
    {
      v15 = v4 - v11;
      goto LABEL_32;
    }
    return v2;
  }
LABEL_33:
  if ( *(_BYTE *)(v2 + 64) >= *(_BYTE *)(a2 + 32) )
    return v2;
  return 0;
}
