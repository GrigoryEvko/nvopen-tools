// Function: sub_15621F0
// Address: 0x15621f0
//
__int64 __fastcall sub_15621F0(__int64 a1, __int64 a2)
{
  __int64 v2; // r15
  const void *v3; // rbx
  size_t v4; // r14
  __int64 v5; // rax
  char v6; // si
  size_t v7; // r12
  const void *v8; // r13
  size_t v9; // rdx
  signed __int64 v10; // rax
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
      LODWORD(v10) = memcmp(v3, *(const void **)(v2 + 32), v9);
      v9 = n;
      if ( (_DWORD)v10 )
        goto LABEL_11;
    }
    v10 = v4 - v7;
    if ( (__int64)(v4 - v7) >= 0x80000000LL )
      break;
    if ( v10 > (__int64)0xFFFFFFFF7FFFFFFFLL )
    {
LABEL_11:
      if ( (int)v10 >= 0 )
        break;
    }
    v5 = *(_QWORD *)(v2 + 16);
    v6 = 1;
    if ( !v5 )
      goto LABEL_13;
LABEL_4:
    v2 = v5;
  }
  v5 = *(_QWORD *)(v2 + 24);
  v6 = 0;
  if ( v5 )
    goto LABEL_4;
LABEL_13:
  if ( !v6 )
    goto LABEL_14;
LABEL_22:
  if ( v2 == *(_QWORD *)(a1 + 24) )
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
  if ( v9 )
  {
    v11 = memcmp(v8, v3, v9);
    if ( v11 )
    {
LABEL_19:
      if ( v11 < 0 )
        return 0;
      return v2;
    }
  }
  if ( (__int64)(v7 - v4) > 0x7FFFFFFF )
    return v2;
  if ( (__int64)(v7 - v4) >= (__int64)0xFFFFFFFF80000000LL )
  {
    v11 = v7 - v4;
    goto LABEL_19;
  }
  return 0;
}
