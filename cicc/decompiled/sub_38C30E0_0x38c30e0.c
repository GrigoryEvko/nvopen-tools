// Function: sub_38C30E0
// Address: 0x38c30e0
//
__int64 __fastcall sub_38C30E0(__int64 a1, __int64 a2)
{
  __int64 v3; // rbx
  const void *v4; // r13
  size_t v5; // r15
  size_t v6; // rdx
  signed __int64 v7; // rax
  __int64 v8; // rax
  char v9; // si
  size_t v10; // r14
  const void *v11; // rsi
  size_t v12; // r9
  size_t v13; // r14
  const void *v14; // rdi
  const void *v15; // rsi
  bool v16; // zf
  __int64 result; // rax
  size_t v18; // [rsp+0h] [rbp-50h]
  size_t v20; // [rsp+18h] [rbp-38h]
  size_t v21; // [rsp+18h] [rbp-38h]

  v3 = *(_QWORD *)(a1 + 16);
  if ( !v3 )
  {
    v3 = a1 + 8;
    goto LABEL_35;
  }
  v4 = *(const void **)a2;
  v5 = *(_QWORD *)(a2 + 8);
  while ( 1 )
  {
    v10 = *(_QWORD *)(v3 + 40);
    v11 = *(const void **)(v3 + 32);
    if ( v5 != v10 )
    {
      v6 = *(_QWORD *)(v3 + 40);
      if ( v5 <= v10 )
        v6 = v5;
      if ( !v6 || (LODWORD(v7) = memcmp(v4, v11, v6), !(_DWORD)v7) )
      {
        v7 = v5 - v10;
        if ( (__int64)(v5 - v10) >= 0x80000000LL )
          goto LABEL_21;
        if ( v7 <= (__int64)0xFFFFFFFF7FFFFFFFLL )
          goto LABEL_10;
      }
LABEL_9:
      if ( (int)v7 >= 0 )
        goto LABEL_21;
      goto LABEL_10;
    }
    if ( v5 )
    {
      LODWORD(v7) = memcmp(v4, v11, v5);
      if ( (_DWORD)v7 )
        goto LABEL_9;
    }
    v12 = *(_QWORD *)(a2 + 40);
    v13 = *(_QWORD *)(v3 + 72);
    v14 = *(const void **)(a2 + 32);
    v15 = *(const void **)(v3 + 64);
    if ( v12 == v13 )
      break;
    if ( v12 <= v13 )
      goto LABEL_17;
    v21 = *(_QWORD *)(a2 + 40);
    if ( !v13 )
    {
LABEL_21:
      v8 = *(_QWORD *)(v3 + 24);
      v9 = 0;
      goto LABEL_22;
    }
    LODWORD(v7) = memcmp(v14, v15, v13);
    v12 = v21;
    if ( (_DWORD)v7 )
      goto LABEL_9;
LABEL_20:
    if ( v12 >= v13 )
      goto LABEL_21;
LABEL_10:
    v8 = *(_QWORD *)(v3 + 16);
    v9 = 1;
    if ( !v8 )
      goto LABEL_23;
LABEL_11:
    v3 = v8;
  }
  if ( v12 )
  {
    v18 = *(_QWORD *)(a2 + 40);
    if ( memcmp(v14, v15, v18) )
    {
      v12 = v18;
LABEL_17:
      if ( v12 )
      {
        v20 = v12;
        LODWORD(v7) = memcmp(v14, v15, v12);
        v12 = v20;
        if ( (_DWORD)v7 )
          goto LABEL_9;
      }
      if ( v12 == v13 )
        goto LABEL_21;
      goto LABEL_20;
    }
  }
  if ( *(_DWORD *)(a2 + 48) < *(_DWORD *)(v3 + 80) )
    goto LABEL_10;
  v8 = *(_QWORD *)(v3 + 24);
  v9 = 0;
LABEL_22:
  if ( v8 )
    goto LABEL_11;
LABEL_23:
  if ( !v9 )
  {
LABEL_24:
    v16 = sub_38BC8E0(v3 + 32, a2) == 0;
    result = v3;
    if ( !v16 )
      return 0;
    return result;
  }
LABEL_35:
  if ( v3 != *(_QWORD *)(a1 + 24) )
  {
    v3 = sub_220EF80(v3);
    goto LABEL_24;
  }
  return 0;
}
