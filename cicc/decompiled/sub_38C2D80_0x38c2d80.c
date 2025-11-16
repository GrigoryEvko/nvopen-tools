// Function: sub_38C2D80
// Address: 0x38c2d80
//
__int64 __fastcall sub_38C2D80(__int64 a1, __int64 a2)
{
  __int64 v2; // r15
  size_t v3; // r14
  size_t v5; // rdx
  signed __int64 v6; // rax
  size_t v7; // rdx
  signed __int64 v8; // rax
  size_t v9; // r12
  const void *v10; // r13
  size_t v11; // r12
  size_t v12; // r9
  const void *v13; // rdi
  const void *v14; // rsi
  int v15; // eax
  size_t v17; // r12
  size_t v18; // r8
  const void *v19; // r13
  const void *v20; // rsi
  int v21; // eax
  __int64 v22; // r14
  __int64 v23; // r13
  char v24; // al
  __int64 v25; // rcx
  __int64 v26; // rdx
  char v27; // al
  __int64 v28; // rdx
  size_t v29; // [rsp+18h] [rbp-48h]
  size_t v30; // [rsp+18h] [rbp-48h]
  size_t v31; // [rsp+18h] [rbp-48h]
  size_t v32; // [rsp+18h] [rbp-48h]
  size_t v33; // [rsp+18h] [rbp-48h]
  size_t v34; // [rsp+18h] [rbp-48h]
  void *s2; // [rsp+20h] [rbp-40h]
  __int64 v36; // [rsp+28h] [rbp-38h]

  v2 = *(_QWORD *)(a1 + 16);
  v36 = a1 + 8;
  if ( !v2 )
    return v36;
  v3 = *(_QWORD *)(a2 + 8);
  s2 = *(void **)a2;
  while ( 1 )
  {
    while ( 1 )
    {
      v9 = *(_QWORD *)(v2 + 40);
      v10 = *(const void **)(v2 + 32);
      if ( v9 == v3 )
        break;
      v5 = v3;
      if ( v9 <= v3 )
        v5 = *(_QWORD *)(v2 + 40);
      if ( v5 )
      {
        LODWORD(v6) = memcmp(*(const void **)(v2 + 32), s2, v5);
        if ( (_DWORD)v6 )
          goto LABEL_9;
      }
      v6 = v9 - v3;
      if ( (__int64)(v9 - v3) >= 0x80000000LL )
        goto LABEL_11;
      if ( v6 > (__int64)0xFFFFFFFF7FFFFFFFLL )
        goto LABEL_9;
LABEL_28:
      v2 = *(_QWORD *)(v2 + 24);
      if ( !v2 )
        return v36;
    }
    if ( !v3 || (LODWORD(v6) = memcmp(*(const void **)(v2 + 32), s2, v3), !(_DWORD)v6) )
    {
      v11 = *(_QWORD *)(v2 + 72);
      v12 = *(_QWORD *)(a2 + 40);
      v13 = *(const void **)(v2 + 64);
      v14 = *(const void **)(a2 + 32);
      if ( v11 == v12 )
      {
        v30 = *(_QWORD *)(a2 + 40);
        if ( !v11 || !memcmp(v13, v14, v11) )
        {
          if ( *(_DWORD *)(v2 + 80) < *(_DWORD *)(a2 + 48) )
            goto LABEL_28;
          goto LABEL_34;
        }
        v12 = v30;
      }
      else if ( v11 > v12 )
      {
        if ( !v12 )
          goto LABEL_34;
        v32 = *(_QWORD *)(a2 + 40);
        v15 = memcmp(v13, v14, v32);
        v12 = v32;
        if ( v15 )
        {
LABEL_53:
          if ( v15 < 0 )
            goto LABEL_28;
          goto LABEL_34;
        }
        goto LABEL_27;
      }
      if ( v11 )
      {
        v29 = v12;
        v15 = memcmp(v13, v14, v11);
        v12 = v29;
        if ( v15 )
          goto LABEL_53;
      }
      if ( v11 == v12 )
        goto LABEL_34;
LABEL_27:
      if ( v11 < v12 )
        goto LABEL_28;
      goto LABEL_34;
    }
LABEL_9:
    if ( (int)v6 < 0 )
      goto LABEL_28;
    if ( v9 != v3 )
    {
LABEL_11:
      v7 = v3;
      if ( v9 <= v3 )
        v7 = v9;
      if ( !v7 || (LODWORD(v8) = memcmp(s2, v10, v7), !(_DWORD)v8) )
      {
        v8 = v3 - v9;
        if ( (__int64)(v3 - v9) >= 0x80000000LL )
          goto LABEL_42;
        if ( v8 <= (__int64)0xFFFFFFFF7FFFFFFFLL )
          goto LABEL_18;
      }
LABEL_17:
      if ( (int)v8 >= 0 )
        goto LABEL_42;
      goto LABEL_18;
    }
LABEL_34:
    if ( v3 )
    {
      LODWORD(v8) = memcmp(s2, v10, v3);
      if ( (_DWORD)v8 )
        goto LABEL_17;
    }
    v17 = *(_QWORD *)(a2 + 40);
    v18 = *(_QWORD *)(v2 + 72);
    v19 = *(const void **)(a2 + 32);
    v20 = *(const void **)(v2 + 64);
    if ( v17 != v18 )
      break;
    v33 = *(_QWORD *)(v2 + 72);
    if ( v17 && memcmp(v19, v20, v17) )
    {
      v18 = v33;
      goto LABEL_38;
    }
    if ( *(_DWORD *)(a2 + 48) >= *(_DWORD *)(v2 + 80) )
      goto LABEL_42;
LABEL_18:
    v36 = v2;
    v2 = *(_QWORD *)(v2 + 16);
    if ( !v2 )
      return v36;
  }
  if ( v17 <= v18 )
  {
LABEL_38:
    if ( v17 )
    {
      v31 = v18;
      v21 = memcmp(v19, v20, v17);
      v18 = v31;
      if ( v21 )
        goto LABEL_61;
    }
    if ( v17 == v18 )
      goto LABEL_42;
    goto LABEL_41;
  }
  if ( !v18 )
    goto LABEL_42;
  v34 = *(_QWORD *)(v2 + 72);
  v21 = memcmp(*(const void **)(a2 + 32), v20, v34);
  v18 = v34;
  if ( !v21 )
  {
LABEL_41:
    if ( v17 >= v18 )
      goto LABEL_42;
    goto LABEL_18;
  }
LABEL_61:
  if ( v21 < 0 )
    goto LABEL_18;
LABEL_42:
  v22 = *(_QWORD *)(v2 + 24);
  v23 = *(_QWORD *)(v2 + 16);
  if ( v22 )
  {
    do
    {
      while ( 1 )
      {
        v24 = sub_38BC8E0(a2, v22 + 32);
        v25 = *(_QWORD *)(v22 + 16);
        v26 = *(_QWORD *)(v22 + 24);
        if ( v24 )
          break;
        v22 = *(_QWORD *)(v22 + 24);
        if ( !v26 )
          goto LABEL_47;
      }
      v22 = *(_QWORD *)(v22 + 16);
    }
    while ( v25 );
  }
LABEL_47:
  while ( v23 )
  {
    while ( 1 )
    {
      v27 = sub_38BC8E0(v23 + 32, a2);
      v28 = *(_QWORD *)(v23 + 16);
      if ( v27 )
        break;
      v2 = v23;
      v23 = *(_QWORD *)(v23 + 16);
      if ( !v28 )
        return v2;
    }
    v23 = *(_QWORD *)(v23 + 24);
  }
  return v2;
}
