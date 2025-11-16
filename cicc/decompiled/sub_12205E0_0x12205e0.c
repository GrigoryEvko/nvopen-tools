// Function: sub_12205E0
// Address: 0x12205e0
//
__int64 __fastcall sub_12205E0(__int64 a1, __int64 a2)
{
  __int64 v3; // r15
  unsigned int v4; // r13d
  unsigned int v5; // eax
  __int64 v6; // rdx
  __int64 v7; // rsi
  int v8; // r12d
  size_t v9; // rbx
  size_t v10; // r10
  size_t v11; // rdx
  unsigned int v12; // eax
  __int64 v13; // rbx
  __int64 v15; // rax
  size_t v16; // r12
  size_t v17; // r13
  size_t v18; // rdx
  int v19; // eax
  __int64 v20; // r12
  size_t v21; // [rsp+8h] [rbp-38h]

  v3 = *(_QWORD *)(a1 + 16);
  if ( !v3 )
  {
    v3 = a1 + 8;
    goto LABEL_24;
  }
  v4 = *(_DWORD *)a2;
  while ( 1 )
  {
    v8 = *(_DWORD *)(v3 + 32);
    if ( v4 != v8 )
    {
      LOBYTE(v5) = (int)v4 < v8;
      goto LABEL_4;
    }
    if ( v4 <= 1 )
    {
      LOBYTE(v5) = *(_DWORD *)(a2 + 16) < *(_DWORD *)(v3 + 48);
LABEL_4:
      v6 = *(_QWORD *)(v3 + 16);
      v7 = *(_QWORD *)(v3 + 24);
      if ( (_BYTE)v5 )
        goto LABEL_6;
      goto LABEL_5;
    }
    v9 = *(_QWORD *)(a2 + 40);
    v10 = *(_QWORD *)(v3 + 72);
    v11 = v10;
    if ( v9 <= v10 )
      v11 = *(_QWORD *)(a2 + 40);
    if ( v11 )
    {
      v21 = *(_QWORD *)(v3 + 72);
      v12 = memcmp(*(const void **)(a2 + 32), *(const void **)(v3 + 64), v11);
      v10 = v21;
      if ( v12 )
      {
        v5 = v12 >> 31;
        goto LABEL_4;
      }
    }
    v13 = v9 - v10;
    if ( v13 >= 0x80000000LL )
    {
      v7 = *(_QWORD *)(v3 + 24);
LABEL_5:
      v6 = v7;
      LOBYTE(v5) = 0;
      goto LABEL_6;
    }
    if ( v13 > (__int64)0xFFFFFFFF7FFFFFFFLL )
    {
      LOBYTE(v5) = (int)v13 < 0;
      goto LABEL_4;
    }
    v6 = *(_QWORD *)(v3 + 16);
    LOBYTE(v5) = 1;
LABEL_6:
    if ( !v6 )
      break;
    v3 = v6;
  }
  if ( (_BYTE)v5 )
  {
LABEL_24:
    if ( *(_QWORD *)(a1 + 24) == v3 )
      return 0;
    v15 = sub_220EF80(v3);
    v4 = *(_DWORD *)a2;
    v8 = *(_DWORD *)(v15 + 32);
    v3 = v15;
    if ( *(_DWORD *)a2 == v8 )
      goto LABEL_26;
LABEL_21:
    if ( v8 < (int)v4 )
      return 0;
    return v3;
  }
  if ( v4 != v8 )
    goto LABEL_21;
LABEL_26:
  if ( v4 <= 1 )
  {
    if ( *(_DWORD *)(v3 + 48) < *(_DWORD *)(a2 + 16) )
      return 0;
    return v3;
  }
  v16 = *(_QWORD *)(v3 + 72);
  v17 = *(_QWORD *)(a2 + 40);
  v18 = v17;
  if ( v16 <= v17 )
    v18 = *(_QWORD *)(v3 + 72);
  if ( v18 )
  {
    v19 = memcmp(*(const void **)(v3 + 64), *(const void **)(a2 + 32), v18);
    if ( v19 )
    {
LABEL_34:
      if ( v19 < 0 )
        return 0;
      return v3;
    }
  }
  v20 = v16 - v17;
  if ( v20 > 0x7FFFFFFF )
    return v3;
  if ( v20 >= (__int64)0xFFFFFFFF80000000LL )
  {
    v19 = v20;
    goto LABEL_34;
  }
  return 0;
}
