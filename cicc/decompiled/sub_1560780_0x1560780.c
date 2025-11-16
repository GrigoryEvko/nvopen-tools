// Function: sub_1560780
// Address: 0x1560780
//
__int64 __fastcall sub_1560780(__int64 a1, _BYTE *a2, __int64 a3)
{
  _QWORD *v3; // r15
  __int64 v4; // rbx
  size_t v5; // r14
  __int64 v6; // r12
  size_t v7; // r13
  size_t v8; // rdx
  int v9; // eax
  size_t v10; // rbx
  size_t v11; // rdx
  int v12; // eax
  __int64 v13; // rax
  __int64 v14; // rdi
  __int64 v15; // r12
  __int64 v16; // rdi
  __int64 v19; // [rsp+18h] [rbp-58h]
  void *s2; // [rsp+20h] [rbp-50h] BYREF
  size_t v21; // [rsp+28h] [rbp-48h]
  _QWORD v22[8]; // [rsp+30h] [rbp-40h] BYREF

  if ( a2 )
  {
    s2 = v22;
    sub_155C9E0((__int64 *)&s2, a2, (__int64)&a2[a3]);
    v3 = s2;
    v4 = *(_QWORD *)(a1 + 24);
    v19 = a1 + 16;
    if ( !v4 )
      goto LABEL_31;
  }
  else
  {
    v4 = *(_QWORD *)(a1 + 24);
    v3 = v22;
    LOBYTE(v22[0]) = 0;
    s2 = v22;
    v21 = 0;
    v19 = a1 + 16;
    if ( !v4 )
      return a1;
  }
  v5 = v21;
  v6 = v19;
  do
  {
    while ( 1 )
    {
      v7 = *(_QWORD *)(v4 + 40);
      v8 = v5;
      if ( v7 <= v5 )
        v8 = *(_QWORD *)(v4 + 40);
      if ( v8 )
      {
        v9 = memcmp(*(const void **)(v4 + 32), v3, v8);
        if ( v9 )
          break;
      }
      if ( (__int64)(v7 - v5) >= 0x80000000LL )
        goto LABEL_13;
      if ( (__int64)(v7 - v5) > (__int64)0xFFFFFFFF7FFFFFFFLL )
      {
        v9 = v7 - v5;
        break;
      }
LABEL_4:
      v4 = *(_QWORD *)(v4 + 24);
      if ( !v4 )
        goto LABEL_14;
    }
    if ( v9 < 0 )
      goto LABEL_4;
LABEL_13:
    v6 = v4;
    v4 = *(_QWORD *)(v4 + 16);
  }
  while ( v4 );
LABEL_14:
  if ( v6 == v19 )
    goto LABEL_31;
  v10 = *(_QWORD *)(v6 + 40);
  v11 = v5;
  if ( v10 <= v5 )
    v11 = *(_QWORD *)(v6 + 40);
  if ( v11 )
  {
    v12 = memcmp(v3, *(const void **)(v6 + 32), v11);
    if ( v12 )
    {
LABEL_22:
      if ( v12 < 0 )
        goto LABEL_31;
      goto LABEL_23;
    }
  }
  if ( (__int64)(v5 - v10) > 0x7FFFFFFF )
  {
LABEL_23:
    if ( v3 != v22 )
      j_j___libc_free_0(v3, v22[0] + 1LL);
    v13 = sub_220F330(v6, v19);
    v14 = *(_QWORD *)(v13 + 64);
    v15 = v13;
    if ( v14 != v13 + 80 )
      j_j___libc_free_0(v14, *(_QWORD *)(v13 + 80) + 1LL);
    v16 = *(_QWORD *)(v15 + 32);
    if ( v16 != v15 + 48 )
      j_j___libc_free_0(v16, *(_QWORD *)(v15 + 48) + 1LL);
    j_j___libc_free_0(v15, 96);
    --*(_QWORD *)(a1 + 48);
    return a1;
  }
  if ( (__int64)(v5 - v10) >= (__int64)0xFFFFFFFF80000000LL )
  {
    v12 = v5 - v10;
    goto LABEL_22;
  }
LABEL_31:
  if ( v3 != v22 )
    j_j___libc_free_0(v3, v22[0] + 1LL);
  return a1;
}
