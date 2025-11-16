// Function: sub_1279E60
// Address: 0x1279e60
//
__int64 __fastcall sub_1279E60(__int64 a1, unsigned __int64 a2, unsigned __int8 a3)
{
  unsigned __int64 i; // rbx
  __int64 v7; // rax
  __int64 v8; // r15
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // r9
  __int64 v12; // r8
  __int64 v13; // rax
  __int64 v14; // r14
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // r14
  _BOOL8 v19; // rdi
  __int64 v21; // [rsp+0h] [rbp-40h]
  __int64 v22; // [rsp+8h] [rbp-38h]

  for ( i = a2; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v7 = *(_QWORD *)(a1 + 72);
  v8 = a1 + 64;
  if ( !v7 )
    goto LABEL_12;
  v9 = a1 + 64;
  do
  {
    while ( *(_QWORD *)(v7 + 32) >= i && (*(_QWORD *)(v7 + 32) != i || a3 <= *(_BYTE *)(v7 + 40)) )
    {
      v9 = v7;
      v7 = *(_QWORD *)(v7 + 16);
      if ( !v7 )
        goto LABEL_10;
    }
    v7 = *(_QWORD *)(v7 + 24);
  }
  while ( v7 );
LABEL_10:
  if ( v8 != v9 && *(_QWORD *)(v9 + 32) <= i && (*(_QWORD *)(v9 + 32) != i || a3 >= *(_BYTE *)(v9 + 40)) )
    return *(_QWORD *)(v9 + 48);
LABEL_12:
  v10 = sub_1279AB0((_QWORD **)a1, i, a3);
  v11 = a1 + 64;
  v12 = v10;
  v13 = *(_QWORD *)(a1 + 72);
  if ( !v13 )
    goto LABEL_23;
  do
  {
    while ( *(_QWORD *)(v13 + 32) >= i && (*(_QWORD *)(v13 + 32) != i || a3 <= *(_BYTE *)(v13 + 40)) )
    {
      v11 = v13;
      v13 = *(_QWORD *)(v13 + 16);
      if ( !v13 )
        goto LABEL_19;
    }
    v13 = *(_QWORD *)(v13 + 24);
  }
  while ( v13 );
LABEL_19:
  if ( v8 == v11 || *(_QWORD *)(v11 + 32) > i || *(_QWORD *)(v11 + 32) == i && a3 < *(_BYTE *)(v11 + 40) )
  {
LABEL_23:
    v14 = v11;
    v21 = v12;
    v15 = sub_22077B0(56);
    *(_QWORD *)(v15 + 32) = i;
    *(_BYTE *)(v15 + 40) = a3;
    *(_QWORD *)(v15 + 48) = 0;
    v22 = v15;
    v16 = sub_1278330((_QWORD *)(a1 + 56), v14, v15 + 32);
    v18 = v16;
    if ( v17 )
    {
      v19 = 1;
      if ( v8 != v17 && !v16 && *(_QWORD *)(v17 + 32) <= i )
      {
        v19 = 0;
        if ( *(_QWORD *)(v17 + 32) == i )
          v19 = a3 < *(_BYTE *)(v17 + 40);
      }
      sub_220F040(v19, v22, v17, a1 + 64);
      ++*(_QWORD *)(a1 + 96);
      v11 = v22;
      v12 = v21;
    }
    else
    {
      j_j___libc_free_0(v22, 56);
      v12 = v21;
      v11 = v18;
    }
  }
  *(_QWORD *)(v11 + 48) = v12;
  return v12;
}
