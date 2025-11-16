// Function: sub_91A1C0
// Address: 0x91a1c0
//
__int64 __fastcall sub_91A1C0(__int64 a1, unsigned __int64 a2, unsigned __int8 a3, __int64 a4)
{
  unsigned __int64 i; // rbx
  __int64 v8; // rax
  __int64 v9; // r15
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // r9
  __int64 v13; // r8
  __int64 v14; // rax
  __int64 v15; // r14
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // r14
  _BOOL8 v20; // rdi
  __int64 v22; // [rsp+0h] [rbp-40h]
  __int64 v23; // [rsp+8h] [rbp-38h]

  for ( i = a2; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v8 = *(_QWORD *)(a1 + 72);
  v9 = a1 + 64;
  if ( !v8 )
    goto LABEL_12;
  v10 = a1 + 64;
  do
  {
    while ( *(_QWORD *)(v8 + 32) >= i && (*(_QWORD *)(v8 + 32) != i || a3 <= *(_BYTE *)(v8 + 40)) )
    {
      v10 = v8;
      v8 = *(_QWORD *)(v8 + 16);
      if ( !v8 )
        goto LABEL_10;
    }
    v8 = *(_QWORD *)(v8 + 24);
  }
  while ( v8 );
LABEL_10:
  if ( v9 != v10 && *(_QWORD *)(v10 + 32) <= i && (*(_QWORD *)(v10 + 32) != i || a3 >= *(_BYTE *)(v10 + 40)) )
    return *(_QWORD *)(v10 + 48);
LABEL_12:
  v11 = sub_919DE0((_QWORD **)a1, i, a3, a4);
  v12 = a1 + 64;
  v13 = v11;
  v14 = *(_QWORD *)(a1 + 72);
  if ( !v14 )
    goto LABEL_23;
  do
  {
    while ( *(_QWORD *)(v14 + 32) >= i && (*(_QWORD *)(v14 + 32) != i || a3 <= *(_BYTE *)(v14 + 40)) )
    {
      v12 = v14;
      v14 = *(_QWORD *)(v14 + 16);
      if ( !v14 )
        goto LABEL_19;
    }
    v14 = *(_QWORD *)(v14 + 24);
  }
  while ( v14 );
LABEL_19:
  if ( v9 == v12 || *(_QWORD *)(v12 + 32) > i || *(_QWORD *)(v12 + 32) == i && a3 < *(_BYTE *)(v12 + 40) )
  {
LABEL_23:
    v15 = v12;
    v22 = v13;
    v16 = sub_22077B0(56);
    *(_QWORD *)(v16 + 32) = i;
    *(_BYTE *)(v16 + 40) = a3;
    *(_QWORD *)(v16 + 48) = 0;
    v23 = v16;
    v17 = sub_918690((_QWORD *)(a1 + 56), v15, v16 + 32);
    v19 = v17;
    if ( v18 )
    {
      v20 = 1;
      if ( v9 != v18 && !v17 && *(_QWORD *)(v18 + 32) <= i )
      {
        v20 = 0;
        if ( *(_QWORD *)(v18 + 32) == i )
          v20 = a3 < *(_BYTE *)(v18 + 40);
      }
      sub_220F040(v20, v23, v18, a1 + 64);
      ++*(_QWORD *)(a1 + 96);
      v12 = v23;
      v13 = v22;
    }
    else
    {
      j_j___libc_free_0(v23, 56);
      v13 = v22;
      v12 = v19;
    }
  }
  *(_QWORD *)(v12 + 48) = v13;
  return v13;
}
