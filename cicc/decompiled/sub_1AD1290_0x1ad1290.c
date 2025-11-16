// Function: sub_1AD1290
// Address: 0x1ad1290
//
unsigned __int64 __fastcall sub_1AD1290(__int64 a1, __int64 a2)
{
  unsigned __int8 *v3; // rax
  size_t v4; // rdx
  size_t v5; // r13
  unsigned __int8 *v6; // r15
  __int64 v7; // rdx
  _QWORD *v8; // r9
  __int64 v9; // rbx
  unsigned __int64 v10; // r12
  __int64 v12; // rax
  unsigned int v13; // r8d
  _QWORD *v14; // r9
  _QWORD *v15; // rcx
  void *v16; // rdi
  __int64 *v17; // rax
  _QWORD *v18; // rax
  unsigned __int64 *v19; // r12
  __int64 v20; // rax
  void *v21; // rax
  _QWORD *v22; // [rsp+0h] [rbp-50h]
  _QWORD *v23; // [rsp+8h] [rbp-48h]
  _QWORD *v24; // [rsp+8h] [rbp-48h]
  _QWORD *v25; // [rsp+8h] [rbp-48h]
  unsigned int v26; // [rsp+10h] [rbp-40h]
  _QWORD *v27; // [rsp+10h] [rbp-40h]
  unsigned int v28; // [rsp+10h] [rbp-40h]
  unsigned int v29; // [rsp+18h] [rbp-38h]

  v3 = (unsigned __int8 *)sub_1649960(a2);
  v5 = v4;
  v6 = v3;
  v7 = (unsigned int)sub_16D19C0(a1, v3, v4);
  v8 = (_QWORD *)(*(_QWORD *)a1 + 8 * v7);
  v9 = *v8;
  if ( *v8 )
  {
    if ( v9 != -8 )
    {
      v10 = *(_QWORD *)(v9 + 8);
      if ( v10 )
        return v10;
      goto LABEL_12;
    }
    --*(_DWORD *)(a1 + 16);
  }
  v23 = v8;
  v26 = v7;
  v12 = malloc(v5 + 17);
  v13 = v26;
  v14 = v23;
  v15 = (_QWORD *)v12;
  if ( v12 )
  {
LABEL_7:
    v16 = v15 + 2;
    if ( v5 + 1 <= 1 )
      goto LABEL_8;
    goto LABEL_21;
  }
  if ( v5 != -17 || (v20 = malloc(1u), v13 = v26, v14 = v23, v15 = 0, !v20) )
  {
    v22 = v15;
    v25 = v14;
    v28 = v13;
    sub_16BD1C0("Allocation failed", 1u);
    v13 = v28;
    v14 = v25;
    v15 = v22;
    goto LABEL_7;
  }
  v16 = (void *)(v20 + 16);
  v15 = (_QWORD *)v20;
LABEL_21:
  v24 = v15;
  v27 = v14;
  v29 = v13;
  v21 = memcpy(v16, v6, v5);
  v15 = v24;
  v14 = v27;
  v13 = v29;
  v16 = v21;
LABEL_8:
  *((_BYTE *)v16 + v5) = 0;
  *v15 = v5;
  v15[1] = 0;
  *v14 = v15;
  ++*(_DWORD *)(a1 + 12);
  v17 = (__int64 *)(*(_QWORD *)a1 + 8LL * (unsigned int)sub_16D1CD0(a1, v13));
  v9 = *v17;
  if ( *v17 )
    goto LABEL_10;
  do
  {
    do
    {
      v9 = v17[1];
      ++v17;
    }
    while ( !v9 );
LABEL_10:
    ;
  }
  while ( v9 == -8 );
  v10 = *(_QWORD *)(v9 + 8);
  if ( !v10 )
  {
LABEL_12:
    v18 = (_QWORD *)sub_22077B0(96);
    if ( v18 )
    {
      memset64(v18, v10, 0xCu);
      *v18 = v18 + 2;
      v18[1] = 0x800000000LL;
    }
    v19 = *(unsigned __int64 **)(v9 + 8);
    *(_QWORD *)(v9 + 8) = v18;
    if ( v19 )
    {
      if ( (unsigned __int64 *)*v19 != v19 + 2 )
        _libc_free(*v19);
      j_j___libc_free_0(v19, 96);
    }
    *(_BYTE *)(*(_QWORD *)(v9 + 8) + 88LL) = sub_1626CE0(a2, "thinlto_src_module", 0x12u) != 0;
    return *(_QWORD *)(v9 + 8);
  }
  return v10;
}
