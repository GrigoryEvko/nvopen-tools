// Function: sub_164D1F0
// Address: 0x164d1f0
//
__int64 __fastcall sub_164D1F0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 i; // rax
  _BYTE *v5; // rax
  _BYTE *j; // rdx
  size_t v7; // r14
  __int64 v8; // rsi
  _QWORD *v9; // r13
  unsigned int v10; // r8d
  unsigned int v11; // r9d
  _QWORD *v12; // rcx
  _QWORD *v13; // rax
  size_t v14; // r13
  const void *v15; // r14
  __int64 v16; // rax
  unsigned int v17; // r9d
  _QWORD *v18; // rcx
  _QWORD *v19; // r12
  __int64 *v20; // rbx
  __int64 *v21; // rax
  __int64 v22; // rdx
  __int64 v23; // r12
  size_t v25; // rdx
  _QWORD *v26; // rax
  unsigned int v27; // [rsp+4h] [rbp-1CCh]
  unsigned int v28; // [rsp+4h] [rbp-1CCh]
  unsigned int v29; // [rsp+10h] [rbp-1C0h]
  _QWORD *v30; // [rsp+10h] [rbp-1C0h]
  _QWORD *v31; // [rsp+10h] [rbp-1C0h]
  unsigned int v32; // [rsp+18h] [rbp-1B8h]
  unsigned int v33; // [rsp+18h] [rbp-1B8h]
  _QWORD v35[2]; // [rsp+50h] [rbp-180h] BYREF
  _BYTE *v36; // [rsp+60h] [rbp-170h]
  _BYTE *v37; // [rsp+68h] [rbp-168h]
  int v38; // [rsp+70h] [rbp-160h]
  __int64 v39; // [rsp+78h] [rbp-158h]
  _QWORD *v40; // [rsp+80h] [rbp-150h] BYREF
  __int64 v41; // [rsp+88h] [rbp-148h]
  _QWORD v42[2]; // [rsp+90h] [rbp-140h] BYREF
  int v43; // [rsp+A0h] [rbp-130h]
  __int64 v44; // [rsp+190h] [rbp-40h]

  v29 = *(_DWORD *)(a3 + 8);
  for ( i = v29; ; i = *(unsigned int *)(a3 + 8) )
  {
    if ( v29 >= i )
    {
      if ( v29 <= i )
        goto LABEL_11;
      if ( v29 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
      {
        sub_16CD150(a3, a3 + 16, v29, 1);
        i = *(unsigned int *)(a3 + 8);
      }
      v5 = (_BYTE *)(*(_QWORD *)a3 + i);
      for ( j = (_BYTE *)(*(_QWORD *)a3 + v29); j != v5; ++v5 )
      {
        if ( v5 )
          *v5 = 0;
      }
    }
    *(_DWORD *)(a3 + 8) = v29;
LABEL_11:
    v38 = 1;
    v37 = 0;
    v35[0] = &unk_49EFC48;
    v36 = 0;
    v35[1] = 0;
    v39 = a3;
    sub_16E7A40(v35, 0, 0, 0);
    if ( *(_BYTE *)(a2 + 16) > 3u )
      goto LABEL_18;
    if ( !*(_QWORD *)(a2 + 40) )
      goto LABEL_42;
    sub_16E1010(&v40);
    if ( (unsigned int)(v43 - 34) > 1 )
    {
      if ( v40 != v42 )
        j_j___libc_free_0(v40, v42[0] + 1LL);
LABEL_42:
      if ( v36 == v37 )
        sub_16E7EE0(v35, ".", 1);
      else
        *v37++ = 46;
      goto LABEL_18;
    }
    if ( v40 != v42 )
      j_j___libc_free_0(v40, v42[0] + 1LL);
    if ( v36 == v37 )
      sub_16E7EE0(v35, "$", 1);
    else
      *v37++ = 36;
LABEL_18:
    v7 = 0;
    v8 = (unsigned int)(*(_DWORD *)(a1 + 32) + 1);
    *(_DWORD *)(a1 + 32) = v8;
    sub_16E7A90(v35, v8);
    v9 = v42;
    v10 = *(_DWORD *)(a3 + 8);
    v41 = 0x10000000000LL;
    v40 = v42;
    if ( v10 )
    {
      v7 = v10;
      v25 = v10;
      if ( v10 <= 0x100
        || (v28 = v10,
            sub_16CD150(&v40, v42, v10, 1),
            v25 = *(unsigned int *)(a3 + 8),
            v9 = v40,
            v10 = v28,
            *(_DWORD *)(a3 + 8)) )
      {
        v27 = v10;
        memcpy(v9, *(const void **)a3, v25);
        v9 = v40;
        v10 = v27;
      }
      LODWORD(v41) = v10;
    }
    v44 = a2;
    v11 = sub_16D19C0(a1, v9, v7);
    v12 = (_QWORD *)(*(_QWORD *)a1 + 8LL * v11);
    if ( !*v12 )
    {
      v13 = v9;
      v14 = v7;
      v15 = v13;
      goto LABEL_25;
    }
    if ( *v12 == -8 )
      break;
    if ( v40 != v42 )
      _libc_free((unsigned __int64)v40);
    v35[0] = &unk_49EFD28;
    sub_16E7960(v35);
  }
  v26 = v9;
  --*(_DWORD *)(a1 + 16);
  v14 = v7;
  v15 = v26;
LABEL_25:
  v30 = v12;
  v32 = v11;
  v16 = malloc(v14 + 17);
  v17 = v32;
  v18 = v30;
  v19 = (_QWORD *)v16;
  if ( !v16 )
  {
    sub_16BD1C0("Allocation failed");
    v18 = v30;
    v17 = v32;
  }
  if ( v14 )
  {
    v31 = v18;
    v33 = v17;
    memcpy(v19 + 2, v15, v14);
    v18 = v31;
    v17 = v33;
  }
  *((_BYTE *)v19 + v14 + 16) = 0;
  *v19 = v14;
  v19[1] = a2;
  *v18 = v19;
  ++*(_DWORD *)(a1 + 12);
  v20 = (__int64 *)(*(_QWORD *)a1 + 8LL * (unsigned int)sub_16D1CD0(a1, v17));
  if ( *v20 == -8 || !*v20 )
  {
    v21 = v20 + 1;
    do
    {
      do
      {
        v22 = *v21;
        v20 = v21++;
      }
      while ( !v22 );
    }
    while ( v22 == -8 );
  }
  if ( v40 != v42 )
    _libc_free((unsigned __int64)v40);
  v23 = *v20;
  v35[0] = &unk_49EFD28;
  sub_16E7960(v35);
  return v23;
}
