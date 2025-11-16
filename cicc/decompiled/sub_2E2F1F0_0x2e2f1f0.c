// Function: sub_2E2F1F0
// Address: 0x2e2f1f0
//
__int64 __fastcall sub_2E2F1F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r15
  __int64 v8; // rbx
  __int64 v10; // rax
  unsigned __int64 v11; // rdx
  __int64 *v12; // rdi
  __int64 *v13; // r15
  __int64 *v14; // rbx
  char v15; // r14
  __int64 v16; // rsi
  void **v17; // rdx
  __int64 v18; // r8
  __int64 v19; // r9
  void **v20; // rcx
  void **v21; // rax
  void **v22; // rcx
  __int64 v23; // rcx
  void **v24; // rax
  void **v25; // rax
  void **v27; // rax
  unsigned int v28; // edi
  __int64 *v29; // rax
  __int64 *v30; // [rsp+20h] [rbp-E0h] BYREF
  __int64 v31; // [rsp+28h] [rbp-D8h]
  _BYTE v32[64]; // [rsp+30h] [rbp-D0h] BYREF
  __int64 v33; // [rsp+70h] [rbp-90h] BYREF
  void **v34; // [rsp+78h] [rbp-88h]
  __int64 v35; // [rsp+80h] [rbp-80h]
  int v36; // [rsp+88h] [rbp-78h]
  char v37; // [rsp+8Ch] [rbp-74h]
  _QWORD v38[2]; // [rsp+90h] [rbp-70h] BYREF
  __int64 v39; // [rsp+A0h] [rbp-60h] BYREF
  void **v40; // [rsp+A8h] [rbp-58h]
  __int64 v41; // [rsp+B0h] [rbp-50h]
  int v42; // [rsp+B8h] [rbp-48h]
  char v43; // [rsp+BCh] [rbp-44h]
  _QWORD v44[8]; // [rsp+C0h] [rbp-40h] BYREF

  v6 = a3 + 8;
  v8 = *(_QWORD *)(a3 + 16);
  v30 = (__int64 *)v32;
  v31 = 0x800000000LL;
  if ( v8 == a3 + 8 )
  {
    v12 = (__int64 *)v32;
LABEL_36:
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 8) = a1 + 32;
    *(_QWORD *)(a1 + 56) = a1 + 80;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
    goto LABEL_37;
  }
  do
  {
    while ( 1 )
    {
      if ( !v8 )
        BUG();
      if ( (*(_BYTE *)(v8 - 23) & 0x1C) != 0 )
        break;
      v8 = *(_QWORD *)(v8 + 8);
      if ( v6 == v8 )
        goto LABEL_9;
    }
    v10 = (unsigned int)v31;
    v11 = (unsigned int)v31 + 1LL;
    if ( v11 > HIDWORD(v31) )
    {
      sub_C8D5F0((__int64)&v30, v32, v11, 8u, a5, a6);
      v10 = (unsigned int)v31;
    }
    v30[v10] = v8 - 56;
    LODWORD(v31) = v31 + 1;
    v8 = *(_QWORD *)(v8 + 8);
  }
  while ( v6 != v8 );
LABEL_9:
  v12 = v30;
  v13 = &v30[(unsigned int)v31];
  if ( v13 == v30 )
    goto LABEL_36;
  v14 = v30;
  v15 = 0;
  do
  {
    v16 = *v14++;
    v15 |= sub_2E2EC50((__int64 **)a3, v16);
  }
  while ( v13 != v14 );
  v12 = v30;
  if ( !v15 )
    goto LABEL_36;
  v43 = 1;
  v34 = (void **)v38;
  v35 = 0x100000002LL;
  v40 = (void **)v44;
  v41 = 2;
  v42 = 0;
  v36 = 0;
  v37 = 1;
  v38[0] = &qword_4F82400;
  v33 = 1;
  if ( &qword_4F82400 == (__int64 *)&unk_4F86B78 )
  {
    HIDWORD(v35) = 0;
    v33 = 2;
  }
  v44[0] = &unk_4F86B78;
  HIDWORD(v41) = 1;
  v20 = (void **)&v38[HIDWORD(v35)];
  v39 = 1;
  if ( v20 == v38 )
  {
    v27 = v40;
    v28 = 1;
    v22 = v40 + 1;
LABEL_43:
    while ( *v27 != &unk_4F87818 )
    {
      if ( ++v27 == v22 )
        goto LABEL_51;
    }
  }
  else
  {
    v21 = (void **)v38;
    while ( *v21 != &unk_4F87818 )
    {
      if ( v20 == ++v21 )
        goto LABEL_40;
    }
    --HIDWORD(v35);
    v22 = (void **)v38[HIDWORD(v35)];
    *v21 = v22;
    ++v33;
    if ( !v43 )
      goto LABEL_20;
LABEL_40:
    v27 = v40;
    v28 = HIDWORD(v41);
    v22 = &v40[HIDWORD(v41)];
    if ( v40 != v22 )
      goto LABEL_43;
LABEL_51:
    if ( (unsigned int)v41 <= v28 )
    {
LABEL_20:
      sub_C8CC70((__int64)&v39, (__int64)&unk_4F87818, (__int64)v17, (__int64)v22, v18, v19);
      v15 = v43;
      if ( !v37 )
        goto LABEL_45;
      goto LABEL_21;
    }
    HIDWORD(v41) = v28 + 1;
    *v22 = &unk_4F87818;
    v15 = v43;
    ++v39;
  }
  if ( !v37 )
  {
LABEL_45:
    v29 = sub_C8CA60((__int64)&v33, (__int64)&unk_4F87F18);
    if ( v29 )
    {
      *v29 = -2;
      ++v36;
      ++v33;
    }
    if ( !v43 )
      goto LABEL_48;
    goto LABEL_27;
  }
LABEL_21:
  v23 = (__int64)&v34[HIDWORD(v35)];
  if ( v34 != (void **)v23 )
  {
    v24 = v34;
    while ( *v24 != &unk_4F87F18 )
    {
      if ( (void **)v23 == ++v24 )
        goto LABEL_26;
    }
    --HIDWORD(v35);
    v17 = (void **)v34[HIDWORD(v35)];
    *v24 = v17;
    v15 = v43;
    ++v33;
  }
LABEL_26:
  if ( !v15 )
    goto LABEL_48;
LABEL_27:
  v25 = v40;
  v23 = HIDWORD(v41);
  v17 = &v40[HIDWORD(v41)];
  if ( v40 != v17 )
  {
    while ( *v25 != &unk_4F87F18 )
    {
      if ( v17 == ++v25 )
        goto LABEL_49;
    }
    goto LABEL_31;
  }
LABEL_49:
  if ( HIDWORD(v41) < (unsigned int)v41 )
  {
    ++HIDWORD(v41);
    *v17 = &unk_4F87F18;
    ++v39;
    goto LABEL_31;
  }
LABEL_48:
  sub_C8CC70((__int64)&v39, (__int64)&unk_4F87F18, (__int64)v17, v23, (__int64)&v33, v19);
LABEL_31:
  sub_C8CF70(a1, (void *)(a1 + 32), 2, (__int64)v38, (__int64)&v33);
  sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v44, (__int64)&v39);
  if ( !v43 )
    _libc_free((unsigned __int64)v40);
  v12 = v30;
  if ( !v37 )
  {
    _libc_free((unsigned __int64)v34);
    v12 = v30;
  }
LABEL_37:
  if ( v12 != (__int64 *)v32 )
    _libc_free((unsigned __int64)v12);
  return a1;
}
