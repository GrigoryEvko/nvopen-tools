// Function: sub_F6ADD0
// Address: 0xf6add0
//
__int64 __fastcall sub_F6ADD0(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 *v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // r14
  __int64 *v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // r13
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rbx
  __int64 v20; // rax
  __int64 v21; // r15
  void *v22; // rsi
  bool v23; // al
  __int64 v24; // r10
  __int64 v25; // rbx
  unsigned int v26; // r12d
  _QWORD *v27; // rbx
  _QWORD *v28; // r13
  __int64 v29; // rax
  _QWORD *v30; // rbx
  _QWORD *v31; // r13
  __int64 v32; // rax
  _QWORD *v34; // [rsp+8h] [rbp-58h]
  __int64 v35; // [rsp+10h] [rbp-50h]
  __int64 v36; // [rsp+18h] [rbp-48h]
  _QWORD *v37; // [rsp+20h] [rbp-40h]
  unsigned __int8 v38; // [rsp+20h] [rbp-40h]
  __int64 v39; // [rsp+28h] [rbp-38h]
  __int64 v40; // [rsp+28h] [rbp-38h]

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_54:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4F875EC )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_54;
  }
  v5 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4F875EC);
  v6 = *(__int64 **)(a1 + 8);
  v39 = v5;
  v35 = v5 + 176;
  v7 = *v6;
  v8 = v6[1];
  if ( v7 == v8 )
LABEL_55:
    BUG();
  while ( *(_UNKNOWN **)v7 != &unk_4F8144C )
  {
    v7 += 16;
    if ( v8 == v7 )
      goto LABEL_55;
  }
  v36 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v7 + 8) + 104LL))(*(_QWORD *)(v7 + 8), &unk_4F8144C)
      + 176;
  v9 = sub_B82360(*(_QWORD *)(a1 + 8), (__int64)&unk_4F881C8);
  if ( v9 && (v10 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v9 + 104LL))(v9, &unk_4F881C8)) != 0 )
    v11 = *(_QWORD *)(v10 + 176);
  else
    v11 = 0;
  v12 = *(__int64 **)(a1 + 8);
  v13 = *v12;
  v14 = v12[1];
  if ( v13 == v14 )
LABEL_56:
    BUG();
  while ( *(_UNKNOWN **)v13 != &unk_4F8662C )
  {
    v13 += 16;
    if ( v14 == v13 )
      goto LABEL_56;
  }
  v15 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v13 + 8) + 104LL))(*(_QWORD *)(v13 + 8), &unk_4F8662C);
  v16 = sub_CFFAC0(v15, a2);
  v17 = sub_B82360(*(_QWORD *)(a1 + 8), (__int64)&unk_4F8F808);
  if ( !v17 || (v18 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v17 + 104LL))(v17, &unk_4F8F808)) == 0 )
  {
    v21 = 0;
    v23 = sub_BB9560(a1, (__int64)&unk_4F90E2C);
    v24 = *(_QWORD *)(v39 + 208);
    v40 = *(_QWORD *)(v39 + 216);
    if ( v40 != v24 )
      goto LABEL_20;
    return 0;
  }
  v19 = *(_QWORD *)(v18 + 176);
  v20 = sub_22077B0(760);
  v21 = v20;
  if ( !v20 )
  {
    v23 = sub_BB9560(a1, (__int64)&unk_4F90E2C);
    v24 = *(_QWORD *)(v39 + 208);
    v40 = *(_QWORD *)(v39 + 216);
    if ( v24 != v40 )
      goto LABEL_20;
    return 0;
  }
  *(_QWORD *)v20 = v19;
  v37 = (_QWORD *)(v20 + 24);
  *(_QWORD *)(v20 + 8) = v20 + 24;
  *(_QWORD *)(v20 + 16) = 0x1000000000LL;
  *(_QWORD *)(v20 + 416) = v20 + 440;
  v34 = (_QWORD *)(v20 + 520);
  *(_QWORD *)(v20 + 504) = v20 + 520;
  *(_QWORD *)(v20 + 512) = 0x800000000LL;
  *(_QWORD *)(v20 + 736) = v20 + 720;
  *(_QWORD *)(v20 + 744) = v20 + 720;
  *(_QWORD *)(v20 + 408) = 0;
  *(_QWORD *)(v20 + 424) = 8;
  v22 = &unk_4F90E2C;
  *(_DWORD *)(v20 + 432) = 0;
  *(_BYTE *)(v20 + 436) = 1;
  *(_DWORD *)(v20 + 720) = 0;
  *(_QWORD *)(v20 + 728) = 0;
  *(_QWORD *)(v20 + 752) = 0;
  v23 = sub_BB9560(a1, (__int64)&unk_4F90E2C);
  v24 = *(_QWORD *)(v39 + 208);
  v40 = *(_QWORD *)(v39 + 216);
  if ( v40 == v24 )
  {
    v26 = 0;
    goto LABEL_24;
  }
LABEL_20:
  v25 = v24;
  v26 = 0;
  v38 = v23;
  do
  {
    v22 = (void *)v36;
    v25 += 8;
    v26 |= sub_F6AC10(*(char **)(v25 - 8), v36, v35, v11, v16, (__int64 *)v21, v38);
  }
  while ( v40 != v25 );
  if ( !v21 )
    return v26;
  v37 = (_QWORD *)(v21 + 24);
  v34 = (_QWORD *)(v21 + 520);
LABEL_24:
  sub_F67080(*(_QWORD **)(v21 + 728));
  v27 = *(_QWORD **)(v21 + 504);
  v28 = &v27[3 * *(unsigned int *)(v21 + 512)];
  if ( v27 != v28 )
  {
    do
    {
      v29 = *(v28 - 1);
      v28 -= 3;
      if ( v29 != -4096 && v29 != 0 && v29 != -8192 )
        sub_BD60C0(v28);
    }
    while ( v27 != v28 );
    v28 = *(_QWORD **)(v21 + 504);
  }
  if ( v28 != v34 )
    _libc_free(v28, v22);
  if ( !*(_BYTE *)(v21 + 436) )
    _libc_free(*(_QWORD *)(v21 + 416), v22);
  v30 = *(_QWORD **)(v21 + 8);
  v31 = &v30[3 * *(unsigned int *)(v21 + 16)];
  if ( v30 != v31 )
  {
    do
    {
      v32 = *(v31 - 1);
      v31 -= 3;
      if ( v32 != 0 && v32 != -4096 && v32 != -8192 )
        sub_BD60C0(v31);
    }
    while ( v30 != v31 );
    v31 = *(_QWORD **)(v21 + 8);
  }
  if ( v31 != v37 )
    _libc_free(v31, v22);
  j_j___libc_free_0(v21, 760);
  return v26;
}
