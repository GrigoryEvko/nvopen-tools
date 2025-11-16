// Function: sub_22AC840
// Address: 0x22ac840
//
__int64 __fastcall sub_22AC840(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 *v7; // rdx
  __int64 v8; // r13
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 *v12; // rdx
  __int64 v13; // r14
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 *v17; // rdx
  __int64 v18; // r8
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // r15
  unsigned __int64 v23; // r13
  unsigned __int64 *v24; // r15
  unsigned __int64 v25; // rdx
  __int64 v26; // rdx
  unsigned __int64 *v27; // rbx
  unsigned __int64 v28; // rcx
  bool v29; // zf
  __int64 v31; // [rsp+0h] [rbp-40h]
  __int64 v32; // [rsp+8h] [rbp-38h]

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_47:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4F8662C )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_47;
  }
  v5 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4F8662C);
  v6 = sub_CFFAC0(v5, *(_QWORD *)(**(_QWORD **)(a2 + 32) + 72LL));
  v7 = *(__int64 **)(a1 + 8);
  v8 = v6;
  v9 = *v7;
  v10 = v7[1];
  if ( v9 == v10 )
LABEL_45:
    BUG();
  while ( *(_UNKNOWN **)v9 != &unk_4F875EC )
  {
    v9 += 16;
    if ( v10 == v9 )
      goto LABEL_45;
  }
  v11 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v9 + 8) + 104LL))(*(_QWORD *)(v9 + 8), &unk_4F875EC);
  v12 = *(__int64 **)(a1 + 8);
  v13 = v11 + 176;
  v14 = *v12;
  v15 = v12[1];
  if ( v14 == v15 )
LABEL_46:
    BUG();
  while ( *(_UNKNOWN **)v14 != &unk_4F8144C )
  {
    v14 += 16;
    if ( v15 == v14 )
      goto LABEL_46;
  }
  v16 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v14 + 8) + 104LL))(*(_QWORD *)(v14 + 8), &unk_4F8144C);
  v17 = *(__int64 **)(a1 + 8);
  v18 = v16 + 176;
  v19 = *v17;
  v20 = v17[1];
  if ( v19 == v20 )
LABEL_44:
    BUG();
  while ( *(_UNKNOWN **)v19 != &unk_4F881C8 )
  {
    v19 += 16;
    if ( v20 == v19 )
      goto LABEL_44;
  }
  v31 = v18;
  v32 = *(_QWORD *)((*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v19 + 8) + 104LL))(
                      *(_QWORD *)(v19 + 8),
                      &unk_4F881C8)
                  + 176);
  v21 = sub_22077B0(0x1F8u);
  v22 = v21;
  if ( v21 )
    sub_22AC730(v21, a2, v8, v13, v31, v32);
  v23 = *(_QWORD *)(a1 + 176);
  *(_QWORD *)(a1 + 176) = v22;
  if ( v23 )
  {
    if ( !*(_BYTE *)(v23 + 244) )
      _libc_free(*(_QWORD *)(v23 + 224));
    v24 = *(unsigned __int64 **)(v23 + 208);
    while ( (unsigned __int64 *)(v23 + 200) != v24 )
    {
      v27 = v24;
      v24 = (unsigned __int64 *)v24[1];
      v28 = *v27 & 0xFFFFFFFFFFFFFFF8LL;
      *v24 = v28 | *v24 & 7;
      *(_QWORD *)(v28 + 8) = v24;
      *v27 &= 7u;
      v29 = *((_BYTE *)v27 + 76) == 0;
      v27[1] = 0;
      *(v27 - 4) = (unsigned __int64)&unk_4A09CC0;
      if ( v29 )
        _libc_free(v27[7]);
      v25 = v27[5];
      if ( v25 != 0 && v25 != -4096 && v25 != -8192 )
        sub_BD60C0(v27 + 3);
      *(v27 - 4) = (unsigned __int64)&unk_49DB368;
      v26 = *(v27 - 1);
      if ( v26 != 0 && v26 != -4096 && v26 != -8192 )
        sub_BD60C0(v27 - 3);
      j_j___libc_free_0((unsigned __int64)(v27 - 4));
    }
    if ( !*(_BYTE *)(v23 + 68) )
      _libc_free(*(_QWORD *)(v23 + 48));
    j_j___libc_free_0(v23);
  }
  return 0;
}
