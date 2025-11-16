// Function: sub_13CB0F0
// Address: 0x13cb0f0
//
__int64 __fastcall sub_13CB0F0(__int64 a1, __int64 a2)
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
  _QWORD *v23; // r12
  unsigned __int64 v24; // rdi
  unsigned __int64 *v25; // r15
  unsigned __int64 *v26; // rbx
  unsigned __int64 v27; // rcx
  unsigned __int64 v28; // rdi
  unsigned __int64 v29; // rdx
  __int64 v30; // rdx
  unsigned __int64 v31; // rdi
  __int64 v33; // [rsp+0h] [rbp-40h]
  __int64 v34; // [rsp+8h] [rbp-38h]

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_46:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4F9D764 )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_46;
  }
  v5 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4F9D764);
  v6 = sub_14CF090(v5, *(_QWORD *)(**(_QWORD **)(a2 + 32) + 56LL));
  v7 = *(__int64 **)(a1 + 8);
  v8 = v6;
  v9 = *v7;
  v10 = v7[1];
  if ( v9 == v10 )
LABEL_43:
    BUG();
  while ( *(_UNKNOWN **)v9 != &unk_4F9920C )
  {
    v9 += 16;
    if ( v10 == v9 )
      goto LABEL_43;
  }
  v11 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v9 + 8) + 104LL))(*(_QWORD *)(v9 + 8), &unk_4F9920C);
  v12 = *(__int64 **)(a1 + 8);
  v13 = v11 + 160;
  v14 = *v12;
  v15 = v12[1];
  if ( v14 == v15 )
LABEL_44:
    BUG();
  while ( *(_UNKNOWN **)v14 != &unk_4F9E06C )
  {
    v14 += 16;
    if ( v15 == v14 )
      goto LABEL_44;
  }
  v16 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v14 + 8) + 104LL))(*(_QWORD *)(v14 + 8), &unk_4F9E06C);
  v17 = *(__int64 **)(a1 + 8);
  v18 = v16 + 160;
  v19 = *v17;
  v20 = v17[1];
  if ( v19 == v20 )
LABEL_45:
    BUG();
  while ( *(_UNKNOWN **)v19 != &unk_4F9A488 )
  {
    v19 += 16;
    if ( v20 == v19 )
      goto LABEL_45;
  }
  v33 = v18;
  v34 = *(_QWORD *)((*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v19 + 8) + 104LL))(
                      *(_QWORD *)(v19 + 8),
                      &unk_4F9A488)
                  + 160);
  v21 = sub_22077B0(520);
  v22 = v21;
  if ( v21 )
    sub_13CB010(v21, a2, v8, v13, v33, v34);
  v23 = *(_QWORD **)(a1 + 160);
  *(_QWORD *)(a1 + 160) = v22;
  if ( v23 )
  {
    v24 = v23[30];
    if ( v24 != v23[29] )
      _libc_free(v24);
    v25 = (unsigned __int64 *)v23[27];
    while ( v23 + 26 != v25 )
    {
      v26 = v25;
      v25 = (unsigned __int64 *)v25[1];
      v27 = *v26 & 0xFFFFFFFFFFFFFFF8LL;
      *v25 = v27 | *v25 & 7;
      *(_QWORD *)(v27 + 8) = v25;
      v28 = v26[8];
      *v26 &= 7u;
      v26[1] = 0;
      *(v26 - 4) = (unsigned __int64)&unk_49EA628;
      if ( v28 != v26[7] )
        _libc_free(v28);
      v29 = v26[5];
      if ( v29 != 0 && v29 != -8 && v29 != -16 )
        sub_1649B30(v26 + 3);
      *(v26 - 4) = (unsigned __int64)&unk_49EE2B0;
      v30 = *(v26 - 1);
      if ( v30 != 0 && v30 != -8 && v30 != -16 )
        sub_1649B30(v26 - 3);
      j_j___libc_free_0(v26 - 4, 136);
    }
    v31 = v23[7];
    if ( v31 != v23[6] )
      _libc_free(v31);
    j_j___libc_free_0(v23, 520);
  }
  return 0;
}
