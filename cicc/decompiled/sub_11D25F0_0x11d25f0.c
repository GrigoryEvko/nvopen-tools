// Function: sub_11D25F0
// Address: 0x11d25f0
//
__int64 __fastcall sub_11D25F0(_QWORD *a1)
{
  __int64 *v1; // rdx
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v5; // rax
  __int64 *v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rdi
  __int64 v11; // rax
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rax
  __int64 v15; // r13
  __int64 v16; // r12
  unsigned int v17; // r15d
  __int64 v18; // r14
  __int64 *v19; // rbx
  __int64 v20; // rdi
  __int64 *i; // [rsp+8h] [rbp-38h]

  v1 = (__int64 *)a1[1];
  v2 = *v1;
  v3 = v1[1];
  if ( v2 == v3 )
LABEL_20:
    BUG();
  while ( *(_UNKNOWN **)v2 != &unk_4F875EC )
  {
    v2 += 16;
    if ( v3 == v2 )
      goto LABEL_20;
  }
  v5 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v2 + 8) + 104LL))(*(_QWORD *)(v2 + 8), &unk_4F875EC);
  v6 = (__int64 *)a1[1];
  a1[23] = v5 + 176;
  v7 = *v6;
  v8 = v6[1];
  if ( v7 == v8 )
LABEL_19:
    BUG();
  while ( *(_UNKNOWN **)v7 != &unk_4F8144C )
  {
    v7 += 16;
    if ( v8 == v7 )
      goto LABEL_19;
  }
  v9 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v7 + 8) + 104LL))(*(_QWORD *)(v7 + 8), &unk_4F8144C);
  v10 = a1[1];
  a1[22] = v9 + 176;
  v11 = sub_B82360(v10, (__int64)&unk_4F881C8);
  if ( v11 && (v14 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v11 + 104LL))(v11, &unk_4F881C8)) != 0 )
    v15 = *(_QWORD *)(v14 + 176);
  else
    v15 = 0;
  v16 = a1[23];
  a1[24] = v15;
  v17 = 0;
  v18 = a1[22];
  v19 = *(__int64 **)(v16 + 32);
  for ( i = *(__int64 **)(v16 + 40); i != v19; v17 |= sub_11D2180(v20, v18, v16, v15, v12, v13) )
    v20 = *v19++;
  return v17;
}
