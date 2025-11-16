// Function: sub_228A6F0
// Address: 0x228a6f0
//
__int64 __fastcall sub_228A6F0(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v6; // rax
  __int64 *v7; // rdx
  __int64 v8; // r12
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 *v12; // rdx
  __int64 v13; // r15
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // r14
  _QWORD *v17; // rax
  unsigned __int64 v18; // rdi

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_25:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4F86530 )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_25;
  }
  v6 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4F86530);
  v7 = *(__int64 **)(a1 + 8);
  v8 = *(_QWORD *)(v6 + 176);
  v9 = *v7;
  v10 = v7[1];
  if ( v9 == v10 )
LABEL_23:
    BUG();
  while ( *(_UNKNOWN **)v9 != &unk_4F881C8 )
  {
    v9 += 16;
    if ( v10 == v9 )
      goto LABEL_23;
  }
  v11 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v9 + 8) + 104LL))(*(_QWORD *)(v9 + 8), &unk_4F881C8);
  v12 = *(__int64 **)(a1 + 8);
  v13 = *(_QWORD *)(v11 + 176);
  v14 = *v12;
  v15 = v12[1];
  if ( v14 == v15 )
LABEL_24:
    BUG();
  while ( *(_UNKNOWN **)v14 != &unk_4F875EC )
  {
    v14 += 16;
    if ( v15 == v14 )
      goto LABEL_24;
  }
  v16 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v14 + 8) + 104LL))(*(_QWORD *)(v14 + 8), &unk_4F875EC)
      + 176;
  v17 = (_QWORD *)sub_22077B0(0x30u);
  if ( v17 )
  {
    *v17 = v8;
    v17[1] = v13;
    v17[2] = v16;
    v17[3] = a2;
  }
  v18 = *(_QWORD *)(a1 + 176);
  *(_QWORD *)(a1 + 176) = v17;
  if ( v18 )
    j_j___libc_free_0(v18);
  return 0;
}
