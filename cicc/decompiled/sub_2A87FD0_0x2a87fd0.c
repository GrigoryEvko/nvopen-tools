// Function: sub_2A87FD0
// Address: 0x2a87fd0
//
__int64 __fastcall sub_2A87FD0(__int64 a1)
{
  __int64 *v1; // rdx
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // rax
  __int64 *v5; // rdx
  __int64 v6; // r12
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rax

  v1 = *(__int64 **)(a1 + 8);
  v2 = *v1;
  v3 = v1[1];
  if ( v2 == v3 )
LABEL_14:
    BUG();
  while ( *(_UNKNOWN **)v2 != &unk_4F875EC )
  {
    v2 += 16;
    if ( v3 == v2 )
      goto LABEL_14;
  }
  v4 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v2 + 8) + 104LL))(*(_QWORD *)(v2 + 8), &unk_4F875EC);
  v5 = *(__int64 **)(a1 + 8);
  v6 = v4 + 176;
  v7 = *v5;
  v8 = v5[1];
  if ( v7 == v8 )
LABEL_13:
    BUG();
  while ( *(_UNKNOWN **)v7 != &unk_4F8144C )
  {
    v7 += 16;
    if ( v8 == v7 )
      goto LABEL_13;
  }
  v9 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v7 + 8) + 104LL))(*(_QWORD *)(v7 + 8), &unk_4F8144C);
  return sub_2A87F40(v6, v9 + 176);
}
