// Function: sub_1DE2F70
// Address: 0x1de2f70
//
__int64 __fastcall sub_1DE2F70(__int64 a1, _QWORD *a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 *v6; // rdx
  __int64 v7; // r14
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rax

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_14:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4FC5828 )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_14;
  }
  v5 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4FC5828);
  v6 = *(__int64 **)(a1 + 8);
  v7 = v5;
  v8 = *v6;
  v9 = v6[1];
  if ( v8 == v9 )
LABEL_13:
    BUG();
  while ( *(_UNKNOWN **)v8 != &unk_4FC6A0C )
  {
    v8 += 16;
    if ( v9 == v8 )
      goto LABEL_13;
  }
  v10 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v8 + 8) + 104LL))(*(_QWORD *)(v8 + 8), &unk_4FC6A0C);
  sub_1DE2CD0(a1, a2, v7, v10);
  return 0;
}
