// Function: sub_2950010
// Address: 0x2950010
//
__int64 __fastcall sub_2950010(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v6; // r13
  __int64 v7; // rax
  __int64 v8; // rax
  unsigned __int64 v9; // rdx

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_10:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4F89C28 )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_10;
  }
  v5 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4F89C28);
  v6 = sub_DFED00(v5, a2);
  v7 = sub_B82360(*(_QWORD *)(a1 + 8), (__int64)&unk_4F8144C);
  if ( v7 && (v8 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v7 + 104LL))(v7, &unk_4F8144C)) != 0 )
    v9 = v8 + 176;
  else
    v9 = 0;
  return sub_294D310(a2, v6, v9);
}
