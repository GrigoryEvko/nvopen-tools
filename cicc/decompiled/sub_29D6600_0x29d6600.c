// Function: sub_29D6600
// Address: 0x29d6600
//
__int64 __fastcall sub_29D6600(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rax
  __int64 v3; // r12
  __int64 *v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 *v8; // rdx
  __int64 v9; // r13
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rax

  v1 = sub_B82360(*(_QWORD *)(a1 + 8), (__int64)&unk_4F875EC);
  if ( v1 && (v2 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v1 + 104LL))(v1, &unk_4F875EC)) != 0 )
    v3 = v2 + 176;
  else
    v3 = 0;
  v4 = *(__int64 **)(a1 + 8);
  v5 = *v4;
  v6 = v4[1];
  if ( v5 == v6 )
LABEL_18:
    BUG();
  while ( *(_UNKNOWN **)v5 != &unk_4F92384 )
  {
    v5 += 16;
    if ( v6 == v5 )
      goto LABEL_18;
  }
  v7 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v5 + 8) + 104LL))(*(_QWORD *)(v5 + 8), &unk_4F92384);
  v8 = *(__int64 **)(a1 + 8);
  v9 = v7 + 184;
  v10 = *v8;
  v11 = v8[1];
  if ( v10 == v11 )
LABEL_17:
    BUG();
  while ( *(_UNKNOWN **)v10 != &unk_4F8144C )
  {
    v10 += 16;
    if ( v11 == v10 )
      goto LABEL_17;
  }
  v12 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v10 + 8) + 104LL))(*(_QWORD *)(v10 + 8), &unk_4F8144C);
  return sub_29D3E80(v9, v12 + 176, v3);
}
