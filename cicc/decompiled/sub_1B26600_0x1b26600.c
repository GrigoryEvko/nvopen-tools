// Function: sub_1B26600
// Address: 0x1b26600
//
__int64 __fastcall sub_1B26600(__int64 a1, __int64 a2)
{
  __int64 *v3; // rdx
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 *v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rax

  if ( (unsigned __int8)sub_1636880(a1, a2) )
    return 0;
  v3 = *(__int64 **)(a1 + 8);
  v4 = *v3;
  v5 = v3[1];
  if ( v4 == v5 )
    goto LABEL_12;
  while ( *(_UNKNOWN **)v4 != &unk_4F9E06C )
  {
    v4 += 16;
    if ( v5 == v4 )
      goto LABEL_12;
  }
  (*(void (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v4 + 8) + 104LL))(*(_QWORD *)(v4 + 8), &unk_4F9E06C);
  v6 = *(__int64 **)(a1 + 8);
  v7 = *v6;
  v8 = v6[1];
  if ( v7 == v8 )
LABEL_12:
    BUG();
  while ( *(_UNKNOWN **)v7 != &unk_4F9D764 )
  {
    v7 += 16;
    if ( v8 == v7 )
      goto LABEL_12;
  }
  v9 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v7 + 8) + 104LL))(*(_QWORD *)(v7 + 8), &unk_4F9D764);
  sub_14CF090(v9, a2);
  return sub_1B26410(a2);
}
