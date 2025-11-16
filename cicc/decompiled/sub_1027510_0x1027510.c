// Function: sub_1027510
// Address: 0x1027510
//
__int64 __fastcall sub_1027510(_QWORD *a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 *v6; // rdx
  __int64 v7; // r12
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rax

  v2 = (__int64 *)a1[1];
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_14:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4F8EE50 )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_14;
  }
  v5 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4F8EE50);
  v6 = (__int64 *)a1[1];
  v7 = v5;
  v8 = *v6;
  v9 = v6[1];
  if ( v8 == v9 )
LABEL_13:
    BUG();
  while ( *(_UNKNOWN **)v8 != &unk_4F875EC )
  {
    v8 += 16;
    if ( v9 == v8 )
      goto LABEL_13;
  }
  v10 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v8 + 8) + 104LL))(*(_QWORD *)(v8 + 8), &unk_4F875EC);
  a1[24] = a2;
  a1[25] = v7;
  a1[26] = v10 + 176;
  return 0;
}
