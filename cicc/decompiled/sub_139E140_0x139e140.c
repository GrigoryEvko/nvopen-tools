// Function: sub_139E140
// Address: 0x139e140
//
__int64 __fastcall sub_139E140(_QWORD *a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 *v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rdx

  v2 = (__int64 *)a1[1];
  a1[20] = a2;
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_14:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4F9A488 )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_14;
  }
  v5 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4F9A488);
  v6 = (__int64 *)a1[1];
  a1[22] = *(_QWORD *)(v5 + 160);
  v7 = *v6;
  v8 = v6[1];
  if ( v7 == v8 )
LABEL_13:
    BUG();
  while ( *(_UNKNOWN **)v7 != &unk_4F9920C )
  {
    v7 += 16;
    if ( v8 == v7 )
      goto LABEL_13;
  }
  a1[21] = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v7 + 8) + 104LL))(
             *(_QWORD *)(v7 + 8),
             &unk_4F9920C)
         + 160;
  return 0;
}
