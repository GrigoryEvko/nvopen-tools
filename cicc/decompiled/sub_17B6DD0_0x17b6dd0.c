// Function: sub_17B6DD0
// Address: 0x17b6dd0
//
__int64 __fastcall sub_17B6DD0(__int64 a1, __int64 a2, double a3, double a4, double a5)
{
  __int64 *v5; // rdx
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 *v9; // rdx
  __int64 v10; // r13
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax

  v5 = *(__int64 **)(a1 + 8);
  v6 = *v5;
  v7 = v5[1];
  if ( v6 == v7 )
LABEL_14:
    BUG();
  while ( *(_UNKNOWN **)v6 != &unk_4F9B6E8 )
  {
    v6 += 16;
    if ( v7 == v6 )
      goto LABEL_14;
  }
  v8 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v6 + 8) + 104LL))(*(_QWORD *)(v6 + 8), &unk_4F9B6E8);
  v9 = *(__int64 **)(a1 + 8);
  v10 = v8 + 360;
  v11 = *v9;
  v12 = v9[1];
  if ( v11 == v12 )
LABEL_13:
    BUG();
  while ( *(_UNKNOWN **)v11 != &unk_4F9A488 )
  {
    v11 += 16;
    if ( v12 == v11 )
      goto LABEL_13;
  }
  v13 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v11 + 8) + 104LL))(*(_QWORD *)(v11 + 8), &unk_4F9A488);
  return sub_17B6650(a2, v10, *(_QWORD *)(v13 + 160), a3, a4, a5);
}
