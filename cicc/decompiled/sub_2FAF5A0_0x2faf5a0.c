// Function: sub_2FAF5A0
// Address: 0x2faf5a0
//
__int64 __fastcall sub_2FAF5A0(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 *v6; // rdx
  __int64 v7; // r13
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rax

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_14:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_501D134 )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_14;
  }
  v5 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_501D134);
  v6 = *(__int64 **)(a1 + 8);
  v7 = *(_QWORD *)(v5 + 200);
  v8 = *v6;
  v9 = v6[1];
  if ( v8 == v9 )
LABEL_13:
    BUG();
  while ( *(_UNKNOWN **)v8 != &unk_501EC08 )
  {
    v8 += 16;
    if ( v9 == v8 )
      goto LABEL_13;
  }
  v10 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v8 + 8) + 104LL))(*(_QWORD *)(v8 + 8), &unk_501EC08);
  sub_2FAF320(a1 + 200, a2, v7, (__int64 *)(v10 + 200));
  return 0;
}
