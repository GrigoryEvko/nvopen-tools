// Function: sub_1C07360
// Address: 0x1c07360
//
__int64 __fastcall sub_1C07360(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v6; // rdi
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  void *v10; // rax

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_12:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4FBA0D1 )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_12;
  }
  v6 = *(_QWORD *)(v3 + 8);
  *(_QWORD *)(a1 + 160) = *(_QWORD *)((*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v6 + 104LL))(
                                        v6,
                                        &unk_4FBA0D1)
                                    + 160);
  if ( byte_4FBA0A0 )
  {
    v10 = sub_16E8C20(v6, (__int64)&unk_4FBA0D1, v7, v8, v9);
    sub_1559E80(a2, (__int64)v10, 0, 0, 0);
  }
  if ( byte_4FB9FC0 )
  {
    sub_1C06430(a1, a2);
    if ( !byte_4FB9EE0 )
      return 0;
  }
  else if ( !byte_4FB9EE0 )
  {
    return 0;
  }
  sub_1C06C80(a1, a2);
  return 0;
}
