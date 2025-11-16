// Function: sub_2E15790
// Address: 0x2e15790
//
__int64 __fastcall sub_2E15790(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 *v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rdx

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_14:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_5025C1C )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_14;
  }
  v5 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_5025C1C);
  v6 = *(__int64 **)(a1 + 8);
  *(_QWORD *)(a1 + 232) = v5 + 200;
  v7 = *v6;
  v8 = v6[1];
  if ( v7 == v8 )
LABEL_13:
    BUG();
  while ( *(_UNKNOWN **)v7 != &unk_501FE44 )
  {
    v7 += 16;
    if ( v8 == v7 )
      goto LABEL_13;
  }
  *(_QWORD *)(a1 + 240) = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v7 + 8) + 104LL))(
                            *(_QWORD *)(v7 + 8),
                            &unk_501FE44)
                        + 200;
  sub_2E15500((unsigned int *)(a1 + 200), a2);
  return 0;
}
