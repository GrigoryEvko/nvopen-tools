// Function: sub_1498000
// Address: 0x1498000
//
__int64 __fastcall sub_1498000(__int64 a1)
{
  __int64 *v1; // rdx
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v5; // r12
  __int64 v6; // rax
  __int64 v7; // rdi

  v1 = *(__int64 **)(a1 + 8);
  v2 = *v1;
  v3 = v1[1];
  if ( v2 == v3 )
LABEL_10:
    BUG();
  while ( *(_UNKNOWN **)v2 != &unk_4F9A488 )
  {
    v2 += 16;
    if ( v3 == v2 )
      goto LABEL_10;
  }
  v5 = *(_QWORD *)((*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v2 + 8) + 104LL))(
                     *(_QWORD *)(v2 + 8),
                     &unk_4F9A488)
                 + 160);
  v6 = sub_22077B0(16);
  if ( v6 )
    *(_QWORD *)(v6 + 8) = v5;
  v7 = *(_QWORD *)(a1 + 160);
  *(_QWORD *)(a1 + 160) = v6;
  if ( v7 )
    j_j___libc_free_0(v7, 16);
  return 0;
}
