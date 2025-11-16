// Function: sub_DF3780
// Address: 0xdf3780
//
__int64 __fastcall sub_DF3780(__int64 a1)
{
  __int64 *v1; // rdx
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v5; // r12
  _QWORD *v6; // rax
  __int64 v7; // rdi

  v1 = *(__int64 **)(a1 + 8);
  v2 = *v1;
  v3 = v1[1];
  if ( v2 == v3 )
LABEL_10:
    BUG();
  while ( *(_UNKNOWN **)v2 != &unk_4F881C8 )
  {
    v2 += 16;
    if ( v3 == v2 )
      goto LABEL_10;
  }
  v5 = *(_QWORD *)((*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v2 + 8) + 104LL))(
                     *(_QWORD *)(v2 + 8),
                     &unk_4F881C8)
                 + 176);
  v6 = (_QWORD *)sub_22077B0(8);
  if ( v6 )
    *v6 = v5;
  v7 = *(_QWORD *)(a1 + 176);
  *(_QWORD *)(a1 + 176) = v6;
  if ( v7 )
    j_j___libc_free_0(v7, 8);
  return 0;
}
