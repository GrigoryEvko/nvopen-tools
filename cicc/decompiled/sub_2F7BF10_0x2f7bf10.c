// Function: sub_2F7BF10
// Address: 0x2f7bf10
//
__int64 __fastcall sub_2F7BF10(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v6; // [rsp+8h] [rbp-18h] BYREF

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_6:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_5024E70 )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_6;
  }
  v6 = *(_QWORD *)((*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(
                     *(_QWORD *)(v3 + 8),
                     &unk_5024E70)
                 + 176);
  return sub_2F7B910(&v6, a2);
}
