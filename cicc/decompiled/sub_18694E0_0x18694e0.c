// Function: sub_18694E0
// Address: 0x18694e0
//
__int64 __fastcall sub_18694E0(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_6:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4F9D3C0 )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_6;
  }
  *(_QWORD *)(a1 + 344) = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(
                            *(_QWORD *)(v3 + 8),
                            &unk_4F9D3C0);
  return sub_186EFE0(a1, a2);
}
