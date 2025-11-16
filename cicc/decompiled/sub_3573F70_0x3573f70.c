// Function: sub_3573F70
// Address: 0x3573f70
//
__int64 __fastcall sub_3573F70(__int64 a1)
{
  __int64 *v1; // rdx
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // r12
  void (__fastcall *v5)(__int64, void *, _QWORD); // rbx
  void *v6; // rax

  v1 = *(__int64 **)(a1 + 8);
  v2 = *v1;
  v3 = v1[1];
  if ( v2 == v3 )
LABEL_6:
    BUG();
  while ( *(_UNKNOWN **)v2 != &unk_503F08C )
  {
    v2 += 16;
    if ( v3 == v2 )
      goto LABEL_6;
  }
  v4 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v2 + 8) + 104LL))(*(_QWORD *)(v2 + 8), &unk_503F08C);
  v5 = *(void (__fastcall **)(__int64, void *, _QWORD))(*(_QWORD *)v4 + 40LL);
  v6 = sub_CB72A0();
  v5(v4, v6, 0);
  return 0;
}
