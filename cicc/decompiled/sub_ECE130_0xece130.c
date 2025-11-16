// Function: sub_ECE130
// Address: 0xece130
//
__int64 __fastcall sub_ECE130(__int64 a1, _QWORD *a2, __int64 a3)
{
  __int64 v5; // rax
  bool v6; // cc
  _QWORD *v7; // rax

  if ( *(_DWORD *)sub_ECD7B0(a1) != 4 )
    return sub_ECE0E0(a1, a3, 0, 0);
  v5 = sub_ECD7B0(a1);
  v6 = *(_DWORD *)(v5 + 32) <= 0x40u;
  v7 = *(_QWORD **)(v5 + 24);
  if ( !v6 )
    v7 = (_QWORD *)*v7;
  *a2 = v7;
  (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 184LL))(a1);
  return 0;
}
