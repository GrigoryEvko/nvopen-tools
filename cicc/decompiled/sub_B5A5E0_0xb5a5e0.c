// Function: sub_B5A5E0
// Address: 0xb5a5e0
//
__int16 __fastcall sub_B5A5E0(__int64 a1)
{
  __int64 v1; // rax
  int v2; // eax

  v1 = *(_QWORD *)(a1 - 32);
  if ( !v1 || *(_BYTE *)v1 || *(_QWORD *)(v1 + 24) != *(_QWORD *)(a1 + 80) )
    BUG();
  v2 = sub_B5A570(*(_DWORD *)(v1 + 36));
  return sub_A74840((_QWORD *)(a1 + 72), v2);
}
