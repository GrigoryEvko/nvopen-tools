// Function: sub_B5B520
// Address: 0xb5b520
//
__int64 __fastcall sub_B5B520(__int64 a1)
{
  __int64 v1; // rax

  v1 = *(_QWORD *)(a1 - 32);
  if ( !v1 || *(_BYTE *)v1 || *(_QWORD *)(v1 + 24) != *(_QWORD *)(a1 + 80) )
    BUG();
  return sub_B5B240(*(_DWORD *)(v1 + 36));
}
