// Function: sub_B59B70
// Address: 0xb59b70
//
__int64 __fastcall sub_B59B70(__int64 a1)
{
  __int64 v1; // rax

  v1 = *(_QWORD *)(a1 - 32);
  if ( !v1 || *(_BYTE *)v1 || *(_QWORD *)(v1 + 24) != *(_QWORD *)(a1 + 80) )
    BUG();
  if ( *(_DWORD *)(v1 + 36) == 203 )
    BUG();
  return *(_QWORD *)(a1 + 32 * (2LL - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF)));
}
