// Function: sub_B5A1B0
// Address: 0xb5a1b0
//
__int64 __fastcall sub_B5A1B0(__int64 a1)
{
  __int64 v1; // rax

  v1 = *(_QWORD *)(a1 - 32);
  if ( !v1 || *(_BYTE *)v1 || *(_QWORD *)(v1 + 24) != *(_QWORD *)(a1 + 80) )
    BUG();
  return sub_B6AFF0(*(unsigned int *)(v1 + 36));
}
