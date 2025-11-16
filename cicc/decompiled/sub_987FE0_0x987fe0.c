// Function: sub_987FE0
// Address: 0x987fe0
//
__int64 __fastcall sub_987FE0(__int64 a1)
{
  __int64 v1; // rax

  v1 = *(_QWORD *)(a1 - 32);
  if ( !v1 || *(_BYTE *)v1 || *(_QWORD *)(v1 + 24) != *(_QWORD *)(a1 + 80) )
    BUG();
  return *(unsigned int *)(v1 + 36);
}
