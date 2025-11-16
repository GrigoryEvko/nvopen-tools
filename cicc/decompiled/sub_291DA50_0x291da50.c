// Function: sub_291DA50
// Address: 0x291da50
//
__int64 __fastcall sub_291DA50(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rdx

  v1 = *(_QWORD *)(a1 - 32);
  if ( !v1 || *(_BYTE *)v1 || *(_QWORD *)(v1 + 24) != *(_QWORD *)(a1 + 80) )
    BUG();
  v2 = *(_DWORD *)(a1 + 4) & 0x7FFFFFF;
  if ( *(_DWORD *)(v1 + 36) == 68 )
    return *(_QWORD *)(*(_QWORD *)(a1 + 32 * (5 - v2)) + 24LL);
  else
    return *(_QWORD *)(*(_QWORD *)(a1 + 32 * (2 - v2)) + 24LL);
}
