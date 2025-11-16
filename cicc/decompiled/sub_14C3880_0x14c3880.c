// Function: sub_14C3880
// Address: 0x14c3880
//
__int64 __fastcall sub_14C3880(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rax

  if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
    v5 = *(_QWORD *)(a1 - 8);
  else
    v5 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
  return sub_14C3040(*(_QWORD *)v5, *(__int64 **)(v5 + 24), a1, a2, a3, a4, a5);
}
