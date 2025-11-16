// Function: sub_8788F0
// Address: 0x8788f0
//
__int64 __fastcall sub_8788F0(__int64 a1)
{
  __int64 v1; // rdx
  __int64 v2; // r8

  v1 = *(_QWORD *)(a1 + 88);
  v2 = 0;
  if ( (*(_BYTE *)(v1 + 177) & 0x30) == 0x10 && (*(_BYTE *)(v1 + 178) & 1) == 0 )
    return *(_QWORD *)(*(_QWORD *)(a1 + 96) + 104LL);
  return v2;
}
