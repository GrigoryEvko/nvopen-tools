// Function: sub_153EE90
// Address: 0x153ee90
//
__int64 __fastcall sub_153EE90(__int64 a1)
{
  __int64 v1; // r8
  __int64 v2; // rax

  v1 = 0;
  v2 = (__int64)(*(_QWORD *)(a1 + 64) - *(_QWORD *)(a1 + 56)) >> 3;
  if ( (_DWORD)v2 )
  {
    _BitScanReverse((unsigned int *)&v2, v2);
    return 32 - ((unsigned int)v2 ^ 0x1F);
  }
  return v1;
}
