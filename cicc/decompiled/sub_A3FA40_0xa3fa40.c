// Function: sub_A3FA40
// Address: 0xa3fa40
//
__int64 __fastcall sub_A3FA40(__int64 a1)
{
  __int64 v1; // r8
  __int64 v2; // rax

  v1 = 0;
  v2 = (__int64)(*(_QWORD *)(a1 + 64) - *(_QWORD *)(a1 + 56)) >> 3;
  if ( (_DWORD)v2 )
  {
    _BitScanReverse((unsigned int *)&v2, v2);
    return (int)(32 - (v2 ^ 0x1F));
  }
  return v1;
}
