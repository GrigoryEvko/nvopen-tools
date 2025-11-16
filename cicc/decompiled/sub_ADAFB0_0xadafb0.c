// Function: sub_ADAFB0
// Address: 0xadafb0
//
__int64 __fastcall sub_ADAFB0(unsigned __int64 a1, __int64 a2)
{
  int v3; // edx
  char v4; // cl
  __int64 v5; // rdi
  __int64 v6; // rax

  v3 = *(unsigned __int8 *)(a2 + 8);
  v4 = *(_BYTE *)(a2 + 8);
  if ( (unsigned int)(v3 - 17) <= 1 )
    v4 = *(_BYTE *)(**(_QWORD **)(a2 + 16) + 8LL);
  if ( v4 == 12 )
    return sub_AD4C50(a1, (__int64 **)a2, 0);
  v5 = *(_QWORD *)(a1 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v5 + 8) - 17 <= 1 )
    v5 = **(_QWORD **)(v5 + 16);
  if ( (unsigned int)(v3 - 17) > 1 )
  {
    if ( (_BYTE)v3 != 14 || *(_DWORD *)(v5 + 8) >> 8 == *(_DWORD *)(a2 + 8) >> 8 )
      return sub_AD4C90(a1, (__int64 **)a2, 0);
  }
  else
  {
    v6 = **(_QWORD **)(a2 + 16);
    if ( *(_BYTE *)(v6 + 8) != 14 )
      return sub_AD4C90(a1, (__int64 **)a2, 0);
    if ( (unsigned __int8)(v3 - 17) >= 2u )
      v6 = a2;
    if ( *(_DWORD *)(v5 + 8) >> 8 == *(_DWORD *)(v6 + 8) >> 8 )
      return sub_AD4C90(a1, (__int64 **)a2, 0);
  }
  return sub_ADA8A0(a1, a2, 0);
}
