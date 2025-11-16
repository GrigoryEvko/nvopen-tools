// Function: sub_31DB6B0
// Address: 0x31db6b0
//
__int64 __fastcall sub_31DB6B0(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  int v3; // eax
  __int64 result; // rax

  if ( (*(_BYTE *)(a2 + 32) & 0xF) == 1 )
    return 0;
  if ( sub_B2FC80(a2) )
    return 0;
  v2 = *(_QWORD *)(a1 + 208);
  v3 = *(_DWORD *)(v2 + 336);
  if ( v3 == 1 )
  {
    if ( (unsigned int)sub_A746B0((_QWORD *)(a2 + 120))
      || !(unsigned __int8)sub_B2D610(a2, 41)
      || (*(_BYTE *)(a2 + 2) & 8) != 0 )
    {
      return 1;
    }
    v2 = *(_QWORD *)(a1 + 208);
    v3 = *(_DWORD *)(v2 + 336);
  }
  if ( !v3 && *(_BYTE *)(v2 + 340) && (unsigned int)sub_A746B0((_QWORD *)(a2 + 120)) )
    return 1;
  result = 2;
  if ( !*(_BYTE *)(a1 + 782) && (*(_BYTE *)(*(_QWORD *)(a1 + 200) + 904LL) & 0x10) == 0 )
    return 0;
  return result;
}
