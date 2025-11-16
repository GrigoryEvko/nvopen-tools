// Function: sub_246EEA0
// Address: 0x246eea0
//
__int64 __fastcall sub_246EEA0(__int64 a1, __int64 a2, int a3)
{
  __int64 v3; // rax
  __int64 v4; // rsi
  unsigned __int8 *v5; // r12
  unsigned __int8 v6; // dl

  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
  {
    v3 = *(_QWORD *)(a1 + 8);
    v4 = *(_QWORD *)(a2 - 8);
    if ( *(_DWORD *)(v3 + 4) )
      goto LABEL_3;
    return 0;
  }
  v4 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  v3 = *(_QWORD *)(a1 + 8);
  if ( !*(_DWORD *)(v3 + 4) )
    return 0;
LABEL_3:
  if ( !*(_BYTE *)(a1 + 633) )
    return sub_AD6530(*(_QWORD *)(v3 + 88), v4);
  v5 = *(unsigned __int8 **)(v4 + 32LL * a3);
  v6 = *v5;
  if ( *v5 == 25 || v6 <= 0x15u )
    return sub_AD6530(*(_QWORD *)(v3 + 88), v4);
  if ( v6 > 0x1Cu && (v5[7] & 0x20) != 0 )
  {
    v4 = 31;
    if ( sub_B91C10((__int64)v5, 31) )
    {
      v3 = *(_QWORD *)(a1 + 8);
      return sub_AD6530(*(_QWORD *)(v3 + 88), v4);
    }
  }
  return *sub_246EC10(a1 + 384, (__int64)v5);
}
