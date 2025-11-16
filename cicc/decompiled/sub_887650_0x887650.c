// Function: sub_887650
// Address: 0x887650
//
__int64 __fastcall sub_887650(__int64 a1)
{
  _DWORD *v1; // rsi
  __int64 v2; // rdx
  __int64 *v3; // rax

  v1 = (_DWORD *)(a1 + 8);
  v2 = *(_QWORD *)(*(_QWORD *)a1 + 8LL);
  if ( (*(_BYTE *)(a1 + 18) & 2) != 0 )
    return sub_6851A0(0x9F6u, v1, v2);
  v3 = *(__int64 **)(a1 + 32);
  if ( !v3 )
    return sub_6851A0(0x9F6u, v1, v2);
  else
    return sub_686A10(0x2E6u, v1, v2, *v3);
}
