// Function: sub_15F4DF0
// Address: 0x15f4df0
//
__int64 __fastcall sub_15F4DF0(__int64 a1, unsigned int a2)
{
  __int64 v2; // rsi
  __int64 v3; // rdi
  __int64 result; // rax
  __int64 v5; // rdi

  switch ( *(_BYTE *)(a1 + 16) )
  {
    case 0x18:
    case 0x19:
    case 0x1C:
    case 0x1E:
    case 0x1F:
    case 0x22:
      v2 = a2 + 1;
      if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
        v3 = *(_QWORD *)(a1 - 8);
      else
        v3 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
      result = *(_QWORD *)(v3 + 24 * v2);
      break;
    case 0x1A:
      result = *(_QWORD *)(a1 - 24LL * a2 - 24);
      break;
    case 0x1B:
      if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
        v5 = *(_QWORD *)(a1 - 8);
      else
        v5 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
      result = *(_QWORD *)(v5 + 24LL * (2 * a2 + 1));
      break;
    case 0x1D:
      if ( a2 )
        goto LABEL_6;
      result = *(_QWORD *)(a1 - 48);
      break;
    case 0x20:
      result = 0;
      if ( (*(_BYTE *)(a1 + 18) & 1) != 0 )
        result = *(_QWORD *)(a1 + 24 * (1LL - (*(_DWORD *)(a1 + 20) & 0xFFFFFFF)));
      break;
    case 0x21:
LABEL_6:
      result = *(_QWORD *)(a1 - 24);
      break;
  }
  return result;
}
