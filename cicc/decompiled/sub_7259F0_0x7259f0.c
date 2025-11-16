// Function: sub_7259F0
// Address: 0x7259f0
//
__int64 __fastcall sub_7259F0(__int64 a1, unsigned __int8 a2)
{
  int v2; // edx
  int v3; // eax
  __int64 result; // rax

  *(_BYTE *)(a1 + 48) = a2;
  v2 = a2;
  switch ( a2 )
  {
    case 0u:
    case 1u:
      return result;
    case 2u:
    case 6u:
    case 8u:
    case 9u:
      v3 = *(unsigned __int8 *)(a1 + 72);
      *(_QWORD *)(a1 + 56) = 0;
      LOBYTE(v2) = a2 == 6;
      *(_QWORD *)(a1 + 64) = 0;
      result = v2 | v3 & 0xFFFFFFFE;
      *(_BYTE *)(a1 + 72) = result;
      break;
    case 3u:
    case 4u:
    case 7u:
      *(_QWORD *)(a1 + 56) = 0;
      break;
    case 5u:
      *(_BYTE *)(a1 + 72) &= 0xE0u;
      *(_QWORD *)(a1 + 56) = 0;
      *(_QWORD *)(a1 + 64) = 0;
      break;
    default:
      sub_721090();
  }
  return result;
}
