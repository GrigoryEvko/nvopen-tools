// Function: sub_B46EC0
// Address: 0xb46ec0
//
__int64 __fastcall sub_B46EC0(__int64 a1, unsigned int a2)
{
  __int64 result; // rax
  __int64 v3; // rax

  switch ( *(_BYTE *)a1 )
  {
    case 0x1F:
      return *(_QWORD *)(a1 - 32LL * a2 - 32);
    case 0x20:
      return *(_QWORD *)(*(_QWORD *)(a1 - 8) + 32LL * (2 * a2 + 1));
    case 0x21:
    case 0x27:
      return *(_QWORD *)(*(_QWORD *)(a1 - 8) + 32LL * (a2 + 1));
    case 0x22:
      if ( a2 )
        goto LABEL_8;
      result = *(_QWORD *)(a1 - 96);
      break;
    case 0x25:
      result = 0;
      if ( (*(_BYTE *)(a1 + 2) & 1) != 0 )
        return *(_QWORD *)(a1 + 32 * (1LL - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF)));
      return result;
    case 0x26:
      return *(_QWORD *)(a1 - 32);
    case 0x28:
      v3 = *(unsigned int *)(a1 + 88);
      if ( a2 )
      {
        result = *(_QWORD *)(a1 - 32 + 32 * (a2 - 1 - v3));
      }
      else
      {
        a1 -= 32 * v3;
LABEL_8:
        result = *(_QWORD *)(a1 - 64);
      }
      break;
    default:
      BUG();
  }
  return result;
}
