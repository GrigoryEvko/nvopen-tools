// Function: sub_EA19F0
// Address: 0xea19f0
//
__int64 __fastcall sub_EA19F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v6; // ebx
  __int64 result; // rax

  v6 = a3;
  sub_E5CB20(*(_QWORD *)(a1 + 296), a2, a3, a4, a5, a6);
  switch ( v6 )
  {
    case 0:
    case 13:
    case 14:
    case 16:
    case 19:
    case 21:
    case 22:
    case 23:
    case 25:
    case 27:
      result = 0;
      break;
    case 1:
    case 4:
      result = 1;
      break;
    case 2:
      *(_BYTE *)(a2 + 36) = 1;
      *(_DWORD *)(a2 + 32) = 0;
      result = 1;
      break;
    case 5:
      *(_WORD *)(a2 + 12) |= 0x100u;
      result = 1;
      break;
    case 9:
      *(_BYTE *)(a2 + 8) |= 0x20u;
      result = 1;
      break;
    case 12:
      *(_BYTE *)(a2 + 41) = 1;
      result = 1;
      break;
    case 18:
      *(_WORD *)(a2 + 12) |= 0x80u;
      result = 1;
      break;
    case 24:
    case 26:
      *(_BYTE *)(a2 + 8) |= 0x20u;
      *(_BYTE *)(a2 + 40) = 1;
      result = 1;
      break;
    default:
      BUG();
  }
  return result;
}
