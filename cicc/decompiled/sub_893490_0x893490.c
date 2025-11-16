// Function: sub_893490
// Address: 0x893490
//
__int64 __fastcall sub_893490(__int64 a1)
{
  __int64 result; // rax

  switch ( *(_BYTE *)(a1 + 80) )
  {
    case 4:
    case 5:
      result = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 80LL);
      ++*(_DWORD *)(result + 40);
      break;
    case 6:
      result = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 32LL);
      ++*(_DWORD *)(result + 40);
      break;
    case 9:
    case 0xA:
      result = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 56LL);
      ++*(_DWORD *)(result + 40);
      break;
    case 0x13:
    case 0x14:
    case 0x15:
    case 0x16:
      result = *(_QWORD *)(a1 + 88);
      ++*(_DWORD *)(result + 40);
      break;
    default:
      BUG();
  }
  return result;
}
