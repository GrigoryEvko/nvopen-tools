// Function: sub_890B60
// Address: 0x890b60
//
__int64 __fastcall sub_890B60(__int64 a1)
{
  char v1; // dl
  __int64 v2; // rcx
  __int64 result; // rax
  __int64 v4; // rax
  __int64 v5; // rdx

  v1 = *(_BYTE *)(a1 + 80);
  switch ( v1 )
  {
    case 4:
    case 5:
      v2 = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 80LL);
      break;
    case 6:
      v2 = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 32LL);
      break;
    case 9:
    case 10:
      v2 = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 56LL);
      break;
    case 19:
    case 20:
    case 21:
    case 22:
      v2 = *(_QWORD *)(a1 + 88);
      break;
    default:
LABEL_17:
      BUG();
  }
  result = 1;
  if ( !*(_QWORD *)(v2 + 48) )
  {
    result = 0;
    if ( v1 == 20 )
    {
      v4 = *(_QWORD *)(a1 + 88);
      v5 = *(_QWORD *)(v4 + 88);
      if ( v5 && (*(_BYTE *)(v4 + 160) & 1) == 0 )
      {
        switch ( *(_BYTE *)(v5 + 80) )
        {
          case 4:
          case 5:
            v4 = *(_QWORD *)(*(_QWORD *)(v5 + 96) + 80LL);
            return *(_QWORD *)(v4 + 48) != 0;
          case 6:
            v4 = *(_QWORD *)(*(_QWORD *)(v5 + 96) + 32LL);
            return *(_QWORD *)(v4 + 48) != 0;
          case 9:
          case 0xA:
            v4 = *(_QWORD *)(*(_QWORD *)(v5 + 96) + 56LL);
            return *(_QWORD *)(v4 + 48) != 0;
          case 0x13:
          case 0x14:
          case 0x15:
          case 0x16:
            v4 = *(_QWORD *)(v5 + 88);
            return *(_QWORD *)(v4 + 48) != 0;
          default:
            goto LABEL_17;
        }
      }
      return *(_QWORD *)(v4 + 48) != 0;
    }
  }
  return result;
}
