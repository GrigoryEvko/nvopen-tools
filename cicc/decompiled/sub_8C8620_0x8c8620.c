// Function: sub_8C8620
// Address: 0x8c8620
//
__int64 __fastcall sub_8C8620(__int64 a1, __int64 a2)
{
  __int64 v3; // r14
  __int64 v4; // rbx
  __int64 v5; // rdx
  __int64 v6; // r15
  __int64 v7; // rcx
  _UNKNOWN *__ptr32 *v8; // r8
  __int64 v9; // rbx
  __int64 v11; // rax
  __int64 v12; // rdi
  __int64 v13; // rsi

  switch ( *(_BYTE *)(a1 + 80) )
  {
    case 4:
    case 5:
      v3 = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 80LL);
      break;
    case 6:
      v3 = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 32LL);
      break;
    case 9:
    case 0xA:
      v3 = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 56LL);
      break;
    case 0x13:
    case 0x14:
    case 0x15:
    case 0x16:
      v3 = *(_QWORD *)(a1 + 88);
      break;
    default:
      v3 = 0;
      break;
  }
  switch ( *(_BYTE *)(a2 + 80) )
  {
    case 4:
    case 5:
      v4 = *(_QWORD *)(*(_QWORD *)(a2 + 96) + 80LL);
      break;
    case 6:
      v4 = *(_QWORD *)(*(_QWORD *)(a2 + 96) + 32LL);
      break;
    case 9:
    case 0xA:
      v4 = *(_QWORD *)(*(_QWORD *)(a2 + 96) + 56LL);
      break;
    case 0x13:
    case 0x14:
    case 0x15:
    case 0x16:
      v4 = *(_QWORD *)(a2 + 88);
      break;
    default:
      BUG();
  }
  if ( ((*(_BYTE *)(v4 + 265) ^ *(_BYTE *)(v3 + 265)) & 1) != 0 )
    return 0;
  v5 = *(_QWORD *)(v3 + 152);
  if ( !v5 )
  {
    if ( (unsigned int)sub_89B3C0(**(_QWORD **)(v4 + 32), **(_QWORD **)(v3 + 32), 1, 0, (_DWORD *)(a1 + 48), 8u) )
      return *(_QWORD *)(v4 + 104);
    return 0;
  }
  switch ( *(_BYTE *)(v5 + 80) )
  {
    case 4:
    case 5:
      v11 = *(_QWORD *)(*(_QWORD *)(v5 + 96) + 80LL);
      break;
    case 6:
      v11 = *(_QWORD *)(*(_QWORD *)(v5 + 96) + 32LL);
      break;
    case 9:
    case 0xA:
      v11 = *(_QWORD *)(*(_QWORD *)(v5 + 96) + 56LL);
      break;
    case 0x13:
    case 0x14:
    case 0x15:
    case 0x16:
      v11 = *(_QWORD *)(v5 + 88);
      break;
    default:
      BUG();
  }
  v12 = *(_QWORD *)(v11 + 104);
  v13 = *(_QWORD *)(v4 + 104);
  if ( v12 != v13 && (!*qword_4D03FD0 || !v12 || !v13 || !(unsigned int)sub_8C7EB0(v12, v13, 0x3Bu)) )
    return 0;
  v9 = *(_QWORD *)(v4 + 144);
  if ( !v9 )
    return 0;
  while ( 1 )
  {
    switch ( *(_BYTE *)(v9 + 80) )
    {
      case 4:
      case 5:
        v6 = *(_QWORD *)(*(_QWORD *)(v9 + 96) + 80LL);
        break;
      case 6:
        v6 = *(_QWORD *)(*(_QWORD *)(v9 + 96) + 32LL);
        break;
      case 9:
      case 0xA:
        v6 = *(_QWORD *)(*(_QWORD *)(v9 + 96) + 56LL);
        break;
      case 0x13:
      case 0x14:
      case 0x15:
      case 0x16:
        v6 = *(_QWORD *)(v9 + 88);
        break;
      default:
        BUG();
    }
    if ( (unsigned int)sub_89B3C0(**(_QWORD **)(v6 + 32), **(_QWORD **)(v3 + 32), 0, 0, (_DWORD *)(a1 + 48), 8u)
      && sub_89AB40(
           *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v3 + 176) + 88LL) + 168LL) + 168LL),
           *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v6 + 176) + 88LL) + 168LL) + 168LL),
           0,
           v7,
           v8) )
    {
      break;
    }
    v9 = *(_QWORD *)(v9 + 8);
    if ( !v9 )
      return 0;
  }
  return *(_QWORD *)(v6 + 104);
}
