// Function: sub_8C7240
// Address: 0x8c7240
//
_BOOL8 __fastcall sub_8C7240(_QWORD *a1)
{
  char *v1; // rax
  char v2; // al
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // rax
  __int64 i; // rax

  v1 = *(char **)(*(_QWORD *)(a1[19] + 168LL) + 56LL);
  if ( !v1 )
    return 0;
  v2 = *v1;
  if ( (v2 & 2) != 0 )
    return 1;
  if ( (v2 & 0x40) == 0 )
    return 0;
  v4 = *(_QWORD *)(*(_QWORD *)(*a1 + 96LL) + 32LL);
  v5 = *(_QWORD *)(v4 + 88);
  if ( *(_QWORD *)(v5 + 88) && (*(_BYTE *)(v5 + 160) & 1) == 0 )
    v4 = *(_QWORD *)(v5 + 88);
  switch ( *(_BYTE *)(v4 + 80) )
  {
    case 4:
    case 5:
      v6 = *(_QWORD *)(*(_QWORD *)(v4 + 96) + 80LL);
      break;
    case 6:
      v6 = *(_QWORD *)(*(_QWORD *)(v4 + 96) + 32LL);
      break;
    case 9:
    case 0xA:
      v6 = *(_QWORD *)(*(_QWORD *)(v4 + 96) + 56LL);
      break;
    case 0x13:
    case 0x14:
    case 0x15:
    case 0x16:
      v6 = *(_QWORD *)(v4 + 88);
      break;
    default:
      BUG();
  }
  for ( i = *(_QWORD *)(*(_QWORD *)(v6 + 176) + 152LL); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  return (**(_BYTE **)(*(_QWORD *)(i + 168) + 56LL) & 6) != 0;
}
