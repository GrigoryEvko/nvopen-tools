// Function: sub_890550
// Address: 0x890550
//
__int64 __fastcall sub_890550(__int64 a1)
{
  __int64 v1; // rcx
  __int64 result; // rax
  char v3; // dl
  _QWORD *v4; // rdi
  int v5; // esi
  __int64 v6; // rdx
  __int64 i; // rdx
  __int64 v8; // rdx

  switch ( *(_BYTE *)(a1 + 80) )
  {
    case 4:
    case 5:
      v1 = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 80LL);
      break;
    case 6:
      v1 = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 32LL);
      break;
    case 9:
    case 0xA:
      v1 = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 56LL);
      break;
    case 0x13:
    case 0x14:
    case 0x15:
    case 0x16:
      v1 = *(_QWORD *)(a1 + 88);
      break;
    default:
      BUG();
  }
  result = *(_QWORD *)(v1 + 216);
  v3 = *(_BYTE *)(result + 80);
  if ( v3 == 17 )
  {
    v4 = (_QWORD *)(result + 88);
    result = *(_QWORD *)(result + 88);
    if ( !result )
      goto LABEL_15;
    v3 = *(_BYTE *)(result + 80);
    v5 = 1;
  }
  else
  {
    v4 = (_QWORD *)(v1 + 216);
    v5 = 0;
  }
  if ( v3 == 20 )
  {
    v6 = *(_QWORD *)(result + 88);
    if ( !*(_QWORD *)(v6 + 416) )
      goto LABEL_11;
  }
  while ( v5 )
  {
    result = *(_QWORD *)(result + 8);
    if ( !result )
      break;
    if ( *(_BYTE *)(result + 80) != 20 )
      continue;
    v6 = *(_QWORD *)(result + 88);
    if ( *(_QWORD *)(v6 + 416) )
      continue;
LABEL_11:
    for ( i = *(_QWORD *)(*(_QWORD *)(v6 + 176) + 152LL); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    v8 = **(_QWORD **)(i + 168);
    if ( !v8 )
    {
      if ( v5 )
        v8 = *(_QWORD *)(result + 8);
      *v4 = v8;
      continue;
    }
    v4 = (_QWORD *)(result + 8);
  }
LABEL_15:
  *(_BYTE *)(v1 + 267) &= ~2u;
  return result;
}
