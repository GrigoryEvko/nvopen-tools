// Function: sub_8AE140
// Address: 0x8ae140
//
__int64 __fastcall sub_8AE140(__int64 *a1, __int64 a2, _DWORD *a3)
{
  __int64 v4; // r12
  char i; // al
  unsigned int v6; // r12d
  __int64 result; // rax
  unsigned int v8; // edi

  v4 = *a1;
  for ( i = *(_BYTE *)(*a1 + 140); i == 12; i = *(_BYTE *)(v4 + 140) )
    v4 = *(_QWORD *)(v4 + 160);
  switch ( i )
  {
    case 1:
      v8 = 598;
      goto LABEL_15;
    case 3:
      if ( dword_4D04800 )
        goto LABEL_5;
      v8 = 605;
      goto LABEL_15;
    case 4:
    case 5:
      v8 = 2746;
      goto LABEL_15;
    case 6:
      if ( (*(_BYTE *)(v4 + 168) & 2) == 0 || (_DWORD)a2 )
        goto LABEL_5;
      v8 = 1769;
LABEL_15:
      v6 = 0;
      if ( !a3 )
        goto LABEL_6;
      sub_6851C0(v8, a3);
      result = 0;
      break;
    case 7:
    case 8:
      sub_645520(a1);
      goto LABEL_5;
    case 9:
    case 10:
    case 11:
      if ( (*(_BYTE *)(v4 + 177) & 0x20) != 0 )
        goto LABEL_5;
      v8 = 952;
      if ( dword_4F077C4 == 2 && unk_4F07778 > 202001 )
      {
        if ( (unsigned int)sub_8D23B0(v4) )
          sub_8AE000(v4);
        v8 = 3139;
        if ( (unsigned int)sub_8D42F0(v4, a2) )
          goto LABEL_5;
      }
      goto LABEL_15;
    case 15:
    case 16:
      v8 = 1718;
      goto LABEL_15;
    default:
LABEL_5:
      v6 = 1;
LABEL_6:
      result = v6;
      break;
  }
  return result;
}
