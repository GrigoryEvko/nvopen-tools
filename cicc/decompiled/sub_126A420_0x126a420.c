// Function: sub_126A420
// Address: 0x126a420
//
bool __fastcall sub_126A420(__int64 a1, unsigned __int64 a2)
{
  _QWORD *v2; // rax
  _QWORD *v3; // r8
  _QWORD *v4; // rdi
  __int64 v5; // rcx
  __int64 v6; // rdx
  bool result; // al

  v2 = *(_QWORD **)(a1 + 552);
  v3 = (_QWORD *)(a1 + 544);
  if ( v2 )
  {
    v4 = (_QWORD *)(a1 + 544);
    do
    {
      while ( 1 )
      {
        v5 = v2[2];
        v6 = v2[3];
        if ( v2[4] >= a2 )
          break;
        v2 = (_QWORD *)v2[3];
        if ( !v6 )
          goto LABEL_6;
      }
      v4 = v2;
      v2 = (_QWORD *)v2[2];
    }
    while ( v5 );
LABEL_6:
    if ( v3 != v4 && v4[4] <= a2 )
      return 0;
  }
  switch ( *(_DWORD *)(*(_QWORD *)a2 + 8LL) >> 8 )
  {
    case 0:
      result = unk_4D04640 == 1;
      break;
    case 1:
      result = unk_4D04650 == 1;
      break;
    case 3:
      result = unk_4D0464C == 1;
      break;
    case 4:
      result = unk_4D04648 == 1;
      break;
    case 5:
      result = unk_4D04644 == 1;
      break;
    default:
      result = 0;
      break;
  }
  return result;
}
