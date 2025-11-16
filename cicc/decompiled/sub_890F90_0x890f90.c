// Function: sub_890F90
// Address: 0x890f90
//
__int64 __fastcall sub_890F90(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // r15
  __int64 v7; // rax
  __int64 result; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rax

  v6 = *(_QWORD *)(a4 + 8);
  if ( *(char *)(a2 + 192) < 0 )
  {
    *(_BYTE *)(*(_QWORD *)(a4 + 104) + 121LL) &= ~1u;
  }
  else
  {
    if ( *(_DWORD *)(a1 + 60) )
    {
      v9 = *(_QWORD *)(a4 + 104);
      if ( (*(_BYTE *)(v9 + 121) & 1) == 0 && v6 && !*(_DWORD *)(a1 + 36) )
      {
        sub_6851C0(0x42Fu, (_DWORD *)(a1 + 140));
        v9 = *(_QWORD *)(a4 + 104);
      }
      goto LABEL_23;
    }
    v7 = *(_QWORD *)(a1 + 240);
    if ( v7 && *(char *)(v7 + 177) < 0 && (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v7 + 168) + 160LL) + 121LL) & 1) != 0 )
      goto LABEL_32;
    if ( *(_BYTE *)(a3 + 80) == 20 )
    {
      v10 = *(_QWORD *)(a3 + 88);
      v11 = *(_QWORD *)(v10 + 88);
      if ( v11 )
      {
        if ( (*(_BYTE *)(v10 + 160) & 1) == 0 && v11 != a3 )
        {
          switch ( *(_BYTE *)(v11 + 80) )
          {
            case 4:
            case 5:
              v12 = *(_QWORD *)(*(_QWORD *)(v11 + 96) + 80LL);
              break;
            case 6:
              v12 = *(_QWORD *)(*(_QWORD *)(v11 + 96) + 32LL);
              break;
            case 9:
            case 0xA:
              v12 = *(_QWORD *)(*(_QWORD *)(v11 + 96) + 56LL);
              break;
            case 0x13:
            case 0x14:
            case 0x15:
            case 0x16:
              v12 = *(_QWORD *)(v11 + 88);
              break;
            default:
              BUG();
          }
          if ( (*(_BYTE *)(*(_QWORD *)(v12 + 104) + 121LL) & 3) == 1 )
          {
LABEL_32:
            v9 = *(_QWORD *)(a4 + 104);
LABEL_23:
            *(_BYTE *)(v9 + 121) |= 1u;
          }
        }
      }
    }
  }
  result = *(_QWORD *)(a4 + 104);
  if ( (*(_BYTE *)(result + 121) & 1) == 0 )
    return result;
  if ( *(_BYTE *)(a2 + 172) == 2 )
    goto LABEL_26;
  if ( (unsigned int)sub_8D9730(*(_QWORD *)(a2 + 152)) )
  {
    result = *(_QWORD *)(a4 + 104);
LABEL_26:
    *(_BYTE *)(result + 121) |= 2u;
  }
  result = *(unsigned __int8 *)(*(_QWORD *)(a4 + 104) + 121LL);
  if ( v6 )
  {
    if ( (result & 1) != 0 && (result & 2) == 0 )
    {
      result = *(unsigned int *)(a1 + 52);
      if ( !(_DWORD)result )
      {
        result = (__int64)sub_878440();
        *(_QWORD *)(result + 8) = a3;
        if ( !unk_4D03B70 )
          unk_4D03B70 = result;
        if ( qword_4F601C0 )
          *(_QWORD *)qword_4F601C0 = result;
        qword_4F601C0 = result;
      }
    }
  }
  return result;
}
