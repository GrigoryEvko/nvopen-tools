// Function: sub_8DCE90
// Address: 0x8dce90
//
__int64 *__fastcall sub_8DCE90(__int64 a1)
{
  __int64 *result; // rax
  char v2; // al
  char v3; // al
  _QWORD *v4; // rax
  __int64 v5; // rax

  if ( !dword_4F07588
    || dword_4F04C64 == -1
    || dword_4F04C44 == -1
    && (v5 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v5 + 6) & 6) == 0)
    && *(_BYTE *)(v5 + 4) != 12
    || (result = (__int64 *)&dword_4F07590, dword_4F07590) )
  {
    result = (__int64 *)*(unsigned __int8 *)(a1 + 141);
    if ( ((unsigned __int8)result & 0x40) == 0 )
    {
      *(_BYTE *)(a1 + 141) = (unsigned __int8)result | 0x40;
      sub_8DCD80(a1);
      if ( *(_QWORD *)(a1 + 8) )
        return sub_8DCD50(a1);
      v2 = *(_BYTE *)(a1 + 140);
      if ( (unsigned __int8)(v2 - 9) <= 2u || v2 == 2 && (*(_BYTE *)(a1 + 161) & 8) != 0 )
        return sub_8DCD50(a1);
      if ( (unsigned int)sub_8DBE70(a1) )
        return sub_8DCD50(a1);
      v3 = *(_BYTE *)(a1 - 8) & 2;
      if ( dword_4D03FE8[0] )
      {
        if ( v3 )
        {
          v4 = (_QWORD *)(*qword_4D03FD0 + 184LL);
LABEL_16:
          *(_QWORD *)(a1 + 112) = v4[16];
          v4[16] = a1;
          return sub_8DCD50(a1);
        }
      }
      else if ( !v3 )
      {
        v4 = qword_4D03FD0 + 23;
        goto LABEL_16;
      }
      v4 = &qword_4F07280;
      goto LABEL_16;
    }
  }
  return result;
}
