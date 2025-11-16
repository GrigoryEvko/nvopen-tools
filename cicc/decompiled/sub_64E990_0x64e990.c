// Function: sub_64E990
// Address: 0x64e990
//
__int64 __fastcall sub_64E990(__int64 a1, __int64 a2, int a3, int a4, int a5, int a6)
{
  __int64 v11; // rdx
  __int64 result; // rax
  char v13; // cl
  __int64 v14; // rsi
  int v15; // esi
  __int64 v16; // rdi

  v11 = sub_8D4940(a2);
  result = *(unsigned __int8 *)(v11 + 140);
  switch ( (_BYTE)result )
  {
    case 0xC:
      result = v11;
      do
      {
        result = *(_QWORD *)(result + 160);
        v13 = *(_BYTE *)(result + 140);
      }
      while ( v13 == 12 );
      if ( !v13 )
        return result;
      do
      {
        v11 = *(_QWORD *)(v11 + 160);
        result = *(unsigned __int8 *)(v11 + 140);
      }
      while ( (_BYTE)result == 12 );
      break;
    case 0:
      return result;
    case 0x15:
      return result;
  }
  v14 = 938;
  if ( !a5 )
  {
    v15 = -(a6 == 0);
    LOBYTE(v15) = v15 & 0x49;
    v14 = (unsigned int)(v15 + 260);
  }
  result = (unsigned int)dword_4F077C4;
  if ( dword_4F077C4 == 1 )
  {
    if ( !a3 )
      goto LABEL_18;
  }
  else
  {
    if ( dword_4F077C4 == 2 )
    {
      if ( qword_4D0495C && !unk_4F0775C )
      {
        v16 = 4;
        if ( !a5 )
        {
          if ( a3 )
          {
            v14 = 837;
          }
          else
          {
            v16 = a6 == 0 ? 7 : 5;
            if ( a6 )
              v14 = 837;
          }
        }
        return sub_684AA0(v16, v14, a1);
      }
LABEL_13:
      if ( !a5 )
      {
        v16 = 7;
        if ( a3 )
          v14 = 260;
        return sub_684AA0(v16, v14, a1);
      }
      v16 = 4;
      if ( !(unk_4F0775C | dword_4D04964) )
        return sub_684AA0(v16, v14, a1);
LABEL_15:
      result = (__int64)&unk_4F07471;
      v16 = unk_4F07471;
      if ( unk_4F07471 == 3 )
        return result;
      return sub_684AA0(v16, v14, a1);
    }
    result = unk_4F07778;
    if ( unk_4F07778 > 199900 && !dword_4F077C0 )
      goto LABEL_13;
    if ( !a3 )
    {
      v16 = (dword_4F077C0 | a6) == 0 ? 7 : 5;
      return sub_684AA0(v16, v14, a1);
    }
    if ( !a5 )
    {
      if ( a4 | a6 )
      {
        if ( !dword_4F077C0 || (v16 = 5, v14 = 260, unk_4F07778 <= 199900) )
        {
          v16 = 4;
          v14 = 260;
        }
        return sub_684AA0(v16, v14, a1);
      }
      if ( dword_4D04964 )
        goto LABEL_15;
LABEL_18:
      v16 = 5;
      return sub_684AA0(v16, v14, a1);
    }
  }
  return result;
}
