// Function: sub_862ED0
// Address: 0x862ed0
//
__int64 sub_862ED0()
{
  __int64 result; // rax
  __int64 v1; // rdx
  __int64 v2; // rcx
  __int64 v3; // rsi
  char v4; // cl

  if ( dword_4F04C58 != -1 || (result = dword_4F04C38) != 0 )
  {
    v1 = qword_4F04C68[0];
    result = qword_4F04C68[0] + 776LL * dword_4F04C64;
    v2 = *(int *)(result + 192);
    if ( (_DWORD)v2 == dword_4F073B8[0] )
      goto LABEL_7;
    *(_BYTE *)(*((_QWORD *)qword_4F072B0 + v2) + 29LL) |= 2u;
    if ( *(_BYTE *)(result + 4) == 17 )
      goto LABEL_8;
LABEL_5:
    result = *(int *)(result + 552);
    if ( (_DWORD)result != -1 )
    {
      while ( 1 )
      {
        result = v1 + 776 * result;
        if ( !result )
          break;
LABEL_7:
        if ( *(_BYTE *)(result + 4) != 17 )
          goto LABEL_5;
LABEL_8:
        v3 = *(_QWORD *)(result + 216);
        v4 = *(_BYTE *)(v3 + 207);
        if ( (v4 & 0x40) == 0 )
        {
          *(_BYTE *)(v3 + 207) = v4 | 0x40;
          result = *(int *)(result + 552);
          if ( (_DWORD)result != -1 )
            continue;
        }
        return result;
      }
    }
  }
  return result;
}
