// Function: sub_643C90
// Address: 0x643c90
//
__int64 __fastcall sub_643C90(__int64 a1)
{
  __int64 result; // rax

  result = word_4F06418[0];
  if ( word_4F06418[0] > 0xC1u )
    goto LABEL_7;
  if ( word_4F06418[0] <= 0x9Fu )
  {
    if ( word_4F06418[0] == 75 || word_4F06418[0] == 103 )
      return result;
LABEL_7:
    *(_DWORD *)(a1 + 64) = dword_4F06650[0];
    sub_7BDB60(1);
    *(_BYTE *)(a1 + 133) |= 0x20u;
    result = qword_4F04C68[0] + 776LL * dword_4F04C64;
    *(_BYTE *)(result + 12) |= 8u;
    return result;
  }
  result = (unsigned __int16)(word_4F06418[0] - 160);
  switch ( word_4F06418[0] )
  {
    case 0xA0u:
    case 0xAFu:
    case 0xB3u:
    case 0xB8u:
    case 0xC0u:
    case 0xC1u:
      return result;
    default:
      goto LABEL_7;
  }
  return result;
}
