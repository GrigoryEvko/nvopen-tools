// Function: sub_7D3810
// Address: 0x7d3810
//
__int64 __fastcall sub_7D3810(unsigned __int8 a1)
{
  __int64 v1; // rdx
  __int64 result; // rax
  __int64 v3; // rsi
  unsigned __int64 v4; // rcx

  v1 = qword_4D04A60[a1];
  if ( !v1 )
    return 0;
  result = *(_QWORD *)(v1 + 32);
  if ( (*(_BYTE *)(qword_4F04C68[0] + 10LL) & 4) == 0 )
    result = *(_QWORD *)(v1 + 24);
  if ( result )
  {
    v3 = 1182720;
    do
    {
      if ( (*(_BYTE *)(result + 81) & 0x10) == 0 && *(_DWORD *)(result + 40) == unk_4F066A8 )
      {
        v4 = *(unsigned __int8 *)(result + 80);
        if ( (unsigned __int8)v4 <= 0x14u )
        {
          if ( _bittest64(&v3, v4) )
            break;
        }
      }
      result = *(_QWORD *)(result + 8);
    }
    while ( result );
  }
  return result;
}
