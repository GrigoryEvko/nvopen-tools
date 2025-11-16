// Function: sub_730770
// Address: 0x730770
//
__int64 __fastcall sub_730770(__int64 a1, _QWORD *a2)
{
  __int64 i; // rbx
  __int64 v3; // rax
  _QWORD *v4; // rax
  __int64 v5; // r8

  for ( i = *(_QWORD *)(sub_730290(a1) + 64); ; i = *(_QWORD *)(v3 + 56) )
  {
    for ( ; sub_730740(i); i = *(_QWORD *)(i + 72) )
    {
      if ( (*(_BYTE *)(i + 27) & 2) == 0 )
        break;
    }
    if ( *(_BYTE *)(i + 24) != 5 )
      break;
    v3 = *(_QWORD *)(i + 56);
    if ( (*(_BYTE *)(v3 + 51) & 8) == 0 )
      break;
  }
  v4 = *(_QWORD **)(i + 72);
  v5 = v4[7];
  if ( a2 )
    *a2 = *v4;
  return v5;
}
