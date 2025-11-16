// Function: sub_2F612C0
// Address: 0x2f612c0
//
__int64 __fastcall sub_2F612C0(int a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v4; // rcx

  if ( a1 < 0 )
    v3 = *(_QWORD *)(*(_QWORD *)(a3 + 56) + 16LL * (a1 & 0x7FFFFFFF) + 8);
  else
    v3 = *(_QWORD *)(*(_QWORD *)(a3 + 304) + 8LL * (unsigned int)a1);
  if ( v3 )
  {
    if ( (*(_BYTE *)(v3 + 4) & 8) != 0 )
    {
      while ( 1 )
      {
        v3 = *(_QWORD *)(v3 + 32);
        if ( !v3 )
          break;
        if ( (*(_BYTE *)(v3 + 4) & 8) == 0 )
          goto LABEL_5;
      }
    }
    else
    {
LABEL_5:
      v4 = *(_QWORD *)(v3 + 16);
LABEL_6:
      if ( a2 != v4 && ((*(_WORD *)(v4 + 68) - 12) & 0xFFF7) == 0 )
        return 0;
      while ( 1 )
      {
        v3 = *(_QWORD *)(v3 + 32);
        if ( !v3 )
          break;
        if ( (*(_BYTE *)(v3 + 4) & 8) == 0 && *(_QWORD *)(v3 + 16) != v4 )
        {
          v4 = *(_QWORD *)(v3 + 16);
          goto LABEL_6;
        }
      }
    }
  }
  return 1;
}
