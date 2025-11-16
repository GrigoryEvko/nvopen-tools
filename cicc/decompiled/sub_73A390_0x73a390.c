// Function: sub_73A390
// Address: 0x73a390
//
__int64 __fastcall sub_73A390(__int64 a1, __int64 a2, __int64 a3, __int64 a4, _UNKNOWN *__ptr32 *a5)
{
  __int64 v5; // rax
  __int64 result; // rax

  v5 = *(_QWORD *)a1;
  if ( *(_QWORD *)a1 )
  {
    if ( (*(_BYTE *)(v5 + 81) & 0x40) != 0 )
    {
      return sub_739430(a1, *(_QWORD *)(v5 + 88), 1u, a4, a5);
    }
    else
    {
      result = 1;
      if ( *(_QWORD *)(a1 + 8) )
      {
        if ( *(_BYTE *)(a1 + 173) != 12 )
          return (*(_BYTE *)(a1 + 170) & 0x10) != 0;
      }
    }
  }
  else
  {
    result = 0;
    if ( !*(_QWORD *)(a1 + 144) && !*(_DWORD *)(a1 + 64) && (*(_BYTE *)(a1 + 171) & 4) == 0 )
    {
      switch ( *(_BYTE *)(a1 + 173) )
      {
        case 2:
          result = unk_4F068AC;
          break;
        case 6:
          result = 0;
          if ( !*(_QWORD *)(a1 + 200) )
          {
            result = 1;
            if ( *(_BYTE *)(a1 + 176) == 4 )
              goto LABEL_16;
          }
          break;
        case 7:
LABEL_16:
          result = *(_QWORD *)(a1 + 184) == 0;
          break;
        case 0xA:
        case 0xC:
          result = 0;
          break;
        default:
          result = 1;
          break;
      }
    }
  }
  return result;
}
