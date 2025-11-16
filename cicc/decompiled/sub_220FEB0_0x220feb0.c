// Function: sub_220FEB0
// Address: 0x220feb0
//
__int64 __fastcall sub_220FEB0(__int64 a1, unsigned int a2)
{
  _BYTE *v2; // rdx
  __int64 result; // rax
  _BYTE *v4; // rdx
  _BYTE *v5; // rdx

  if ( a2 > 0x7FF )
  {
    if ( a2 <= 0xFFFF )
    {
      v5 = *(_BYTE **)a1;
      result = 0;
      if ( *(_QWORD *)(a1 + 8) - *(_QWORD *)a1 > 2u )
      {
        *v5 = (a2 >> 12) - 32;
        v5[1] = ((a2 >> 6) & 0x3F) + 0x80;
        *(_QWORD *)a1 = v5 + 3;
        v5[2] = (a2 & 0x3F) + 0x80;
        return 1;
      }
    }
    else
    {
      result = 0;
      if ( a2 <= 0x10FFFF )
      {
        v4 = *(_BYTE **)a1;
        if ( *(_QWORD *)(a1 + 8) - *(_QWORD *)a1 > 3u )
        {
          *v4 = (a2 >> 18) - 16;
          v4[1] = ((a2 >> 12) & 0x3F) + 0x80;
          v4[2] = ((a2 >> 6) & 0x3F) + 0x80;
          *(_QWORD *)a1 = v4 + 4;
          result = 1;
          v4[3] = (a2 & 0x3F) + 0x80;
        }
      }
    }
  }
  else
  {
    v2 = *(_BYTE **)a1;
    result = 0;
    if ( *(_QWORD *)(a1 + 8) - *(_QWORD *)a1 > 1u )
    {
      *v2 = (a2 >> 6) - 64;
      *(_QWORD *)a1 = v2 + 2;
      v2[1] = (a2 & 0x3F) + 0x80;
      return 1;
    }
  }
  return result;
}
