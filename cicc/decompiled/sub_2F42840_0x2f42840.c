// Function: sub_2F42840
// Address: 0x2f42840
//
__int64 __fastcall sub_2F42840(__int64 a1, unsigned int a2)
{
  __int64 result; // rax
  int v3; // ecx
  unsigned int v4; // ecx
  __int64 v5; // rsi
  __int64 v6; // r8
  unsigned int v7; // eax
  __int64 v8; // r9
  __int64 v9; // r9

  result = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) + 24LL * a2 + 16) & 0xFFF;
  v3 = *(_DWORD *)(*(_QWORD *)(a1 + 808) + 4 * result);
  if ( v3 )
  {
    if ( v3 == 1 )
    {
      return sub_2F42240(a1, a2, 0);
    }
    else
    {
      v4 = v3 & 0x7FFFFFFF;
      v5 = *(unsigned int *)(a1 + 424);
      v6 = *(_QWORD *)(a1 + 416);
      v7 = *(unsigned __int16 *)(*(_QWORD *)(a1 + 624) + 2LL * v4);
      if ( v7 < (unsigned int)v5 )
      {
        while ( 1 )
        {
          v8 = v6 + 24LL * v7;
          if ( v4 == (*(_DWORD *)(v8 + 8) & 0x7FFFFFFF) )
            break;
          v7 += 0x10000;
          if ( (unsigned int)v5 <= v7 )
            goto LABEL_10;
        }
      }
      else
      {
LABEL_10:
        v8 = v6 + 24 * v5;
      }
      sub_2F42240(a1, *(unsigned __int16 *)(v8 + 12), 0);
      *(_WORD *)(v9 + 12) = 0;
      return 0;
    }
  }
  return result;
}
