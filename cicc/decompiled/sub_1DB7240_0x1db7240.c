// Function: sub_1DB7240
// Address: 0x1db7240
//
_QWORD *__fastcall sub_1DB7240(__int64 a1, __int64 *a2)
{
  _QWORD *v3; // rbx
  unsigned int v4; // ecx
  _QWORD *v5; // rdx
  char v6; // si
  unsigned int v7; // eax
  _QWORD *result; // rax
  __int64 v9; // rsi

  v3 = *(_QWORD **)(a1 + 16);
  if ( v3 )
  {
    v4 = *(_DWORD *)((*a2 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*a2 >> 1) & 3;
    while ( 1 )
    {
      v7 = *(_DWORD *)((v3[4] & 0xFFFFFFFFFFFFFFF8LL) + 24) | ((__int64)v3[4] >> 1) & 3;
      if ( v7 > v4
        || v7 >= v4
        && (*(_DWORD *)((a2[1] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(a2[1] >> 1) & 3) < (*(_DWORD *)((v3[5] & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                                | (unsigned int)((__int64)v3[5] >> 1)
                                                                                                & 3) )
      {
        v5 = (_QWORD *)v3[2];
        v6 = 1;
        if ( !v5 )
        {
LABEL_9:
          if ( v6 )
            goto LABEL_10;
LABEL_12:
          if ( v4 <= v7
            && (v4 < v7
             || (*(_DWORD *)((v3[5] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)((__int64)v3[5] >> 1) & 3) >= (*(_DWORD *)((a2[1] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(a2[1] >> 1) & 3)) )
          {
            return v3;
          }
          else
          {
            return 0;
          }
        }
      }
      else
      {
        v5 = (_QWORD *)v3[3];
        v6 = 0;
        if ( !v5 )
          goto LABEL_9;
      }
      v3 = v5;
    }
  }
  v3 = (_QWORD *)(a1 + 8);
LABEL_10:
  result = 0;
  if ( *(_QWORD **)(a1 + 24) != v3 )
  {
    v9 = sub_220EF80(v3);
    v4 = *(_DWORD *)((*a2 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*a2 >> 1) & 3;
    v7 = *(_DWORD *)((*(_QWORD *)(v9 + 32) & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*(__int64 *)(v9 + 32) >> 1) & 3;
    v3 = (_QWORD *)v9;
    goto LABEL_12;
  }
  return result;
}
