// Function: sub_34E0B20
// Address: 0x34e0b20
//
__int64 __fastcall sub_34E0B20(_QWORD *a1, __int64 a2, unsigned int a3, unsigned int a4)
{
  __int64 result; // rax
  __int64 v7; // rsi
  unsigned int v10; // eax
  __int64 v11; // rdx
  unsigned int v12; // ecx

  result = *(unsigned __int16 *)(a2 + 68);
  if ( (unsigned __int16)(result - 14) > 4u && (_WORD)result != 7 )
  {
    v7 = a1[4];
    if ( *(_DWORD *)(v7 + 16) == 1 )
    {
LABEL_12:
      sub_34E00F0(a1, a2);
      return sub_34E0650((__int64)a1, a2, a3);
    }
    v10 = 1;
    while ( 1 )
    {
      while ( 1 )
      {
        v11 = v10;
        if ( *(_DWORD *)(a1[24] + 4LL * v10) == -1 )
          break;
        *(_QWORD *)(a1[15] + 8LL * v10) = -1;
        *(_DWORD *)(a1[24] + 4LL * v10) = a3;
        v7 = a1[4];
LABEL_7:
        if ( *(_DWORD *)(v7 + 16) == ++v10 )
          goto LABEL_12;
      }
      v12 = *(_DWORD *)(a1[27] + 4LL * v10);
      if ( a3 > v12 || a4 <= v12 )
        goto LABEL_7;
      ++v10;
      *(_QWORD *)(a1[15] + 8 * v11) = -1;
      *(_DWORD *)(a1[27] + 4 * v11) = a4;
      v7 = a1[4];
      if ( *(_DWORD *)(v7 + 16) == v10 )
        goto LABEL_12;
    }
  }
  return result;
}
