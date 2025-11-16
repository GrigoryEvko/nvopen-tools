// Function: sub_20E7220
// Address: 0x20e7220
//
__int64 __fastcall sub_20E7220(_QWORD *a1, __int64 a2, unsigned int a3, unsigned int a4)
{
  __int64 result; // rax
  __int64 v7; // rsi
  unsigned int v10; // eax
  __int64 v11; // rdx
  unsigned int v12; // ecx

  result = **(unsigned __int16 **)(a2 + 16);
  if ( (unsigned __int16)(result - 12) > 1u && (_WORD)result != 6 )
  {
    v7 = a1[4];
    if ( *(_DWORD *)(v7 + 16) )
    {
      v10 = 0;
      do
      {
        while ( 1 )
        {
          v11 = v10;
          if ( *(_DWORD *)(a1[18] + 4LL * v10) == -1 )
            break;
          *(_QWORD *)(a1[9] + 8LL * v10) = -1;
          *(_DWORD *)(a1[18] + 4LL * v10) = a3;
          v7 = a1[4];
LABEL_6:
          if ( *(_DWORD *)(v7 + 16) == ++v10 )
            goto LABEL_11;
        }
        v12 = *(_DWORD *)(a1[21] + 4LL * v10);
        if ( a3 > v12 || a4 <= v12 )
          goto LABEL_6;
        ++v10;
        *(_QWORD *)(a1[9] + 8 * v11) = -1;
        *(_DWORD *)(a1[21] + 4 * v11) = a4;
        v7 = a1[4];
      }
      while ( *(_DWORD *)(v7 + 16) != v10 );
    }
LABEL_11:
    sub_20E6570((__int64)a1, a2);
    return sub_20E6C60((__int64)a1, a2, a3);
  }
  return result;
}
