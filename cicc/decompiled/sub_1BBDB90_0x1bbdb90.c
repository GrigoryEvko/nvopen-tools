// Function: sub_1BBDB90
// Address: 0x1bbdb90
//
__int64 __fastcall sub_1BBDB90(__int64 a1, __int64 a2, __int64 a3)
{
  int v5; // ebx
  unsigned int v6; // r14d
  int v7; // edx
  __int64 v8; // rsi
  unsigned int v9; // eax
  int v10; // ecx
  int v11; // eax
  int v13; // edi
  int v14; // [rsp+Ch] [rbp-34h]

  v14 = *(_QWORD *)(a2 + 32);
  if ( v14 )
  {
    v5 = 0;
    v6 = 0;
    while ( 1 )
    {
      while ( 1 )
      {
        v11 = *(_DWORD *)(a3 + 24);
        if ( v11 )
          break;
LABEL_6:
        ++v5;
        v6 += sub_14A3470(*(_QWORD *)(a1 + 1320));
        if ( v5 == v14 )
          goto LABEL_7;
      }
      v7 = v11 - 1;
      v8 = *(_QWORD *)(a3 + 8);
      v9 = (v11 - 1) & (37 * v5);
      v10 = *(_DWORD *)(v8 + 4LL * v9);
      if ( v5 != v10 )
      {
        v13 = 1;
        while ( v10 != -1 )
        {
          v9 = v7 & (v13 + v9);
          v10 = *(_DWORD *)(v8 + 4LL * v9);
          if ( v10 == v5 )
            goto LABEL_4;
          ++v13;
        }
        goto LABEL_6;
      }
LABEL_4:
      if ( ++v5 == v14 )
        goto LABEL_7;
    }
  }
  v6 = 0;
LABEL_7:
  if ( *(_DWORD *)(a3 + 16) )
    v6 += sub_14A3380(*(_QWORD *)(a1 + 1320));
  return v6;
}
