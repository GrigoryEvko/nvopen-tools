// Function: sub_1DCB440
// Address: 0x1dcb440
//
__int64 __fastcall sub_1DCB440(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  unsigned int v6; // ecx
  __int64 v7; // rdx
  __int64 v8; // rsi
  __int64 v9; // rdx
  __int16 v10; // ax

  for ( result = *(unsigned int *)(a3 + 8); (_DWORD)result; result = *(unsigned int *)(a3 + 8) )
  {
    v6 = *(_DWORD *)(*(_QWORD *)a3 + 4LL * (unsigned int)result - 4);
    *(_DWORD *)(a3 + 8) = result - 1;
    v7 = a1[45];
    if ( !v7 )
      BUG();
    v8 = *(_QWORD *)(v7 + 56) + 2LL * *(unsigned int *)(*(_QWORD *)(v7 + 8) + 24LL * v6 + 4);
    while ( 1 )
    {
      v9 = v8;
      if ( !v8 )
        break;
      while ( 1 )
      {
        v9 += 2;
        *(_QWORD *)(a1[46] + 8LL * (unsigned __int16)v6) = a2;
        *(_QWORD *)(a1[49] + 8LL * (unsigned __int16)v6) = 0;
        v10 = *(_WORD *)(v9 - 2);
        v8 = 0;
        if ( !v10 )
          break;
        LOWORD(v6) = v10 + v6;
        if ( !v9 )
          goto LABEL_7;
      }
    }
LABEL_7:
    ;
  }
  return result;
}
