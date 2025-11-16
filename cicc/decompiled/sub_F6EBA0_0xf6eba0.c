// Function: sub_F6EBA0
// Address: 0xf6eba0
//
__int64 __fastcall sub_F6EBA0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // r9
  __int64 v6; // rax
  int v7; // edx
  __int64 v8; // rax
  __int64 v9; // rdi
  __int64 v10; // rsi
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v14; // rax
  __int64 v15; // rdx

  v5 = *(_QWORD *)(a1 - 8);
  v6 = 0x1FFFFFFFE0LL;
  v7 = *(_DWORD *)(a1 + 4) & 0x7FFFFFF;
  if ( v7 )
  {
    v8 = 0;
    v9 = v5 + 32LL * *(unsigned int *)(a1 + 72);
    do
    {
      if ( a2 == *(_QWORD *)(v9 + 8 * v8) )
      {
        v6 = 32 * v8;
        goto LABEL_6;
      }
      ++v8;
    }
    while ( v7 != (_DWORD)v8 );
    v6 = 0x1FFFFFFFE0LL;
  }
LABEL_6:
  v10 = *(_QWORD *)(v5 + v6);
  v11 = *(_QWORD *)(a1 + 16);
  if ( v11 )
  {
    while ( 1 )
    {
      v12 = *(_QWORD *)(v11 + 24);
      if ( a3 != v12 && v10 != v12 )
        return 0;
      v11 = *(_QWORD *)(v11 + 8);
      if ( !v11 )
        goto LABEL_12;
    }
  }
  else
  {
LABEL_12:
    v14 = *(_QWORD *)(v10 + 16);
    if ( v14 )
    {
      while ( 1 )
      {
        v15 = *(_QWORD *)(v14 + 24);
        if ( a3 != v15 && a1 != v15 )
          break;
        v14 = *(_QWORD *)(v14 + 8);
        if ( !v14 )
          return 1;
      }
      return 0;
    }
    else
    {
      return 1;
    }
  }
}
