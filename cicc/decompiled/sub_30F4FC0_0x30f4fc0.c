// Function: sub_30F4FC0
// Address: 0x30f4fc0
//
__int16 __fastcall sub_30F4FC0(__int64 a1, __int64 a2, unsigned int a3, __int64 a4)
{
  __int64 v4; // r13
  __int16 result; // ax
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // rsi
  __int64 v9; // rcx
  __int64 v10; // rax
  _QWORD *v11; // rax
  __int64 v12; // rax
  __int64 *v13; // rdx
  unsigned int v14; // eax
  __int64 v15; // rcx

  v4 = a3;
  if ( *(_QWORD *)(a1 + 16) != *(_QWORD *)(a2 + 16) && !sub_30F4F20(a1, a2, a4) )
    return 256;
  v6 = *(unsigned int *)(a1 + 32);
  v7 = *(unsigned int *)(a2 + 32);
  if ( (_DWORD)v6 != (_DWORD)v7 )
    return 256;
  v8 = *(_QWORD *)(a2 + 24);
  v9 = *(_QWORD *)(a1 + 24);
  if ( (_DWORD)v6 != 1 )
  {
    v10 = 0;
    while ( *(_QWORD *)(v8 + 8LL * (unsigned int)v10) == *(_QWORD *)(v9 + 8LL * (unsigned int)v10) )
    {
      if ( (_DWORD)v6 - 1 == ++v10 )
        goto LABEL_10;
    }
    return 256;
  }
LABEL_10:
  v11 = sub_DCC810(*(__int64 **)(a1 + 104), *(_QWORD *)(v9 + 8 * v6 - 8), *(_QWORD *)(v8 + 8 * v7 - 8), 0, 0);
  if ( *((_WORD *)v11 + 12) )
    return 0;
  v12 = v11[4];
  v13 = *(__int64 **)(v12 + 24);
  v14 = *(_DWORD *)(v12 + 32);
  if ( v14 > 0x40 )
  {
    v15 = *v13;
  }
  else
  {
    v15 = 0;
    if ( v14 )
      v15 = (__int64)((_QWORD)v13 << (64 - (unsigned __int8)v14)) >> (64 - (unsigned __int8)v14);
  }
  LOBYTE(result) = v4 > v15;
  HIBYTE(result) = 1;
  return result;
}
