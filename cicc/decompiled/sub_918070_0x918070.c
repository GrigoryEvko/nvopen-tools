// Function: sub_918070
// Address: 0x918070
//
__int64 __fastcall sub_918070(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v5; // edi
  __int64 i; // rbx
  unsigned __int64 v7; // rdx
  __int64 v8; // r8
  __int64 *v9; // rax
  __int64 v10; // r10
  _QWORD *v11; // rax
  _QWORD *v12; // rdx
  __int64 v14; // rbx
  __int64 v15; // rsi
  char j; // al
  int v17; // eax
  int v18; // r11d

  v5 = *(_DWORD *)(a1 + 172);
  if ( *(_DWORD *)(a1 + 176) == v5 )
    return 1;
  for ( i = a2; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v7 = *(unsigned int *)(a1 + 48);
  v8 = *(_QWORD *)(a1 + 32);
  if ( (_DWORD)v7 )
  {
    a4 = ((_DWORD)v7 - 1) & (((unsigned int)i >> 9) ^ ((unsigned int)i >> 4));
    v9 = (__int64 *)(v8 + 16 * a4);
    v10 = *v9;
    if ( *v9 == i )
    {
LABEL_6:
      v7 = v8 + 16 * v7;
      if ( v9 != (__int64 *)v7 && (*(_BYTE *)(v9[1] + 9) & 1) != 0 )
        return 1;
    }
    else
    {
      v17 = 1;
      while ( v10 != -4096 )
      {
        v18 = v17 + 1;
        a4 = ((_DWORD)v7 - 1) & (unsigned int)(v17 + a4);
        v9 = (__int64 *)(v8 + 16LL * (unsigned int)a4);
        v10 = *v9;
        if ( i == *v9 )
          goto LABEL_6;
        v17 = v18;
      }
    }
  }
  if ( !*(_BYTE *)(a1 + 180) )
  {
    if ( !sub_C8CA60(a1 + 152, i, v7, a4, v8) )
      goto LABEL_18;
    return 0;
  }
  v11 = *(_QWORD **)(a1 + 160);
  v12 = &v11[v5];
  if ( v11 != v12 )
  {
    while ( i != *v11 )
    {
      if ( v12 == ++v11 )
        goto LABEL_18;
    }
    return 0;
  }
LABEL_18:
  v14 = *(_QWORD *)(i + 160);
  if ( !v14 )
    return 1;
  while ( 1 )
  {
    v15 = *(_QWORD *)(v14 + 120);
    for ( j = *(_BYTE *)(v15 + 140); j == 12; j = *(_BYTE *)(v15 + 140) )
      v15 = *(_QWORD *)(v15 + 160);
    while ( j == 8 )
    {
      do
      {
        v15 = *(_QWORD *)(v15 + 160);
        j = *(_BYTE *)(v15 + 140);
      }
      while ( j == 12 );
    }
    if ( (unsigned __int8)(j - 10) <= 1u && !(unsigned __int8)sub_918070(a1) )
      break;
    v14 = *(_QWORD *)(v14 + 112);
    if ( !v14 )
      return 1;
  }
  return 0;
}
