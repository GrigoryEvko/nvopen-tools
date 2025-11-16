// Function: sub_1280BE0
// Address: 0x1280be0
//
__int64 __fastcall sub_1280BE0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v5; // rsi
  __int64 i; // rax
  __int64 v7; // rbx
  __int64 v8; // r13
  _BOOL4 v9; // edx
  int v10; // eax
  __int64 v12; // rax

  v5 = *(__int64 **)(a3 + 72);
  for ( i = *v5; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v7 = *(_QWORD *)(i + 160);
  v8 = sub_128F980(a2, v5);
  if ( (*(_BYTE *)(v7 + 140) & 0xFB) == 8 )
  {
    v9 = (sub_8D4C10(v7, dword_4F077C4 != 2) & 2) != 0;
    if ( *(_BYTE *)(v7 + 140) != 12 )
      goto LABEL_6;
    v12 = v7;
    do
      v12 = *(_QWORD *)(v12 + 160);
    while ( *(_BYTE *)(v12 + 140) == 12 );
    if ( *(char *)(v12 + 142) < 0 )
    {
      do
      {
        v7 = *(_QWORD *)(v7 + 160);
        if ( *(_BYTE *)(v7 + 140) != 12 )
          break;
        v7 = *(_QWORD *)(v7 + 160);
      }
      while ( *(_BYTE *)(v7 + 140) == 12 );
      goto LABEL_6;
    }
    do
LABEL_11:
      v7 = *(_QWORD *)(v7 + 160);
    while ( *(_BYTE *)(v7 + 140) == 12 );
    goto LABEL_6;
  }
  if ( *(char *)(v7 + 142) < 0 )
  {
    v9 = 0;
    goto LABEL_6;
  }
  v9 = 0;
  if ( *(_BYTE *)(v7 + 140) == 12 )
    goto LABEL_11;
LABEL_6:
  v10 = *(_DWORD *)(v7 + 136);
  *(_QWORD *)(a1 + 8) = v8;
  *(_DWORD *)a1 = 0;
  *(_DWORD *)(a1 + 16) = v10;
  *(_DWORD *)(a1 + 40) = v9;
  return a1;
}
