// Function: sub_922DA0
// Address: 0x922da0
//
__int64 __fastcall sub_922DA0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v5; // rsi
  __int64 i; // rax
  __int64 v7; // rbx
  __int64 v8; // r13
  int v9; // ecx
  __int64 v10; // rax
  int v11; // edx
  char v13; // cl
  __int64 v14; // rdx

  v5 = *(__int64 **)(a3 + 72);
  for ( i = *v5; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v7 = *(_QWORD *)(i + 160);
  v8 = sub_92F410(a2, v5);
  if ( (*(_BYTE *)(v7 + 140) & 0xFB) == 8 )
  {
    v13 = (unsigned int)sub_8D4C10(v7, dword_4F077C4 != 2) >> 1;
    v10 = v7;
    v9 = v13 & 1;
    if ( *(_BYTE *)(v7 + 140) != 12 )
      goto LABEL_6;
    do
      v10 = *(_QWORD *)(v10 + 160);
    while ( *(_BYTE *)(v10 + 140) == 12 );
    v14 = v7;
    do
      v14 = *(_QWORD *)(v14 + 160);
    while ( *(_BYTE *)(v14 + 140) == 12 );
    if ( *(char *)(v14 + 142) < 0 )
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
LABEL_12:
      v7 = *(_QWORD *)(v7 + 160);
    while ( *(_BYTE *)(v7 + 140) == 12 );
    goto LABEL_6;
  }
  if ( *(char *)(v7 + 142) < 0 )
  {
    v10 = v7;
    v9 = 0;
    goto LABEL_6;
  }
  v9 = 0;
  v10 = v7;
  if ( *(_BYTE *)(v7 + 140) == 12 )
    goto LABEL_12;
LABEL_6:
  v11 = *(_DWORD *)(v7 + 136);
  *(_QWORD *)(a1 + 8) = v8;
  *(_QWORD *)(a1 + 16) = v10;
  *(_DWORD *)a1 = 0;
  *(_DWORD *)(a1 + 48) = v9;
  *(_DWORD *)(a1 + 24) = v11;
  return a1;
}
