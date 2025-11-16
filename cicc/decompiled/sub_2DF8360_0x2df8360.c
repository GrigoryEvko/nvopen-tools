// Function: sub_2DF8360
// Address: 0x2df8360
//
__int64 __fastcall sub_2DF8360(__int64 a1, unsigned __int64 a2, char a3)
{
  unsigned __int64 v3; // rcx
  unsigned __int64 i; // rax
  __int64 j; // rsi
  __int16 v6; // dx
  __int64 v7; // rcx
  __int64 v8; // r8
  unsigned int v9; // edx
  __int64 *v10; // rax
  __int64 v11; // rdi
  int v13; // eax
  int v14; // r10d

  v3 = a2;
  for ( i = a2; (*(_BYTE *)(i + 44) & 4) != 0; i = *(_QWORD *)i & 0xFFFFFFFFFFFFFFF8LL )
    ;
  if ( (*(_DWORD *)(a2 + 44) & 8) != 0 )
  {
    do
      v3 = *(_QWORD *)(v3 + 8);
    while ( (*(_BYTE *)(v3 + 44) & 8) != 0 );
  }
  if ( !a3 )
  {
    for ( j = *(_QWORD *)(v3 + 8); j != i; i = *(_QWORD *)(i + 8) )
    {
      v6 = *(_WORD *)(i + 68);
      if ( (unsigned __int16)(v6 - 14) > 4u && v6 != 24 )
        break;
    }
    a2 = i;
  }
  v7 = *(unsigned int *)(a1 + 144);
  v8 = *(_QWORD *)(a1 + 128);
  if ( (_DWORD)v7 )
  {
    v9 = (v7 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v10 = (__int64 *)(v8 + 16LL * v9);
    v11 = *v10;
    if ( a2 == *v10 )
      return v10[1];
    v13 = 1;
    while ( v11 != -4096 )
    {
      v14 = v13 + 1;
      v9 = (v7 - 1) & (v13 + v9);
      v10 = (__int64 *)(v8 + 16LL * v9);
      v11 = *v10;
      if ( a2 == *v10 )
        return v10[1];
      v13 = v14;
    }
  }
  return *(_QWORD *)(v8 + 16 * v7 + 8);
}
