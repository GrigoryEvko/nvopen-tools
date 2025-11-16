// Function: sub_D90990
// Address: 0xd90990
//
_BYTE *__fastcall sub_D90990(__int64 a1, __int64 a2)
{
  int v4; // ecx
  __int64 v5; // rdx
  __int64 v6; // rsi
  _QWORD *v7; // rax
  __int64 v8; // r9
  __int64 v9; // rcx
  _BYTE *v10; // r8
  _BYTE *v11; // rdx

  v4 = *(_DWORD *)(a1 + 4) & 0x7FFFFFF;
  if ( !v4 )
    return 0;
  v5 = *(_QWORD *)(a1 - 8);
  v6 = 32LL * *(unsigned int *)(a1 + 72);
  v7 = (_QWORD *)(v5 + v6);
  v8 = v5 + v6 + 8 + 8LL * (unsigned int)(v4 - 1);
  v9 = -3 * v5;
  v10 = 0;
  do
  {
    if ( a2 != *v7 )
    {
      v11 = *(_BYTE **)(v9 + 4LL * (_QWORD)&v7[v6 / 0xFFFFFFFFFFFFFFF8LL]);
      if ( *v11 > 0x15u )
        return 0;
      if ( v11 != v10 )
      {
        if ( v10 )
          return 0;
        v10 = *(_BYTE **)(v9 + 4LL * (_QWORD)&v7[v6 / 0xFFFFFFFFFFFFFFF8LL]);
      }
    }
    ++v7;
  }
  while ( v7 != (_QWORD *)v8 );
  return v10;
}
