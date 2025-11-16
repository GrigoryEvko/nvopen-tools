// Function: sub_BD5C70
// Address: 0xbd5c70
//
__int64 __fastcall sub_BD5C70(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // rcx
  __int64 v4; // rsi
  unsigned int v5; // edx
  __int64 *v6; // rax
  __int64 v7; // rdi
  int v9; // eax
  int v10; // r9d

  if ( (*(_BYTE *)(a1 + 7) & 0x10) == 0 )
    return 0;
  v2 = *(_QWORD *)sub_BD5C60(a1);
  v3 = *(unsigned int *)(v2 + 200);
  v4 = *(_QWORD *)(v2 + 184);
  if ( (_DWORD)v3 )
  {
    v5 = (v3 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
    v6 = (__int64 *)(v4 + 16LL * v5);
    v7 = *v6;
    if ( a1 == *v6 )
      return v6[1];
    v9 = 1;
    while ( v7 != -4096 )
    {
      v10 = v9 + 1;
      v5 = (v3 - 1) & (v9 + v5);
      v6 = (__int64 *)(v4 + 16LL * v5);
      v7 = *v6;
      if ( a1 == *v6 )
        return v6[1];
      v9 = v10;
    }
  }
  return *(_QWORD *)(v4 + 16 * v3 + 8);
}
