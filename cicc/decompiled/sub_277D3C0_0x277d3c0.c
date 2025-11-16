// Function: sub_277D3C0
// Address: 0x277d3c0
//
_QWORD *__fastcall sub_277D3C0(__int64 a1, __int64 *a2)
{
  int v2; // edx
  __int64 v3; // r11
  __int64 v4; // rcx
  __int64 v6; // r10
  int v7; // edx
  __int64 v8; // rsi
  __int64 v9; // rdi
  __int64 v10; // r8
  __int64 v11; // r9
  int v12; // r13d
  unsigned int i; // eax
  _QWORD *v14; // r12
  unsigned int v15; // eax

  v2 = *(_DWORD *)(a1 + 24);
  v3 = *(_QWORD *)(a1 + 8);
  if ( !v2 )
    return 0;
  v4 = *a2;
  v6 = a2[1];
  v7 = v2 - 1;
  v8 = a2[2];
  v9 = a2[3];
  v10 = a2[4];
  v11 = a2[5];
  v12 = 1;
  for ( i = v7
          & (((0xBF58476D1CE4E5B9LL * v6) >> 31)
           ^ (484763065 * v6)
           ^ ((unsigned int)v11 >> 9)
           ^ ((unsigned int)v11 >> 4)
           ^ ((unsigned int)v10 >> 9)
           ^ ((unsigned int)v10 >> 4)
           ^ ((unsigned int)v9 >> 9)
           ^ ((unsigned int)v9 >> 4)
           ^ ((unsigned int)v8 >> 9)
           ^ ((unsigned int)v8 >> 4)
           ^ ((unsigned int)v4 >> 9)
           ^ ((unsigned int)v4 >> 4)); ; i = v7 & v15 )
  {
    v14 = (_QWORD *)(v3 + 56LL * i);
    if ( v4 == *v14 && v6 == v14[1] && v8 == v14[2] && v9 == v14[3] && v10 == v14[4] && v11 == v14[5] )
      break;
    if ( *v14 == -4096 && v14[1] == -3 && !v14[2] && !v14[3] && !v14[4] )
    {
      v14 = (_QWORD *)v14[5];
      if ( !v14 )
        break;
    }
    v15 = v12 + i;
    ++v12;
  }
  return v14;
}
