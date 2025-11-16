// Function: sub_25134D0
// Address: 0x25134d0
//
_QWORD *__fastcall sub_25134D0(__int64 a1, __int64 *a2)
{
  __int64 v3; // rdi
  int v4; // r10d
  int v6; // r12d
  __int64 v7; // rcx
  __int64 v8; // rsi
  __int64 v9; // rdx
  int v10; // r11d
  unsigned int i; // eax
  _QWORD *v12; // r10
  unsigned int v13; // eax

  v3 = *(_QWORD *)(a1 + 8);
  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
    return 0;
  v6 = 1;
  v7 = a2[1];
  v8 = a2[2];
  v9 = *a2;
  v10 = v4 - 1;
  for ( i = (v4 - 1)
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned __int64)(((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4)) << 32)
              | ((unsigned int)v8 >> 9)
              ^ ((unsigned int)v8 >> 4)
              ^ (16 * (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4))))) >> 31)
           ^ (484763065
            * (((unsigned int)v8 >> 9)
             ^ ((unsigned int)v8 >> 4)
             ^ (16 * (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4)))))); ; i = v10 & v13 )
  {
    v12 = (_QWORD *)(v3 + 32LL * i);
    if ( *v12 == v9 && v7 == v12[1] && v8 == v12[2] )
      break;
    if ( *v12 == -4096 && unk_4FEE4D0 == v12[1] && unk_4FEE4D8 == v12[2] )
      return 0;
    v13 = v6 + i;
    ++v6;
  }
  return v12;
}
