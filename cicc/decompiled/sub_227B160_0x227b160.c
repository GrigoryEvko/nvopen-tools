// Function: sub_227B160
// Address: 0x227b160
//
__int64 __fastcall sub_227B160(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rdx
  __int64 v5; // r8
  int v6; // r11d
  unsigned int i; // eax
  _QWORD *v8; // rcx
  unsigned int v9; // eax

  v4 = *(unsigned int *)(a1 + 88);
  v5 = *(_QWORD *)(a1 + 72);
  if ( !(_DWORD)v4 )
    return 0;
  v6 = 1;
  for ( i = (v4 - 1)
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
              | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; i = (v4 - 1) & v9 )
  {
    v8 = (_QWORD *)(v5 + 24LL * i);
    if ( a2 == *v8 && a3 == v8[1] )
      break;
    if ( *v8 == -4096 && v8[1] == -4096 )
      return 0;
    v9 = v6 + i;
    ++v6;
  }
  if ( v8 == (_QWORD *)(v5 + 24 * v4) )
    return 0;
  else
    return *(_QWORD *)(v8[2] + 24LL);
}
