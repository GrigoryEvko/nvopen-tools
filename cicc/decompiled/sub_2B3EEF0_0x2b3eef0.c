// Function: sub_2B3EEF0
// Address: 0x2b3eef0
//
__int64 __fastcall sub_2B3EEF0(__int64 a1, __int64 *a2, _QWORD *a3)
{
  __int64 v4; // r9
  int v5; // r8d
  __int64 v6; // rdi
  __int64 v7; // rsi
  __int64 v8; // rcx
  __int64 v9; // r12
  int v10; // ebx
  unsigned int i; // eax
  _QWORD *v12; // r10
  __int64 v13; // r11
  __int64 result; // rax
  unsigned int v15; // eax

  if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
  {
    v4 = a1 + 16;
    v5 = 7;
  }
  else
  {
    result = *(unsigned int *)(a1 + 24);
    v4 = *(_QWORD *)(a1 + 16);
    v5 = result - 1;
    if ( !(_DWORD)result )
    {
      *a3 = 0;
      return result;
    }
  }
  v6 = *a2;
  v7 = a2[1];
  v8 = a2[2];
  v9 = 0;
  v10 = 1;
  for ( i = v5
          & (((0xBF58476D1CE4E5B9LL
             * ((unsigned int)((0xBF58476D1CE4E5B9LL
                              * ((969526130 * (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4)))
                               | ((unsigned __int64)(((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4)) << 32))) >> 31)
              ^ (-279380126 * (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4)))
              | ((unsigned __int64)(((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4)) << 32))) >> 31)
           ^ (484763065
            * (((0xBF58476D1CE4E5B9LL
               * ((969526130 * (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4)))
                | ((unsigned __int64)(((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4)) << 32))) >> 31)
             ^ (-279380126 * (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4)))))); ; i = v5 & v15 )
  {
    v12 = (_QWORD *)(v4 + 88LL * i);
    v13 = v12[2];
    if ( v13 == v8 && v7 == v12[1] && v6 == *v12 )
    {
      *a3 = v12;
      return 1;
    }
    if ( v13 == -4096 )
      break;
    if ( v13 == -8192 && v12[1] == -8192 && *v12 == -8192 && !v9 )
      v9 = v4 + 88LL * i;
LABEL_19:
    v15 = v10 + i;
    ++v10;
  }
  if ( v12[1] != -4096 || *v12 != -4096 )
    goto LABEL_19;
  if ( !v9 )
    v9 = v4 + 88LL * i;
  *a3 = v9;
  return 0;
}
