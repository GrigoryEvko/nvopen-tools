// Function: sub_3139BD0
// Address: 0x3139bd0
//
__int64 __fastcall sub_3139BD0(__int64 a1, __int64 *a2, __int64 **a3)
{
  int v4; // edx
  int v5; // edx
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // rsi
  int v9; // ebx
  __int64 *v10; // r11
  unsigned int i; // eax
  __int64 *v12; // rdi
  __int64 v13; // r10
  unsigned int v14; // eax

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    *a3 = 0;
    return 0;
  }
  v5 = v4 - 1;
  v6 = a2[1];
  v7 = *(_QWORD *)(a1 + 8);
  v8 = *a2;
  v9 = 1;
  v10 = 0;
  for ( i = v5
          & (((0xBF58476D1CE4E5B9LL
             * ((unsigned int)((0xBF58476D1CE4E5B9LL * v6) >> 31) ^ (484763065 * (_DWORD)v6)
              | ((unsigned __int64)(((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((0xBF58476D1CE4E5B9LL * v6) >> 31) ^ (484763065 * v6)))); ; i = v5 & v14 )
  {
    v12 = (__int64 *)(v7 + 24LL * i);
    v13 = *v12;
    if ( *v12 == v8 && v12[1] == v6 )
    {
      *a3 = v12;
      return 1;
    }
    if ( v13 == -4096 )
      break;
    if ( v13 == -8192 && v12[1] == -2 && !v10 )
      v10 = (__int64 *)(v7 + 24LL * i);
LABEL_9:
    v14 = v9 + i;
    ++v9;
  }
  if ( v12[1] != -1 )
    goto LABEL_9;
  if ( !v10 )
    v10 = (__int64 *)(v7 + 24LL * i);
  *a3 = v10;
  return 0;
}
