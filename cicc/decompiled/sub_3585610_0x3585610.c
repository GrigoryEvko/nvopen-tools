// Function: sub_3585610
// Address: 0x3585610
//
__int64 __fastcall sub_3585610(__int64 a1, __int64 *a2, __int64 **a3)
{
  int v4; // edx
  int v5; // edx
  __int64 *v6; // r11
  __int64 v7; // r9
  int v8; // ebx
  __int64 v9; // rdi
  __int64 v10; // rsi
  unsigned int i; // eax
  __int64 *v12; // rcx
  __int64 v13; // r10
  unsigned int v14; // eax

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    *a3 = 0;
    return 0;
  }
  v5 = v4 - 1;
  v6 = 0;
  v7 = *(_QWORD *)(a1 + 8);
  v8 = 1;
  v9 = *a2;
  v10 = a2[1];
  for ( i = v5
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4)
              | ((unsigned __int64)(((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4)))); ; i = v5 & v14 )
  {
    v12 = (__int64 *)(v7 + 24LL * i);
    v13 = *v12;
    if ( *v12 == v9 && v12[1] == v10 )
    {
      *a3 = v12;
      return 1;
    }
    if ( v13 == -4096 )
      break;
    if ( v13 == -8192 && v12[1] == -8192 && !v6 )
      v6 = (__int64 *)(v7 + 24LL * i);
LABEL_9:
    v14 = v8 + i;
    ++v8;
  }
  if ( v12[1] != -4096 )
    goto LABEL_9;
  if ( !v6 )
    v6 = (__int64 *)(v7 + 24LL * i);
  *a3 = v6;
  return 0;
}
