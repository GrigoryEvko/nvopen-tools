// Function: sub_22C3A10
// Address: 0x22c3a10
//
__int64 __fastcall sub_22C3A10(__int64 a1, __int64 *a2, __int64 **a3)
{
  int v4; // edx
  __int64 *v5; // r11
  __int64 v6; // r8
  int v7; // ebx
  __int64 v8; // rdi
  __int64 v9; // rsi
  int v10; // r10d
  unsigned int i; // eax
  __int64 *v12; // rdx
  __int64 v13; // r9
  unsigned int v14; // eax

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    *a3 = 0;
    return 0;
  }
  v5 = 0;
  v6 = *(_QWORD *)(a1 + 8);
  v7 = 1;
  v8 = *a2;
  v9 = a2[1];
  v10 = v4 - 1;
  for ( i = (v4 - 1)
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4)
              | ((unsigned __int64)(((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4)))); ; i = v10 & v14 )
  {
    v12 = (__int64 *)(v6 + 16LL * i);
    v13 = *v12;
    if ( *v12 == v8 && v12[1] == v9 )
    {
      *a3 = v12;
      return 1;
    }
    if ( v13 == -4096 )
      break;
    if ( v13 == -8192 && v12[1] == -8192 && !v5 )
      v5 = (__int64 *)(v6 + 16LL * i);
LABEL_9:
    v14 = v7 + i;
    ++v7;
  }
  if ( v12[1] != -4096 )
    goto LABEL_9;
  if ( !v5 )
    v5 = (__int64 *)(v6 + 16LL * i);
  *a3 = v5;
  return 0;
}
