// Function: sub_34A25D0
// Address: 0x34a25d0
//
__int64 __fastcall sub_34A25D0(__int64 a1, __int64 *a2, __int64 **a3)
{
  int v4; // edx
  int v5; // edx
  int v6; // r12d
  __int64 v7; // r8
  __int64 *v8; // rbx
  __int64 v9; // r9
  __int64 v10; // rdi
  __int64 v11; // rsi
  unsigned int i; // eax
  __int64 *v13; // r10
  __int64 v14; // r11
  unsigned int v16; // eax

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    *a3 = 0;
    return 0;
  }
  v5 = v4 - 1;
  v6 = 1;
  v7 = a2[2];
  v8 = 0;
  v9 = *(_QWORD *)(a1 + 8);
  v10 = a2[1];
  v11 = *a2;
  for ( i = v5
          & (((0xBF58476D1CE4E5B9LL
             * ((unsigned int)(unsigned __int16)v7
              | ((_DWORD)v10 << 16)
              | ((unsigned __int64)(((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4)) << 32))) >> 31)
           ^ (484763065 * ((unsigned __int16)v7 | ((_DWORD)v10 << 16)))); ; i = v5 & v16 )
  {
    v13 = (__int64 *)(v9 + 56LL * i);
    v14 = *v13;
    if ( *v13 == v11 && v10 == v13[1] && v7 == v13[2] )
    {
      *a3 = v13;
      return 1;
    }
    if ( v14 == -4096 )
      break;
    if ( v14 == -8192 && v13[1] == -2 && v13[2] == -2 && !v8 )
      v8 = (__int64 *)(v9 + 56LL * i);
LABEL_17:
    v16 = v6 + i;
    ++v6;
  }
  if ( v13[1] != -1 || v13[2] != -1 )
    goto LABEL_17;
  if ( !v8 )
    v8 = (__int64 *)(v9 + 56LL * i);
  *a3 = v8;
  return 0;
}
