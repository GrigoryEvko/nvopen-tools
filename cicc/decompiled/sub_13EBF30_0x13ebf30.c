// Function: sub_13EBF30
// Address: 0x13ebf30
//
__int64 __fastcall sub_13EBF30(__int64 a1, __int64 *a2, __int64 **a3)
{
  int v4; // edx
  int v6; // edx
  __int64 *v7; // r11
  __int64 v8; // rsi
  int v9; // ebx
  __int64 v10; // r8
  __int64 v11; // rdi
  unsigned int v12; // r10d
  unsigned __int64 v13; // r9
  unsigned __int64 v14; // r9
  unsigned int i; // eax
  __int64 *v16; // r9
  __int64 v17; // r10
  unsigned int v18; // eax

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    *a3 = 0;
    return 0;
  }
  v6 = v4 - 1;
  v7 = 0;
  v8 = *a2;
  v9 = 1;
  v10 = *(_QWORD *)(a1 + 8);
  v11 = a2[1];
  v12 = (unsigned int)v11 >> 9;
  v13 = (((v12 ^ ((unsigned int)v11 >> 4) | ((unsigned __int64)(((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4)) << 32))
        - 1
        - ((unsigned __int64)(v12 ^ ((unsigned int)v11 >> 4)) << 32)) >> 22)
      ^ ((v12 ^ ((unsigned int)v11 >> 4) | ((unsigned __int64)(((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4)) << 32))
       - 1
       - ((unsigned __int64)(v12 ^ ((unsigned int)v11 >> 4)) << 32));
  v14 = ((9 * (((v13 - 1 - (v13 << 13)) >> 8) ^ (v13 - 1 - (v13 << 13)))) >> 15)
      ^ (9 * (((v13 - 1 - (v13 << 13)) >> 8) ^ (v13 - 1 - (v13 << 13))));
  for ( i = v6 & (((v14 - 1 - (v14 << 27)) >> 31) ^ (v14 - 1 - ((_DWORD)v14 << 27))); ; i = v6 & v18 )
  {
    v16 = (__int64 *)(v10 + 16LL * i);
    v17 = *v16;
    if ( *v16 == v8 && v16[1] == v11 )
    {
      *a3 = v16;
      return 1;
    }
    if ( v17 == -8 )
      break;
    if ( v17 == -16 && v16[1] == -16 && !v7 )
      v7 = (__int64 *)(v10 + 16LL * i);
LABEL_9:
    v18 = v9 + i;
    ++v9;
  }
  if ( v16[1] != -8 )
    goto LABEL_9;
  if ( !v7 )
    v7 = (__int64 *)(v10 + 16LL * i);
  *a3 = v7;
  return 0;
}
