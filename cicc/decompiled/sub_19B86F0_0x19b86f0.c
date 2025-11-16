// Function: sub_19B86F0
// Address: 0x19b86f0
//
__int64 __fastcall sub_19B86F0(__int64 a1, __int64 *a2, __int64 **a3)
{
  __int64 v3; // r8
  int v4; // edi
  __int64 *v5; // r11
  __int64 v6; // rcx
  int v7; // ebx
  __int64 v8; // rsi
  unsigned int v9; // r10d
  unsigned __int64 v10; // r9
  unsigned __int64 v11; // r9
  unsigned int i; // eax
  __int64 *v13; // r9
  __int64 v14; // r10
  unsigned int v15; // eax
  __int64 result; // rax

  if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
  {
    v3 = a1 + 16;
    v4 = 3;
  }
  else
  {
    result = *(unsigned int *)(a1 + 24);
    v3 = *(_QWORD *)(a1 + 16);
    v4 = result - 1;
    if ( !(_DWORD)result )
    {
      *a3 = 0;
      return result;
    }
  }
  v5 = 0;
  v6 = *a2;
  v7 = 1;
  v8 = a2[1];
  v9 = (unsigned int)v8 >> 9;
  v10 = (((v9 ^ ((unsigned int)v8 >> 4) | ((unsigned __int64)(((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4)) << 32))
        - 1
        - ((unsigned __int64)(v9 ^ ((unsigned int)v8 >> 4)) << 32)) >> 22)
      ^ ((v9 ^ ((unsigned int)v8 >> 4) | ((unsigned __int64)(((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4)) << 32))
       - 1
       - ((unsigned __int64)(v9 ^ ((unsigned int)v8 >> 4)) << 32));
  v11 = ((9 * (((v10 - 1 - (v10 << 13)) >> 8) ^ (v10 - 1 - (v10 << 13)))) >> 15)
      ^ (9 * (((v10 - 1 - (v10 << 13)) >> 8) ^ (v10 - 1 - (v10 << 13))));
  for ( i = v4 & (((v11 - 1 - (v11 << 27)) >> 31) ^ (v11 - 1 - ((_DWORD)v11 << 27))); ; i = v4 & v15 )
  {
    v13 = (__int64 *)(v3 + 16LL * i);
    v14 = *v13;
    if ( *v13 == v6 && v13[1] == v8 )
    {
      *a3 = v13;
      return 1;
    }
    if ( v14 == -8 )
      break;
    if ( v14 == -16 && v13[1] == -16 && !v5 )
      v5 = (__int64 *)(v3 + 16LL * i);
LABEL_10:
    v15 = v7 + i;
    ++v7;
  }
  if ( v13[1] != -8 )
    goto LABEL_10;
  if ( !v5 )
    v5 = (__int64 *)(v3 + 16LL * i);
  *a3 = v5;
  return 0;
}
