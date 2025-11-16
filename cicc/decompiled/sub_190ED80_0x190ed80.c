// Function: sub_190ED80
// Address: 0x190ed80
//
__int64 __fastcall sub_190ED80(__int64 a1, int *a2, _QWORD *a3)
{
  int v4; // edx
  int v5; // edx
  __int64 v6; // r11
  __int64 v7; // r8
  int v8; // ebx
  __int64 v9; // rdi
  int v10; // esi
  unsigned __int64 v11; // r9
  unsigned __int64 v12; // r9
  unsigned int i; // eax
  int *v14; // r9
  int v15; // r10d
  unsigned int v16; // eax

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
  v9 = *((_QWORD *)a2 + 1);
  v10 = *a2;
  v11 = (((((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4) | ((unsigned __int64)(unsigned int)(37 * v10) << 32))
        - 1
        - ((unsigned __int64)(((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4)) << 32)) >> 22)
      ^ ((((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4) | ((unsigned __int64)(unsigned int)(37 * v10) << 32))
       - 1
       - ((unsigned __int64)(((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4)) << 32));
  v12 = ((9 * (((v11 - 1 - (v11 << 13)) >> 8) ^ (v11 - 1 - (v11 << 13)))) >> 15)
      ^ (9 * (((v11 - 1 - (v11 << 13)) >> 8) ^ (v11 - 1 - (v11 << 13))));
  for ( i = v5 & (((v12 - 1 - (v12 << 27)) >> 31) ^ (v12 - 1 - ((_DWORD)v12 << 27))); ; i = v5 & v16 )
  {
    v14 = (int *)(v7 + 24LL * i);
    v15 = *v14;
    if ( *v14 == v10 && *((_QWORD *)v14 + 1) == v9 )
    {
      *a3 = v14;
      return 1;
    }
    if ( v15 == -1 )
      break;
    if ( v15 == -2 && *((_QWORD *)v14 + 1) == -16 && !v6 )
      v6 = v7 + 24LL * i;
LABEL_9:
    v16 = v8 + i;
    ++v8;
  }
  if ( *((_QWORD *)v14 + 1) != -8 )
    goto LABEL_9;
  if ( !v6 )
    v6 = v7 + 24LL * i;
  *a3 = v6;
  return 0;
}
