// Function: sub_1D66860
// Address: 0x1d66860
//
__int64 __fastcall sub_1D66860(__int64 a1, int *a2, int **a3)
{
  int v4; // edx
  int v6; // edx
  int *v7; // r11
  int v8; // esi
  int v9; // ebx
  __int64 v10; // r8
  int v11; // edi
  unsigned __int64 v12; // r9
  unsigned __int64 v13; // r9
  unsigned int i; // eax
  int *v15; // r9
  int v16; // r10d
  unsigned int v17; // eax

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
  v12 = ((((unsigned int)(37 * v11) | ((unsigned __int64)(unsigned int)(37 * v8) << 32))
        - 1
        - ((unsigned __int64)(unsigned int)(37 * v11) << 32)) >> 22)
      ^ (((unsigned int)(37 * v11) | ((unsigned __int64)(unsigned int)(37 * v8) << 32))
       - 1
       - ((unsigned __int64)(unsigned int)(37 * v11) << 32));
  v13 = ((9 * (((v12 - 1 - (v12 << 13)) >> 8) ^ (v12 - 1 - (v12 << 13)))) >> 15)
      ^ (9 * (((v12 - 1 - (v12 << 13)) >> 8) ^ (v12 - 1 - (v12 << 13))));
  for ( i = v6 & (((v13 - 1 - (v13 << 27)) >> 31) ^ (v13 - 1 - ((_DWORD)v13 << 27))); ; i = v6 & v17 )
  {
    v15 = (int *)(v10 + 16LL * i);
    v16 = *v15;
    if ( *v15 == v8 && v15[1] == v11 )
    {
      *a3 = v15;
      return 1;
    }
    if ( v16 == -1 )
      break;
    if ( v16 == -2 && v15[1] == -2 && !v7 )
      v7 = (int *)(v10 + 16LL * i);
LABEL_9:
    v17 = v9 + i;
    ++v9;
  }
  if ( v15[1] != -1 )
    goto LABEL_9;
  if ( !v7 )
    v7 = (int *)(v10 + 16LL * i);
  *a3 = v7;
  return 0;
}
