// Function: sub_2D6B8A0
// Address: 0x2d6b8a0
//
__int64 __fastcall sub_2D6B8A0(__int64 a1, int *a2, int **a3)
{
  int v4; // edx
  int v5; // edx
  int *v6; // r11
  int v7; // ecx
  int v8; // ebx
  __int64 v9; // r9
  int v10; // edi
  unsigned int i; // eax
  int *v12; // rsi
  int v13; // r10d
  unsigned int v14; // eax

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    *a3 = 0;
    return 0;
  }
  v5 = v4 - 1;
  v6 = 0;
  v7 = *a2;
  v8 = 1;
  v9 = *(_QWORD *)(a1 + 8);
  v10 = a2[1];
  for ( i = v5
          & (((0xBF58476D1CE4E5B9LL * ((unsigned int)(37 * v10) | ((unsigned __int64)(unsigned int)(37 * *a2) << 32))) >> 31)
           ^ (756364221 * v10)); ; i = v5 & v14 )
  {
    v12 = (int *)(v9 + 12LL * i);
    v13 = *v12;
    if ( v7 == *v12 && v10 == v12[1] )
    {
      *a3 = v12;
      return 1;
    }
    if ( v13 == -1 )
      break;
    if ( v13 == -2 && v12[1] == -2 && !v6 )
      v6 = (int *)(v9 + 12LL * i);
LABEL_9:
    v14 = v8 + i;
    ++v8;
  }
  if ( v12[1] != -1 )
    goto LABEL_9;
  if ( !v6 )
    v6 = (int *)(v9 + 12LL * i);
  *a3 = v6;
  return 0;
}
