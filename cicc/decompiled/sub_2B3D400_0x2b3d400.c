// Function: sub_2B3D400
// Address: 0x2b3d400
//
__int64 __fastcall sub_2B3D400(__int64 a1, int *a2, int **a3)
{
  __int64 v4; // r9
  int v5; // edi
  int *v6; // r11
  int v7; // edx
  int v8; // ebx
  int v9; // ecx
  unsigned int i; // eax
  int *v11; // rsi
  int v12; // r10d
  unsigned int v13; // eax
  __int64 result; // rax

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
  v6 = 0;
  v7 = *a2;
  v8 = 1;
  v9 = a2[1];
  for ( i = v5
          & (((0xBF58476D1CE4E5B9LL * ((unsigned int)(37 * v9) | ((unsigned __int64)(unsigned int)(37 * *a2) << 32))) >> 31)
           ^ (756364221 * v9)); ; i = v5 & v13 )
  {
    v11 = (int *)(v4 + 12LL * i);
    v12 = *v11;
    if ( v7 == *v11 && v9 == v11[1] )
    {
      *a3 = v11;
      return 1;
    }
    if ( v12 == -1 )
      break;
    if ( v12 == -2 && v11[1] == -2 && !v6 )
      v6 = (int *)(v4 + 12LL * i);
LABEL_10:
    v13 = v8 + i;
    ++v8;
  }
  if ( v11[1] != -1 )
    goto LABEL_10;
  if ( !v6 )
    v6 = (int *)(v4 + 12LL * i);
  *a3 = v6;
  return 0;
}
