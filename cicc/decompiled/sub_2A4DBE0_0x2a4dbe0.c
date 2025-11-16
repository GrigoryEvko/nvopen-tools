// Function: sub_2A4DBE0
// Address: 0x2a4dbe0
//
__int64 __fastcall sub_2A4DBE0(__int64 a1, int *a2, int **a3)
{
  int v4; // edx
  int v5; // r10d
  int *v6; // r11
  __int64 v7; // r8
  int v8; // ebx
  int v9; // edi
  int v10; // esi
  unsigned int i; // eax
  int *v12; // rdx
  int v13; // r9d
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
  v9 = a2[1];
  v10 = *a2;
  for ( i = (v4 - 1)
          & (((0xBF58476D1CE4E5B9LL * ((unsigned int)(37 * v9) | ((unsigned __int64)(unsigned int)(37 * v10) << 32))) >> 31)
           ^ (756364221 * v9)); ; i = v5 & v14 )
  {
    v12 = (int *)(v7 + 16LL * i);
    v13 = *v12;
    if ( v10 == *v12 && v9 == v12[1] )
    {
      *a3 = v12;
      return 1;
    }
    if ( v13 == -1 )
      break;
    if ( v13 == -2 && v12[1] == -2 && !v6 )
      v6 = (int *)(v7 + 16LL * i);
LABEL_9:
    v14 = v8 + i;
    ++v8;
  }
  if ( v12[1] != -1 )
    goto LABEL_9;
  if ( !v6 )
    v6 = (int *)(v7 + 16LL * i);
  *a3 = v6;
  return 0;
}
