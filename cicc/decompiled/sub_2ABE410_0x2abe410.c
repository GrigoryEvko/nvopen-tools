// Function: sub_2ABE410
// Address: 0x2abe410
//
__int64 __fastcall sub_2ABE410(__int64 a1, __int64 *a2, _QWORD *a3)
{
  int v4; // edx
  int v5; // r8d
  int v6; // ebx
  char v7; // r9
  __int64 v8; // rsi
  __int64 v9; // rdi
  int v10; // edx
  __int64 v11; // r12
  unsigned int i; // eax
  __int64 *v13; // r10
  __int64 v14; // r11
  unsigned int v15; // eax

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    *a3 = 0;
    return 0;
  }
  v5 = *((_DWORD *)a2 + 2);
  v6 = 1;
  v7 = *((_BYTE *)a2 + 12);
  v8 = *a2;
  v9 = *(_QWORD *)(a1 + 8);
  v10 = v4 - 1;
  v11 = 0;
  for ( i = v10
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)(v7 == 0) + 37 * v5 - 1)
              | ((unsigned __int64)(((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4)) << 32))) >> 31)
           ^ (484763065 * ((v7 == 0) + 37 * v5 - 1))); ; i = v10 & v15 )
  {
    v13 = (__int64 *)(v9 + 40LL * i);
    v14 = *v13;
    if ( *v13 == v8 && v5 == *((_DWORD *)v13 + 2) && v7 == *((_BYTE *)v13 + 12) )
    {
      *a3 = v13;
      return 1;
    }
    if ( v14 == -4096 )
      break;
    if ( v14 == -8192 && *((_DWORD *)v13 + 2) == -2 && *((_BYTE *)v13 + 12) != 1 && !v11 )
      v11 = v9 + 40LL * i;
LABEL_7:
    v15 = v6 + i;
    ++v6;
  }
  if ( *((_DWORD *)v13 + 2) != -1 || !*((_BYTE *)v13 + 12) )
    goto LABEL_7;
  if ( !v11 )
    v11 = v9 + 40LL * i;
  *a3 = v11;
  return 0;
}
