// Function: sub_2ABE520
// Address: 0x2abe520
//
__int64 __fastcall sub_2ABE520(__int64 a1, __int64 *a2, _QWORD *a3)
{
  int v4; // edx
  int v5; // r8d
  int v6; // ebx
  char v7; // r9
  __int64 v8; // rsi
  __int64 v9; // rdi
  __int64 v10; // r12
  int v11; // r11d
  unsigned int i; // eax
  __int64 *v13; // rdx
  __int64 v14; // r10
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
  v10 = 0;
  v11 = v4 - 1;
  for ( i = (v4 - 1)
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)(v7 == 0) + 37 * v5 - 1)
              | ((unsigned __int64)(((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4)) << 32))) >> 31)
           ^ (484763065 * ((v7 == 0) + 37 * v5 - 1))); ; i = v11 & v15 )
  {
    v13 = (__int64 *)(v9 + ((unsigned __int64)i << 6));
    v14 = *v13;
    if ( *v13 == v8 && v5 == *((_DWORD *)v13 + 2) && v7 == *((_BYTE *)v13 + 12) )
    {
      *a3 = v13;
      return 1;
    }
    if ( v14 == -4096 )
      break;
    if ( v14 == -8192 && *((_DWORD *)v13 + 2) == -2 && *((_BYTE *)v13 + 12) != 1 && !v10 )
      v10 = v9 + ((unsigned __int64)i << 6);
LABEL_7:
    v15 = v6 + i;
    ++v6;
  }
  if ( *((_DWORD *)v13 + 2) != -1 || !*((_BYTE *)v13 + 12) )
    goto LABEL_7;
  if ( !v10 )
    v10 = v9 + ((unsigned __int64)i << 6);
  *a3 = v10;
  return 0;
}
