// Function: sub_256E050
// Address: 0x256e050
//
__int64 __fastcall sub_256E050(__int64 a1, __int64 *a2, _QWORD *a3)
{
  int v3; // r10d
  int v6; // r10d
  __int64 v7; // r12
  unsigned __int8 v8; // r8
  __int64 v9; // rsi
  __int64 v10; // r9
  __int64 v11; // rdi
  int v12; // ebx
  unsigned __int64 v13; // rax
  unsigned int i; // eax
  __int64 *v15; // rdx
  __int64 v16; // r11
  unsigned int v17; // eax

  v3 = *(_DWORD *)(a1 + 24);
  if ( !v3 )
  {
    *a3 = 0;
    return 0;
  }
  v6 = v3 - 1;
  v7 = 0;
  v8 = *((_BYTE *)a2 + 16);
  v9 = *a2;
  v10 = *(_QWORD *)(a1 + 8);
  v11 = a2[1];
  v12 = 1;
  v13 = 0xBF58476D1CE4E5B9LL
      * ((37 * (unsigned int)v8)
       | (((0xBF58476D1CE4E5B9LL
          * (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4)
           | ((unsigned __int64)(((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4)) << 32)))
         ^ ((0xBF58476D1CE4E5B9LL
           * (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4)
            | ((unsigned __int64)(((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4)) << 32))) >> 31)) << 32));
  for ( i = v6 & ((v13 >> 31) ^ v13); ; i = v6 & v17 )
  {
    v15 = (__int64 *)(v10 + 24LL * i);
    v16 = *v15;
    if ( *v15 == v9 && v11 == v15[1] && v8 == *((_BYTE *)v15 + 16) )
    {
      *a3 = v15;
      return 1;
    }
    if ( v16 == -4096 )
      break;
    if ( v16 == -8192 && v15[1] == -8192 && *((_BYTE *)v15 + 16) == 0xFE && !v7 )
      v7 = v10 + 24LL * i;
LABEL_7:
    v17 = v12 + i;
    ++v12;
  }
  if ( v15[1] != -4096 || *((_BYTE *)v15 + 16) != 0xFF )
    goto LABEL_7;
  if ( !v7 )
    v7 = v10 + 24LL * i;
  *a3 = v7;
  return 0;
}
