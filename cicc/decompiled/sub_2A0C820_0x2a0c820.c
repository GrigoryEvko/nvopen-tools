// Function: sub_2A0C820
// Address: 0x2a0c820
//
__int64 __fastcall sub_2A0C820(__int64 a1, __int64 *a2, __int64 **a3)
{
  __int64 v4; // r10
  int v5; // r9d
  __int64 v6; // rdi
  __int64 v7; // r8
  __int64 v8; // rsi
  __int64 *v9; // r12
  int v10; // ebx
  unsigned int i; // eax
  __int64 *v12; // rdx
  __int64 v13; // r11
  unsigned int v14; // eax
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
  v6 = a2[2];
  v7 = *a2;
  v8 = a2[1];
  v9 = 0;
  v10 = 1;
  for ( i = v5
          & (((0xBF58476D1CE4E5B9LL
             * (((((0xBF58476D1CE4E5B9LL * ((v7 << 32) | ((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4))) >> 31)
                ^ (0xBF58476D1CE4E5B9LL * ((v7 << 32) | ((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4)))) << 32)
              | ((unsigned int)v6 >> 4) ^ ((unsigned int)v6 >> 9))) >> 31)
           ^ (484763065 * (((unsigned int)v6 >> 4) ^ ((unsigned int)v6 >> 9)))); ; i = v5 & v14 )
  {
    v12 = (__int64 *)(v4 + 24LL * i);
    v13 = *v12;
    if ( *v12 == v7 && v8 == v12[1] && v6 == v12[2] )
    {
      *a3 = v12;
      return 1;
    }
    if ( v13 == -1 )
      break;
    if ( v13 == -2 && v12[1] == -8192 && v12[2] == -8192 && !v9 )
      v9 = (__int64 *)(v4 + 24LL * i);
LABEL_8:
    v14 = v10 + i;
    ++v10;
  }
  if ( v12[1] != -4096 || v12[2] != -4096 )
    goto LABEL_8;
  if ( !v9 )
    v9 = (__int64 *)(v4 + 24LL * i);
  result = 0;
  *a3 = v9;
  return result;
}
