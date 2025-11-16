// Function: sub_D0A160
// Address: 0xd0a160
//
__int64 __fastcall sub_D0A160(__int64 a1, unsigned __int64 *a2, __int64 **a3)
{
  __int64 v4; // r10
  int v5; // r9d
  unsigned __int64 v6; // r8
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // r11
  unsigned __int64 v9; // rsi
  __int64 *v10; // r13
  int v11; // r12d
  unsigned int i; // eax
  __int64 *v13; // rdx
  __int64 v14; // rbx
  unsigned int v15; // eax
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
  v8 = a2[3];
  v9 = a2[1];
  v10 = 0;
  v11 = 1;
  for ( i = v5
          & (((0xBF58476D1CE4E5B9LL
             * ((unsigned int)((0xBF58476D1CE4E5B9LL * v8) >> 31)
              ^ (484763065 * (_DWORD)v8)
              ^ (unsigned int)v6
              ^ (unsigned int)(v6 >> 9)
              | ((unsigned __int64)((unsigned int)((0xBF58476D1CE4E5B9LL * v9) >> 31)
                                  ^ (484763065 * (_DWORD)v9)
                                  ^ (unsigned int)v7
                                  ^ (unsigned int)(v7 >> 9)) << 32))) >> 31)
           ^ (484763065 * (((0xBF58476D1CE4E5B9LL * v8) >> 31) ^ (484763065 * v8) ^ v6 ^ (v6 >> 9)))); ; i = v5 & v15 )
  {
    v13 = (__int64 *)(v4 + 40LL * i);
    v14 = *v13;
    if ( v7 == *v13 && v9 == v13[1] && v6 == v13[2] && v8 == v13[3] )
    {
      *a3 = v13;
      return 1;
    }
    if ( v14 == -4 )
      break;
    if ( v14 == -16 && v13[1] == -4 && v13[2] == -16 && v13[3] == -4 && !v10 )
      v10 = (__int64 *)(v4 + 40LL * i);
LABEL_8:
    v15 = v11 + i;
    ++v11;
  }
  if ( v13[1] != -3 || v13[2] != -4 || v13[3] != -3 )
    goto LABEL_8;
  if ( !v10 )
    v10 = (__int64 *)(v4 + 40LL * i);
  *a3 = v10;
  return 0;
}
