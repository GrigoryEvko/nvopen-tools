// Function: sub_297D7C0
// Address: 0x297d7c0
//
__int64 __fastcall sub_297D7C0(__int64 a1, __int64 *a2, _QWORD *a3)
{
  int v3; // r10d
  int v5; // r10d
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // rdi
  unsigned int v9; // edx
  __int64 v10; // rsi
  unsigned __int64 v11; // rdx
  __int64 v12; // r12
  int v13; // ebx
  unsigned int i; // eax
  _QWORD *v15; // rdx
  __int64 v16; // r11
  unsigned int v17; // eax
  __int64 result; // rax

  v3 = *(_DWORD *)(a1 + 24);
  if ( !v3 )
  {
    *a3 = 0;
    return 0;
  }
  v5 = v3 - 1;
  v6 = *a2;
  v7 = *(_QWORD *)(a1 + 8);
  v8 = a2[1];
  v9 = *a2;
  v10 = a2[2];
  v11 = 0xBF58476D1CE4E5B9LL
      * ((969526130 * ((v9 >> 9) ^ ((unsigned int)v6 >> 4)))
       | ((unsigned __int64)(((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4)) << 32));
  v12 = 0;
  v13 = 1;
  for ( i = v5
          & (((0xBF58476D1CE4E5B9LL
             * ((unsigned int)(v11 >> 31) ^ (unsigned int)v11
              | ((unsigned __int64)(((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4)) << 32))) >> 31)
           ^ (484763065 * ((v11 >> 31) ^ v11))); ; i = v5 & v17 )
  {
    v15 = (_QWORD *)(v7 + 56LL * i);
    v16 = v15[2];
    if ( v16 == v10 && v8 == v15[1] && v6 == *v15 )
    {
      *a3 = v15;
      return 1;
    }
    if ( v16 == -4096 )
      break;
    if ( v16 == -8192 && v15[1] == -8192 && *v15 == -8192 && !v12 )
      v12 = v7 + 56LL * i;
LABEL_7:
    v17 = v13 + i;
    ++v13;
  }
  if ( v15[1] != -4096 || *v15 != -4096 )
    goto LABEL_7;
  if ( !v12 )
    v12 = v7 + 56LL * i;
  result = 0;
  *a3 = v12;
  return result;
}
