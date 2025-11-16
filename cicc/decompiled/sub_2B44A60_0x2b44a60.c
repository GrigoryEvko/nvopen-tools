// Function: sub_2B44A60
// Address: 0x2b44a60
//
__int64 __fastcall sub_2B44A60(__int64 a1, int *a2, _QWORD *a3)
{
  int v3; // r9d
  int v5; // ecx
  __int64 v6; // rsi
  __int64 v7; // rdi
  __int64 v8; // r11
  __int64 v9; // rbx
  __int64 v10; // r8
  unsigned __int64 v11; // rax
  unsigned __int64 v12; // r12
  int v13; // r14d
  __int64 v14; // r12
  int v15; // r9d
  __int64 v16; // r13
  unsigned int i; // eax
  __int64 v18; // r10
  __int64 v19; // r12
  unsigned int v20; // eax
  __int64 result; // rax

  v3 = *(_DWORD *)(a1 + 24);
  if ( !v3 )
  {
    *a3 = 0;
    return 0;
  }
  v5 = *a2;
  v6 = *((_QWORD *)a2 + 1);
  v7 = *(_QWORD *)(a1 + 8);
  v8 = *((_QWORD *)a2 + 2);
  v9 = *((_QWORD *)a2 + 3);
  v10 = *((_QWORD *)a2 + 4);
  v11 = ((unsigned __int64)(((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4)) << 32)
      | (unsigned int)((0xBF58476D1CE4E5B9LL
                      * ((unsigned int)(1512728442 * v5)
                       | ((unsigned __int64)(((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4)) << 32))) >> 31)
      ^ (-1747130070 * v5);
  v12 = (unsigned int)((0xBF58476D1CE4E5B9LL * v11) >> 31) ^ (484763065 * (_DWORD)v11)
      | ((unsigned __int64)(((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4)) << 32);
  v13 = 1;
  v14 = (unsigned int)((0xBF58476D1CE4E5B9LL * v12) >> 31) ^ (484763065 * (_DWORD)v12);
  v15 = v3 - 1;
  v16 = 0;
  for ( i = v15
          & (((0xBF58476D1CE4E5B9LL
             * (v14 | ((unsigned __int64)(((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4)) << 32))) >> 31)
           ^ (484763065 * v14)); ; i = v15 & v20 )
  {
    v18 = v7 + 40LL * i;
    v19 = *(_QWORD *)(v18 + 32);
    if ( v19 == v10
      && v9 == *(_QWORD *)(v18 + 24)
      && v8 == *(_QWORD *)(v18 + 16)
      && v6 == *(_QWORD *)(v18 + 8)
      && v5 == *(_DWORD *)v18 )
    {
      *a3 = v18;
      return 1;
    }
    if ( v19 == -4096 )
      break;
    if ( v19 == -8192
      && *(_QWORD *)(v18 + 24) == -8192
      && *(_QWORD *)(v18 + 16) == -8192
      && *(_QWORD *)(v18 + 8) == -8192
      && *(_DWORD *)v18 == -2
      && !v16 )
    {
      v16 = v7 + 40LL * i;
    }
LABEL_7:
    v20 = v13 + i;
    ++v13;
  }
  if ( *(_QWORD *)(v18 + 24) != -4096
    || *(_QWORD *)(v18 + 16) != -4096
    || *(_QWORD *)(v18 + 8) != -4096
    || *(_DWORD *)v18 != -1 )
  {
    goto LABEL_7;
  }
  if ( !v16 )
    v16 = v7 + 40LL * i;
  result = 0;
  *a3 = v16;
  return result;
}
