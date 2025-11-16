// Function: sub_103F4F0
// Address: 0x103f4f0
//
__int64 __fastcall sub_103F4F0(__int64 a1, __int64 *a2, __int64 **a3)
{
  int v4; // edx
  int v6; // edx
  __int64 v7; // r9
  __int64 v8; // r10
  __int64 v9; // r11
  __int64 v10; // r12
  __int64 v11; // rsi
  __int64 v12; // rbx
  __int64 v13; // r14
  __int64 v14; // rdi
  int v15; // r15d
  __int64 v16; // rax
  unsigned int v17; // r13d
  __int64 *v18; // r8
  __int64 *v19; // rax
  __int64 v20; // rsi
  unsigned int v21; // r13d
  __int64 v23; // [rsp-8h] [rbp-8h]

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    *a3 = 0;
    return 0;
  }
  v6 = v4 - 1;
  v7 = a2[1];
  v8 = a2[3];
  v9 = a2[4];
  v10 = a2[5];
  v11 = a2[2];
  *(&v23 - 6) = v11;
  v12 = a2[6];
  v13 = *(_QWORD *)(a1 + 8);
  v14 = *a2;
  *(&v23 - 7) = v12;
  v15 = 1;
  v16 = (unsigned int)((0xBF58476D1CE4E5B9LL * v11) >> 31)
      ^ (484763065 * (_DWORD)v11)
      ^ ((unsigned int)v10 >> 9)
      ^ ((unsigned int)v9 >> 9)
      ^ ((unsigned int)v8 >> 9)
      ^ ((unsigned int)v7 >> 9)
      ^ ((unsigned int)v7 >> 4)
      ^ ((unsigned int)v8 >> 4)
      ^ ((unsigned int)v9 >> 4)
      ^ ((unsigned int)v10 >> 4)
      ^ ((unsigned int)v12 >> 4)
      ^ ((unsigned int)v12 >> 9);
  v17 = v6
      & (((0xBF58476D1CE4E5B9LL * (((unsigned __int64)(((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4)) << 32) | v16)) >> 31)
       ^ (484763065 * v16));
  v18 = 0;
  while ( 1 )
  {
    v19 = (__int64 *)(v13 + 56LL * v17);
    v20 = *v19;
    if ( *v19 == v14
      && v7 == v19[1]
      && *(&v23 - 6) == v19[2]
      && v8 == v19[3]
      && v9 == v19[4]
      && v10 == v19[5]
      && *(&v23 - 7) == v19[6] )
    {
      *a3 = v19;
      return 1;
    }
    if ( v20 == -4096 )
      break;
    if ( v20 == -8192
      && v19[1] == -8192
      && v19[2] == -4
      && !v19[3]
      && !v19[4]
      && !v19[5]
      && !(v19[6] | (unsigned __int64)v18) )
    {
      v18 = (__int64 *)(v13 + 56LL * v17);
    }
LABEL_7:
    v21 = v15 + v17;
    ++v15;
    v17 = v6 & v21;
  }
  if ( v19[1] != -4096 || v19[2] != -3 || v19[3] || v19[4] || v19[5] || v19[6] )
    goto LABEL_7;
  if ( !v18 )
    v18 = (__int64 *)(v13 + 56LL * v17);
  *a3 = v18;
  return 0;
}
