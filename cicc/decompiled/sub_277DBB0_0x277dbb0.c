// Function: sub_277DBB0
// Address: 0x277dbb0
//
__int64 __fastcall sub_277DBB0(__int64 a1, __int64 *a2, __int64 **a3)
{
  int v4; // edx
  int v6; // edx
  __int64 *v7; // r15
  __int64 v8; // rsi
  __int64 v9; // rbx
  __int64 v10; // rdi
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // r11
  __int64 v14; // r10
  int v15; // r14d
  unsigned int i; // eax
  __int64 *v17; // r12
  __int64 v18; // r13
  unsigned int v19; // eax

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    *a3 = 0;
    return 0;
  }
  v6 = v4 - 1;
  v7 = 0;
  v8 = *a2;
  v9 = *(_QWORD *)(a1 + 8);
  v10 = a2[2];
  v11 = a2[3];
  v12 = a2[4];
  v13 = a2[1];
  v14 = a2[5];
  v15 = 1;
  for ( i = v6
          & (((0xBF58476D1CE4E5B9LL * v13) >> 31)
           ^ (484763065 * v13)
           ^ ((unsigned int)v14 >> 9)
           ^ ((unsigned int)v14 >> 4)
           ^ ((unsigned int)v12 >> 9)
           ^ ((unsigned int)v12 >> 4)
           ^ ((unsigned int)v11 >> 9)
           ^ ((unsigned int)v11 >> 4)
           ^ ((unsigned int)v10 >> 9)
           ^ ((unsigned int)v10 >> 4)
           ^ ((unsigned int)v8 >> 9)
           ^ ((unsigned int)v8 >> 4)); ; i = v6 & v19 )
  {
    v17 = (__int64 *)(v9 + 56LL * i);
    v18 = *v17;
    if ( v8 == *v17 && v13 == v17[1] && v10 == v17[2] && v11 == v17[3] && v12 == v17[4] && v14 == v17[5] )
    {
      *a3 = v17;
      return 1;
    }
    if ( v18 == -4096 )
      break;
    if ( v18 == -8192 && v17[1] == -4 && !v17[2] && !v17[3] && !v17[4] && !(v17[5] | (unsigned __int64)v7) )
      v7 = (__int64 *)(v9 + 56LL * i);
LABEL_7:
    v19 = v15 + i;
    ++v15;
  }
  if ( v17[1] != -3 || v17[2] || v17[3] || v17[4] || v17[5] )
    goto LABEL_7;
  if ( !v7 )
    v7 = (__int64 *)(v9 + 56LL * i);
  *a3 = v7;
  return 0;
}
