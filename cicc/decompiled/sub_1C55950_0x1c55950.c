// Function: sub_1C55950
// Address: 0x1c55950
//
__int64 **__fastcall sub_1C55950(__int64 **a1, __int64 *a2, __int64 *a3)
{
  __int64 v4; // rdx
  __int64 v6; // rcx
  __int64 v7; // rdi
  __int64 v8; // r9
  int v9; // r12d
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // rax
  unsigned __int64 v12; // r10
  unsigned int i; // eax
  __int64 *v14; // r10
  unsigned int v15; // eax
  __int64 *v16; // rax
  __int64 *v17; // rdx
  __int64 *v19; // rax

  v4 = *((unsigned int *)a2 + 6);
  v6 = a2[1];
  if ( (_DWORD)v4 )
  {
    v7 = *a3;
    v8 = a3[1];
    v9 = 1;
    v10 = (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4)
         | ((unsigned __int64)(((unsigned int)*a3 >> 9) ^ ((unsigned int)*a3 >> 4)) << 32))
        - 1
        - ((unsigned __int64)(((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4)) << 32);
    v11 = ((v10 >> 22) ^ v10) - 1 - (((v10 >> 22) ^ v10) << 13);
    v12 = ((9 * ((v11 >> 8) ^ v11)) >> 15) ^ (9 * ((v11 >> 8) ^ v11));
    for ( i = (v4 - 1) & (((v12 - 1 - (v12 << 27)) >> 31) ^ (v12 - 1 - ((_DWORD)v12 << 27))); ; i = (v4 - 1) & v15 )
    {
      v14 = (__int64 *)(v6 + 32LL * i);
      if ( *v14 == v7 && v14[1] == v8 )
        break;
      if ( *v14 == -8 && v14[1] == -8 )
        goto LABEL_7;
      v15 = v9 + i;
      ++v9;
    }
    v19 = (__int64 *)*a2;
    *a1 = a2;
    a1[2] = v14;
    a1[1] = v19;
    a1[3] = (__int64 *)(32 * v4 + v6);
  }
  else
  {
LABEL_7:
    v16 = (__int64 *)*a2;
    *a1 = a2;
    v17 = (__int64 *)(v6 + 32 * v4);
    a1[1] = v16;
    a1[2] = v17;
    a1[3] = v17;
  }
  return a1;
}
