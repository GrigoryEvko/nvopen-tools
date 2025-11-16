// Function: sub_2573C90
// Address: 0x2573c90
//
__int64 __fastcall sub_2573C90(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 *v10; // r12
  __int64 *v11; // r14
  __int64 v12; // r8
  unsigned int v13; // eax
  __int64 *v14; // rdi
  __int64 v15; // rcx
  unsigned int v16; // esi
  __int64 *v17; // r10
  int v18; // edx
  int v19; // r11d
  int v20; // eax
  __int64 *v21; // [rsp+8h] [rbp-38h] BYREF

  v7 = *(unsigned int *)(a1 + 40);
  if ( v7 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
  {
    sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), v7 + 1, 8u, a5, a6);
    v7 = *(unsigned int *)(a1 + 40);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8 * v7) = a2;
  v8 = (unsigned int)(*(_DWORD *)(a1 + 40) + 1);
  *(_DWORD *)(a1 + 40) = v8;
  if ( (unsigned int)v8 > 8 )
  {
    v10 = *(__int64 **)(a1 + 32);
    v11 = &v10[v8];
    while ( 1 )
    {
      v16 = *(_DWORD *)(a1 + 24);
      if ( !v16 )
        break;
      v12 = *(_QWORD *)(a1 + 8);
      v13 = (v16 - 1) & (((unsigned int)*v10 >> 9) ^ ((unsigned int)*v10 >> 4));
      v14 = (__int64 *)(v12 + 8LL * v13);
      v15 = *v14;
      if ( *v10 != *v14 )
      {
        v19 = 1;
        v17 = 0;
        while ( v15 != -4096 )
        {
          if ( v17 || v15 != -8192 )
            v14 = v17;
          v13 = (v16 - 1) & (v19 + v13);
          v15 = *(_QWORD *)(v12 + 8LL * v13);
          if ( *v10 == v15 )
            goto LABEL_7;
          ++v19;
          v17 = v14;
          v14 = (__int64 *)(v12 + 8LL * v13);
        }
        v20 = *(_DWORD *)(a1 + 16);
        if ( !v17 )
          v17 = v14;
        ++*(_QWORD *)a1;
        v18 = v20 + 1;
        v21 = v17;
        if ( 4 * (v20 + 1) < 3 * v16 )
        {
          if ( v16 - *(_DWORD *)(a1 + 20) - v18 <= v16 >> 3 )
          {
LABEL_11:
            sub_CF4090(a1, v16);
            sub_23FDF60(a1, v10, &v21);
            v17 = v21;
            v18 = *(_DWORD *)(a1 + 16) + 1;
          }
          *(_DWORD *)(a1 + 16) = v18;
          if ( *v17 != -4096 )
            --*(_DWORD *)(a1 + 20);
          *v17 = *v10;
          goto LABEL_7;
        }
LABEL_10:
        v16 *= 2;
        goto LABEL_11;
      }
LABEL_7:
      if ( v11 == ++v10 )
        return 1;
    }
    ++*(_QWORD *)a1;
    v21 = 0;
    goto LABEL_10;
  }
  return 1;
}
