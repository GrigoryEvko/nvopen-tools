// Function: sub_2BF9A30
// Address: 0x2bf9a30
//
__int64 __fastcall sub_2BF9A30(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  __int64 result; // rax
  __int64 *v9; // r12
  __int64 *v10; // r14
  __int64 v11; // r8
  __int64 *v12; // rdi
  __int64 v13; // rcx
  unsigned int v14; // esi
  __int64 *v15; // r10
  int v16; // edx
  int v17; // r11d
  int v18; // eax
  __int64 *v19; // [rsp+8h] [rbp-38h] BYREF

  v7 = *(unsigned int *)(a1 + 40);
  if ( v7 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
  {
    sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), v7 + 1, 8u, a5, a6);
    v7 = *(unsigned int *)(a1 + 40);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8 * v7) = a2;
  result = (unsigned int)(*(_DWORD *)(a1 + 40) + 1);
  *(_DWORD *)(a1 + 40) = result;
  if ( (unsigned int)result > 8 )
  {
    v9 = *(__int64 **)(a1 + 32);
    v10 = &v9[result];
    while ( 1 )
    {
      v14 = *(_DWORD *)(a1 + 24);
      if ( !v14 )
        break;
      v11 = *(_QWORD *)(a1 + 8);
      result = (v14 - 1) & (((unsigned int)*v9 >> 9) ^ ((unsigned int)*v9 >> 4));
      v12 = (__int64 *)(v11 + 8 * result);
      v13 = *v12;
      if ( *v9 != *v12 )
      {
        v17 = 1;
        v15 = 0;
        while ( v13 != -4096 )
        {
          if ( v15 || v13 != -8192 )
            v12 = v15;
          result = (v14 - 1) & (v17 + (_DWORD)result);
          v13 = *(_QWORD *)(v11 + 8LL * (unsigned int)result);
          if ( *v9 == v13 )
            goto LABEL_7;
          ++v17;
          v15 = v12;
          v12 = (__int64 *)(v11 + 8LL * (unsigned int)result);
        }
        v18 = *(_DWORD *)(a1 + 16);
        if ( !v15 )
          v15 = v12;
        ++*(_QWORD *)a1;
        v16 = v18 + 1;
        v19 = v15;
        if ( 4 * (v18 + 1) < 3 * v14 )
        {
          if ( v14 - *(_DWORD *)(a1 + 20) - v16 <= v14 >> 3 )
          {
LABEL_11:
            sub_2BF9860(a1, v14);
            sub_2BF2A20(a1, v9, &v19);
            v15 = v19;
            v16 = *(_DWORD *)(a1 + 16) + 1;
          }
          *(_DWORD *)(a1 + 16) = v16;
          if ( *v15 != -4096 )
            --*(_DWORD *)(a1 + 20);
          result = *v9;
          *v15 = *v9;
          goto LABEL_7;
        }
LABEL_10:
        v14 *= 2;
        goto LABEL_11;
      }
LABEL_7:
      if ( v10 == ++v9 )
        return result;
    }
    ++*(_QWORD *)a1;
    v19 = 0;
    goto LABEL_10;
  }
  return result;
}
