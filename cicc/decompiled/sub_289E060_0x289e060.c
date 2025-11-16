// Function: sub_289E060
// Address: 0x289e060
//
__int64 __fastcall sub_289E060(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v7; // eax
  _QWORD *v8; // rdi
  __int64 v9; // rsi
  __int64 result; // rax
  __int64 v11; // rsi
  int v12; // edx
  __int64 v13; // rdi
  unsigned int v14; // esi
  __int64 v15; // rcx
  __int64 v16; // r9
  __int64 v17; // rdi
  int v18; // r11d
  __int64 v19; // rdx
  __int64 v20; // r8
  __int64 v21; // rax
  __int64 v22; // r10
  __int64 v23; // rdi
  __int64 v24; // r14
  __int64 *v25; // rbx
  __int64 v26; // rsi
  int v27; // edi
  _QWORD *v28; // rax
  int v29; // r8d
  int v30; // ebx
  __int64 v31[2]; // [rsp+8h] [rbp-48h] BYREF
  __int64 v32[7]; // [rsp+18h] [rbp-38h] BYREF

  v7 = *(_DWORD *)(a3 + 16);
  v31[0] = a2;
  if ( !v7 )
  {
    v8 = *(_QWORD **)(a3 + 32);
    v9 = (__int64)&v8[*(unsigned int *)(a3 + 40)];
    result = (__int64)sub_28946A0(v8, v9, v31);
    if ( v9 == result )
      return result;
    goto LABEL_6;
  }
  result = *(unsigned int *)(a3 + 24);
  v11 = *(_QWORD *)(a3 + 8);
  if ( !(_DWORD)result )
    return result;
  v12 = result - 1;
  result = ((_DWORD)result - 1) & (unsigned int)((LODWORD(v31[0]) >> 9) ^ (LODWORD(v31[0]) >> 4));
  v13 = *(_QWORD *)(v11 + 8 * result);
  if ( v31[0] == v13 )
  {
LABEL_6:
    v14 = *(_DWORD *)(a4 + 24);
    if ( v14 )
    {
      v15 = v31[0];
      v16 = v14 - 1;
      v17 = *(_QWORD *)(a4 + 8);
      v18 = 1;
      v19 = (unsigned int)v16 & ((LODWORD(v31[0]) >> 9) ^ (LODWORD(v31[0]) >> 4));
      v20 = v17 + 56 * v19;
      v21 = 0;
      v22 = *(_QWORD *)v20;
      if ( v31[0] == *(_QWORD *)v20 )
      {
LABEL_8:
        v23 = v20 + 8;
        if ( !*(_BYTE *)(v20 + 36) )
          goto LABEL_9;
        goto LABEL_21;
      }
      while ( v22 != -4096 )
      {
        if ( !v21 && v22 == -8192 )
          v21 = v20;
        v19 = (unsigned int)v16 & (v18 + (_DWORD)v19);
        v20 = v17 + 56LL * (unsigned int)v19;
        v22 = *(_QWORD *)v20;
        if ( v31[0] == *(_QWORD *)v20 )
          goto LABEL_8;
        ++v18;
      }
      v30 = *(_DWORD *)(a4 + 16);
      if ( !v21 )
        v21 = v20;
      ++*(_QWORD *)a4;
      v27 = v30 + 1;
      v32[0] = v21;
      if ( 4 * (v30 + 1) < 3 * v14 )
      {
        v20 = v14 >> 3;
        if ( v14 - *(_DWORD *)(a4 + 20) - v27 > (unsigned int)v20 )
          goto LABEL_18;
        goto LABEL_17;
      }
    }
    else
    {
      ++*(_QWORD *)a4;
      v32[0] = 0;
    }
    v14 *= 2;
LABEL_17:
    sub_289DE10(a4, v14);
    sub_289DC90(a4, v31, v32);
    v15 = v31[0];
    v27 = *(_DWORD *)(a4 + 16) + 1;
    v21 = v32[0];
LABEL_18:
    *(_DWORD *)(a4 + 16) = v27;
    if ( *(_QWORD *)v21 != -4096 )
      --*(_DWORD *)(a4 + 20);
    *(_QWORD *)v21 = v15;
    v23 = v21 + 8;
    *(_QWORD *)(v21 + 8) = 0;
    *(_QWORD *)(v21 + 16) = v21 + 40;
    *(_QWORD *)(v21 + 24) = 2;
    *(_DWORD *)(v21 + 32) = 0;
    *(_BYTE *)(v21 + 36) = 1;
LABEL_21:
    v28 = *(_QWORD **)(v23 + 8);
    v15 = *(unsigned int *)(v23 + 20);
    v19 = (__int64)&v28[v15];
    if ( v28 != (_QWORD *)v19 )
    {
      while ( a1 != *v28 )
      {
        if ( (_QWORD *)v19 == ++v28 )
          goto LABEL_24;
      }
      goto LABEL_10;
    }
LABEL_24:
    if ( (unsigned int)v15 < *(_DWORD *)(v23 + 16) )
    {
      *(_DWORD *)(v23 + 20) = v15 + 1;
      *(_QWORD *)v19 = a1;
      ++*(_QWORD *)v23;
      goto LABEL_10;
    }
LABEL_9:
    sub_C8CC70(v23, a1, v19, v15, v20, v16);
LABEL_10:
    v24 = v31[0];
    result = 32LL * (*(_DWORD *)(v31[0] + 4) & 0x7FFFFFF);
    if ( (*(_BYTE *)(v31[0] + 7) & 0x40) != 0 )
    {
      v25 = *(__int64 **)(v31[0] - 8);
      v24 = (__int64)v25 + result;
    }
    else
    {
      v25 = (__int64 *)(v31[0] - result);
    }
    for ( ; (__int64 *)v24 != v25; result = sub_289E060(a1, v26, a3, a4) )
    {
      v26 = *v25;
      v25 += 4;
    }
    return result;
  }
  v29 = 1;
  while ( v13 != -4096 )
  {
    result = v12 & (unsigned int)(v29 + result);
    v13 = *(_QWORD *)(v11 + 8LL * (unsigned int)result);
    if ( v31[0] == v13 )
      goto LABEL_6;
    ++v29;
  }
  return result;
}
