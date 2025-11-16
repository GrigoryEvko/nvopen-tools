// Function: sub_D4CC10
// Address: 0xd4cc10
//
__int64 __fastcall sub_D4CC10(__int64 a1, __int64 a2)
{
  int v3; // eax
  __int64 result; // rax
  __int64 v5; // rdx
  __int64 i; // rdx
  __int64 *v7; // rbx
  __int64 *v8; // r13
  __int64 v9; // rdi
  __int64 *v10; // rbx
  __int64 *j; // r13
  __int64 v12; // rsi
  __int64 v13; // rdi
  __int64 v14; // rdx
  unsigned int v15; // ecx
  unsigned int v16; // eax
  int v17; // eax
  unsigned __int64 v18; // rax
  __int64 v19; // rax
  int v20; // ebx
  __int64 v21; // r13
  __int64 v22; // rcx
  __int64 *v23; // rbx
  __int64 *v24; // r14
  __int64 v25; // rdi
  unsigned int v26; // ecx
  __int64 v27; // rsi

  v3 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  if ( v3 )
  {
    v15 = 4 * v3;
    a2 = 64;
    v5 = *(unsigned int *)(a1 + 24);
    if ( (unsigned int)(4 * v3) < 0x40 )
      v15 = 64;
    if ( v15 >= (unsigned int)v5 )
      goto LABEL_4;
    v16 = v3 - 1;
    if ( v16 )
    {
      _BitScanReverse(&v16, v16);
      v17 = 1 << (33 - (v16 ^ 0x1F));
      if ( v17 < 64 )
        v17 = 64;
      if ( v17 == (_DWORD)v5 )
        goto LABEL_24;
      v18 = (4 * v17 / 3u + 1) | ((unsigned __int64)(4 * v17 / 3u + 1) >> 1);
      v19 = ((((v18 >> 2) | v18 | (((v18 >> 2) | v18) >> 4)) >> 8)
           | (v18 >> 2)
           | v18
           | (((v18 >> 2) | v18) >> 4)
           | (((((v18 >> 2) | v18 | (((v18 >> 2) | v18) >> 4)) >> 8) | (v18 >> 2) | v18 | (((v18 >> 2) | v18) >> 4)) >> 16))
          + 1;
      v20 = v19;
      v21 = 16 * v19;
    }
    else
    {
      v21 = 2048;
      v20 = 128;
    }
    sub_C7D6A0(*(_QWORD *)(a1 + 8), 16LL * (unsigned int)v5, 8);
    *(_DWORD *)(a1 + 24) = v20;
    a2 = 8;
    *(_QWORD *)(a1 + 8) = sub_C7D670(v21, 8);
LABEL_24:
    result = (__int64)sub_D4CBD0(a1);
    goto LABEL_7;
  }
  result = *(unsigned int *)(a1 + 20);
  if ( !(_DWORD)result )
    goto LABEL_7;
  v5 = *(unsigned int *)(a1 + 24);
  if ( (unsigned int)v5 > 0x40 )
  {
    a2 = 16LL * (unsigned int)v5;
    result = sub_C7D6A0(*(_QWORD *)(a1 + 8), a2, 8);
    *(_QWORD *)(a1 + 8) = 0;
    *(_QWORD *)(a1 + 16) = 0;
    *(_DWORD *)(a1 + 24) = 0;
    goto LABEL_7;
  }
LABEL_4:
  result = *(_QWORD *)(a1 + 8);
  for ( i = result + 16 * v5; i != result; result += 16 )
    *(_QWORD *)result = -4096;
  *(_QWORD *)(a1 + 16) = 0;
LABEL_7:
  v7 = *(__int64 **)(a1 + 32);
  v8 = *(__int64 **)(a1 + 40);
  if ( v7 != v8 )
  {
    do
    {
      v9 = *v7++;
      sub_D47BB0(v9, a2);
    }
    while ( v8 != v7 );
    result = *(_QWORD *)(a1 + 32);
    if ( *(_QWORD *)(a1 + 40) != result )
      *(_QWORD *)(a1 + 40) = result;
  }
  v10 = *(__int64 **)(a1 + 120);
  for ( j = &v10[2 * *(unsigned int *)(a1 + 128)]; j != v10; result = sub_C7D6A0(v13, v12, 16) )
  {
    v12 = v10[1];
    v13 = *v10;
    v10 += 2;
  }
  v14 = *(unsigned int *)(a1 + 80);
  *(_DWORD *)(a1 + 128) = 0;
  if ( (_DWORD)v14 )
  {
    result = *(_QWORD *)(a1 + 72);
    *(_QWORD *)(a1 + 136) = 0;
    v22 = *(_QWORD *)result;
    v23 = (__int64 *)(result + 8 * v14);
    v24 = (__int64 *)(result + 8);
    *(_QWORD *)(a1 + 56) = *(_QWORD *)result;
    *(_QWORD *)(a1 + 64) = v22 + 4096;
    if ( v23 != (__int64 *)(result + 8) )
    {
      while ( 1 )
      {
        v25 = *v24;
        v26 = (unsigned int)(((__int64)v24 - result) >> 3) >> 7;
        v27 = 4096LL << v26;
        if ( v26 >= 0x1E )
          v27 = 0x40000000000LL;
        ++v24;
        result = sub_C7D6A0(v25, v27, 16);
        if ( v23 == v24 )
          break;
        result = *(_QWORD *)(a1 + 72);
      }
    }
    *(_DWORD *)(a1 + 80) = 1;
  }
  return result;
}
