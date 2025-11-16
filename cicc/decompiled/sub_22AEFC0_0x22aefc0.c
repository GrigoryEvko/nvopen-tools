// Function: sub_22AEFC0
// Address: 0x22aefc0
//
__int64 __fastcall sub_22AEFC0(__int64 a1, __int64 a2)
{
  __int64 v4; // r13
  __int64 result; // rax
  __int64 v6; // r9
  __int64 v7; // rsi
  unsigned int v8; // ecx
  __int64 *v9; // rdx
  __int64 v10; // r8
  int v11; // r15d
  __int64 v12; // rdx
  __int64 v13; // r8
  __int64 v14; // r12
  unsigned int v15; // esi
  __int64 *v16; // rcx
  __int64 v17; // r10
  int v18; // eax
  __int64 v19; // rdi
  int v20; // ecx
  int v21; // edx
  int v22; // r10d
  int v23; // [rsp+4h] [rbp-3Ch]
  __int64 v24; // [rsp+8h] [rbp-38h]
  int v25; // [rsp+8h] [rbp-38h]

  v4 = *(_QWORD *)(a1 + 16);
  result = *(unsigned int *)(a2 + 24);
  v6 = *(_QWORD *)(a2 + 8);
  v7 = *(_QWORD *)(v4 + 40);
  if ( (_DWORD)result )
  {
    v8 = (result - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
    v9 = (__int64 *)(v6 + 16LL * v8);
    v10 = *v9;
    if ( v7 == *v9 )
      goto LABEL_3;
    v21 = 1;
    while ( v10 != -4096 )
    {
      v22 = v21 + 1;
      v8 = (result - 1) & (v21 + v8);
      v9 = (__int64 *)(v6 + 16LL * v8);
      v10 = *v9;
      if ( v7 == *v9 )
        goto LABEL_3;
      v21 = v22;
    }
  }
  v9 = (__int64 *)(v6 + 16LL * (unsigned int)result);
LABEL_3:
  v11 = *((_DWORD *)v9 + 2);
  if ( (*(_DWORD *)(v4 + 4) & 0x7FFFFFF) != 0 )
  {
    v12 = *(unsigned int *)(a1 + 136);
    v13 = a1 + 128;
    v14 = 0;
    while ( 1 )
    {
      v19 = *(_QWORD *)(*(_QWORD *)(v4 - 8) + 32LL * *(unsigned int *)(v4 + 72) + 8 * v14);
      if ( (_DWORD)result )
      {
        v15 = (result - 1) & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
        v16 = (__int64 *)(v6 + 16LL * v15);
        v17 = *v16;
        if ( v19 == *v16 )
          goto LABEL_6;
        v20 = 1;
        while ( v17 != -4096 )
        {
          v15 = (result - 1) & (v20 + v15);
          v25 = v20 + 1;
          v16 = (__int64 *)(v6 + 16LL * v15);
          v17 = *v16;
          if ( v19 == *v16 )
            goto LABEL_6;
          v20 = v25;
        }
      }
      v16 = (__int64 *)(v6 + 16 * result);
LABEL_6:
      v18 = *((_DWORD *)v16 + 2) - v11;
      if ( v12 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 140) )
      {
        v23 = *((_DWORD *)v16 + 2) - v11;
        v24 = v13;
        sub_C8D5F0(v13, (const void *)(a1 + 144), v12 + 1, 4u, v13, v12 + 1);
        v12 = *(unsigned int *)(a1 + 136);
        v18 = v23;
        v13 = v24;
      }
      ++v14;
      *(_DWORD *)(*(_QWORD *)(a1 + 128) + 4 * v12) = v18;
      v12 = (unsigned int)(*(_DWORD *)(a1 + 136) + 1);
      *(_DWORD *)(a1 + 136) = v12;
      result = *(_DWORD *)(v4 + 4) & 0x7FFFFFF;
      if ( (unsigned int)result <= (unsigned int)v14 )
        return result;
      v6 = *(_QWORD *)(a2 + 8);
      result = *(unsigned int *)(a2 + 24);
    }
  }
  return result;
}
