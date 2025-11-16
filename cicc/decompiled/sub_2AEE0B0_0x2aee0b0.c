// Function: sub_2AEE0B0
// Address: 0x2aee0b0
//
void __fastcall sub_2AEE0B0(__int64 a1, __int64 a2, __int64 a3)
{
  const void *v5; // r12
  __int64 v6; // rbx
  __int64 v7; // r9
  __int64 v8; // r8
  int v9; // r11d
  __int64 *v10; // r10
  unsigned int v11; // edx
  __int64 *v12; // rdi
  __int64 v13; // rcx
  unsigned int v14; // esi
  __int64 v15; // rax
  int v16; // ecx
  int v17; // edi
  __int64 v18; // rax
  __int64 v19; // r15
  __int64 v20; // [rsp-48h] [rbp-48h] BYREF
  __int64 *v21; // [rsp-40h] [rbp-40h] BYREF

  if ( a2 != a3 )
  {
    v5 = (const void *)(a1 + 48);
    v6 = a2;
    while ( 1 )
    {
      v14 = *(_DWORD *)(a1 + 24);
      v15 = *(_QWORD *)(*(_QWORD *)(v6 + 24) + 40LL);
      v20 = v15;
      if ( !v14 )
        break;
      v7 = v14 - 1;
      v8 = *(_QWORD *)(a1 + 8);
      v9 = 1;
      v10 = 0;
      v11 = v7 & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
      v12 = (__int64 *)(v8 + 8LL * v11);
      v13 = *v12;
      if ( v15 != *v12 )
      {
        while ( v13 != -4096 )
        {
          if ( v10 || v13 != -8192 )
            v12 = v10;
          v11 = v7 & (v9 + v11);
          v13 = *(_QWORD *)(v8 + 8LL * v11);
          if ( v15 == v13 )
            goto LABEL_4;
          ++v9;
          v10 = v12;
          v12 = (__int64 *)(v8 + 8LL * v11);
        }
        if ( !v10 )
          v10 = v12;
        v17 = *(_DWORD *)(a1 + 16);
        ++*(_QWORD *)a1;
        v16 = v17 + 1;
        v21 = v10;
        if ( 4 * (v17 + 1) < 3 * v14 )
        {
          if ( v14 - *(_DWORD *)(a1 + 20) - v16 > v14 >> 3 )
            goto LABEL_20;
          goto LABEL_10;
        }
LABEL_9:
        v14 *= 2;
LABEL_10:
        sub_CF28B0(a1, v14);
        sub_D6B660(a1, &v20, &v21);
        v15 = v20;
        v10 = v21;
        v16 = *(_DWORD *)(a1 + 16) + 1;
LABEL_20:
        *(_DWORD *)(a1 + 16) = v16;
        if ( *v10 != -4096 )
          --*(_DWORD *)(a1 + 20);
        *v10 = v15;
        v18 = *(unsigned int *)(a1 + 40);
        v19 = v20;
        if ( v18 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
        {
          sub_C8D5F0(a1 + 32, v5, v18 + 1, 8u, v8, v7);
          v18 = *(unsigned int *)(a1 + 40);
        }
        *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8 * v18) = v19;
        ++*(_DWORD *)(a1 + 40);
        goto LABEL_4;
      }
      do
LABEL_4:
        v6 = *(_QWORD *)(v6 + 8);
      while ( v6 && (unsigned __int8)(**(_BYTE **)(v6 + 24) - 30) > 0xAu );
      if ( a3 == v6 )
        return;
    }
    ++*(_QWORD *)a1;
    v21 = 0;
    goto LABEL_9;
  }
}
