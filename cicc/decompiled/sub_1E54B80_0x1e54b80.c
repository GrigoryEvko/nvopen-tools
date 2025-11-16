// Function: sub_1E54B80
// Address: 0x1e54b80
//
void __fastcall sub_1E54B80(__int64 a1, __int64 *a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  const void *v7; // r13
  __int64 *v8; // r12
  __int64 v10; // rdi
  int v11; // esi
  unsigned int v12; // eax
  __int64 *v13; // r10
  char v14; // cl
  unsigned int v15; // esi
  unsigned int v16; // eax
  __int64 *v17; // r8
  int v18; // edx
  __int64 v19; // rax
  __int64 v20; // rcx
  int v21; // r11d
  __int64 *v22; // [rsp-40h] [rbp-40h] BYREF

  if ( a2 != a3 )
  {
    v7 = (const void *)(a1 + 96);
    v8 = a2;
    while ( 1 )
    {
      v14 = *(_BYTE *)(a1 + 8) & 1;
      if ( v14 )
      {
        v10 = a1 + 16;
        v11 = 7;
      }
      else
      {
        v15 = *(_DWORD *)(a1 + 24);
        v10 = *(_QWORD *)(a1 + 16);
        if ( !v15 )
        {
          v16 = *(_DWORD *)(a1 + 8);
          ++*(_QWORD *)a1;
          v17 = 0;
          v18 = (v16 >> 1) + 1;
          goto LABEL_10;
        }
        v11 = v15 - 1;
      }
      v12 = v11 & (((unsigned int)*v8 >> 9) ^ ((unsigned int)*v8 >> 4));
      v13 = (__int64 *)(v10 + 8LL * v12);
      a6 = *v13;
      if ( *v8 == *v13 )
      {
LABEL_5:
        if ( a3 == ++v8 )
          return;
      }
      else
      {
        v21 = 1;
        v17 = 0;
        while ( a6 != -8 )
        {
          if ( v17 || a6 != -16 )
            v13 = v17;
          v12 = v11 & (v21 + v12);
          a6 = *(_QWORD *)(v10 + 8LL * v12);
          if ( *v8 == a6 )
            goto LABEL_5;
          ++v21;
          v17 = v13;
          v13 = (__int64 *)(v10 + 8LL * v12);
        }
        v16 = *(_DWORD *)(a1 + 8);
        if ( !v17 )
          v17 = v13;
        ++*(_QWORD *)a1;
        v18 = (v16 >> 1) + 1;
        if ( !v14 )
        {
          v15 = *(_DWORD *)(a1 + 24);
LABEL_10:
          if ( 3 * v15 <= 4 * v18 )
            goto LABEL_24;
          goto LABEL_11;
        }
        v15 = 8;
        if ( (unsigned int)(4 * v18) >= 0x18 )
        {
LABEL_24:
          v15 *= 2;
LABEL_25:
          sub_1E52FE0(a1, v15);
          sub_1E492D0(a1, v8, &v22);
          v17 = v22;
          v16 = *(_DWORD *)(a1 + 8);
          goto LABEL_12;
        }
LABEL_11:
        if ( v15 - *(_DWORD *)(a1 + 12) - v18 <= v15 >> 3 )
          goto LABEL_25;
LABEL_12:
        *(_DWORD *)(a1 + 8) = (2 * (v16 >> 1) + 2) | v16 & 1;
        if ( *v17 != -8 )
          --*(_DWORD *)(a1 + 12);
        *v17 = *v8;
        v19 = *(unsigned int *)(a1 + 88);
        if ( (unsigned int)v19 >= *(_DWORD *)(a1 + 92) )
        {
          sub_16CD150(a1 + 80, v7, 0, 8, (int)v17, a6);
          v19 = *(unsigned int *)(a1 + 88);
        }
        v20 = *v8++;
        *(_QWORD *)(*(_QWORD *)(a1 + 80) + 8 * v19) = v20;
        ++*(_DWORD *)(a1 + 88);
        if ( a3 == v8 )
          return;
      }
    }
  }
}
