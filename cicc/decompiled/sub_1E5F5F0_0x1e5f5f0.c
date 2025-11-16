// Function: sub_1E5F5F0
// Address: 0x1e5f5f0
//
__int64 __fastcall sub_1E5F5F0(__int64 a1, __int64 a2, __int64 a3)
{
  void *v5; // rdi
  _BYTE *v7; // r9
  _BYTE *v8; // r15
  __int64 v9; // rax
  size_t v10; // r8
  __int64 v11; // r14
  __int64 v12; // rax
  size_t v13; // r8
  __int64 v14; // rsi
  unsigned int v15; // edx
  __int64 v16; // rcx
  __int64 v17; // rdi
  __int64 *v18; // r14
  __int64 *i; // rbx
  const void *v20; // r15
  _QWORD *v21; // rcx
  __int64 v22; // rax
  _QWORD *v23; // rax
  __int64 v24; // rax
  unsigned __int64 v25; // rdx
  int v27; // ecx
  size_t v28; // [rsp+0h] [rbp-50h]
  size_t v29; // [rsp+8h] [rbp-48h]
  _BYTE *v30; // [rsp+8h] [rbp-48h]
  __int64 v31[7]; // [rsp+18h] [rbp-38h] BYREF

  v5 = (void *)(a1 + 16);
  v7 = *(_BYTE **)(a2 + 72);
  v8 = *(_BYTE **)(a2 + 64);
  *(_QWORD *)a1 = v5;
  *(_QWORD *)(a1 + 8) = 0x800000000LL;
  LODWORD(v9) = 0;
  v10 = v7 - v8;
  v11 = (v7 - v8) >> 3;
  if ( (unsigned __int64)(v7 - v8) > 0x40 )
  {
    v28 = v7 - v8;
    v30 = v7;
    sub_16CD150(a1, v5, (v7 - v8) >> 3, 8, v10, (int)v7);
    v9 = *(unsigned int *)(a1 + 8);
    v10 = v28;
    v7 = v30;
    v5 = (void *)(*(_QWORD *)a1 + 8 * v9);
  }
  if ( v7 != v8 )
  {
    memmove(v5, v8, v10);
    LODWORD(v9) = *(_DWORD *)(a1 + 8);
  }
  *(_DWORD *)(a1 + 8) = v9 + v11;
  if ( a3 )
  {
    v12 = *(unsigned int *)(a3 + 104);
    if ( (_DWORD)v12 )
    {
      LODWORD(v13) = v12 - 1;
      v14 = *(_QWORD *)(a3 + 88);
      v15 = (v12 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v16 = v14 + 56LL * v15;
      v17 = *(_QWORD *)v16;
      if ( a2 == *(_QWORD *)v16 )
      {
LABEL_8:
        if ( v16 != v14 + 56 * v12 )
        {
          v18 = *(__int64 **)(v16 + 8);
          for ( i = &v18[*(unsigned int *)(v16 + 16)]; i != v18; *(_DWORD *)(a1 + 8) = v16 )
          {
            while ( 1 )
            {
              v24 = *v18;
              v25 = *v18 & 0xFFFFFFFFFFFFFFF8LL;
              v31[0] = v25;
              if ( (v24 & 4) == 0 )
                break;
              ++v18;
              sub_1E05890(a1, v31, v25, v16, v13, (int)v7);
              if ( i == v18 )
                return a1;
            }
            v20 = (const void *)(*(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8));
            v21 = sub_1E06C90(*(_QWORD **)a1, (__int64)v20, v31);
            v22 = *(_QWORD *)a1;
            v13 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8) - (_QWORD)v20;
            if ( v20 != (const void *)(*(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8)) )
            {
              v29 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8) - (_QWORD)v20;
              v23 = memmove(v21, v20, v29);
              v13 = v29;
              v21 = v23;
              v22 = *(_QWORD *)a1;
            }
            ++v18;
            v16 = (__int64)((__int64)v21 + v13 - v22) >> 3;
          }
        }
      }
      else
      {
        v27 = 1;
        while ( v17 != -8 )
        {
          LODWORD(v7) = v27 + 1;
          v15 = v13 & (v27 + v15);
          v16 = v14 + 56LL * v15;
          v17 = *(_QWORD *)v16;
          if ( a2 == *(_QWORD *)v16 )
            goto LABEL_8;
          v27 = (int)v7;
        }
      }
    }
  }
  return a1;
}
