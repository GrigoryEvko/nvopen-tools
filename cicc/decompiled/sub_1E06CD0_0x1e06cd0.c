// Function: sub_1E06CD0
// Address: 0x1e06cd0
//
__int64 __fastcall sub_1E06CD0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 v9; // r8
  char *v10; // rsi
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // r13
  __int64 v14; // rbx
  __int64 v15; // rax
  size_t v16; // r8
  __int64 v17; // rsi
  unsigned int v18; // edx
  __int64 v19; // rcx
  __int64 v20; // rdi
  __int64 *v21; // r14
  __int64 *i; // rbx
  const void *v23; // r15
  _QWORD *v24; // rcx
  __int64 v25; // rax
  _QWORD *v26; // rax
  __int64 v27; // rax
  unsigned __int64 v28; // rdx
  int v30; // ecx
  __int64 v31; // [rsp+0h] [rbp-50h]
  size_t v32; // [rsp+8h] [rbp-48h]
  __int64 v33; // [rsp+8h] [rbp-48h]
  __int64 v34[7]; // [rsp+18h] [rbp-38h] BYREF

  v9 = *(_QWORD *)(a2 + 96);
  v12 = *(_QWORD *)(a2 + 88);
  *(_QWORD *)(a1 + 8) = 0x800000000LL;
  v10 = (char *)(a1 + 16);
  *(_QWORD *)a1 = a1 + 16;
  v11 = v9 - v12;
  LODWORD(v12) = 0;
  v13 = v11 >> 3;
  v14 = v11 >> 3;
  if ( (unsigned __int64)v11 > 0x40 )
  {
    v31 = v11;
    v33 = v9;
    sub_16CD150(a1, v10, v11 >> 3, 8, v9, a6);
    v12 = *(unsigned int *)(a1 + 8);
    v11 = v31;
    v9 = v33;
    v10 = (char *)(*(_QWORD *)a1 + 8 * v12);
  }
  if ( v11 > 0 )
  {
    do
    {
      v10 += 8;
      *((_QWORD *)v10 - 1) = *(_QWORD *)(v9 - 8 * v13 + 8 * v14-- - 8);
    }
    while ( v14 );
    LODWORD(v12) = *(_DWORD *)(a1 + 8);
  }
  *(_DWORD *)(a1 + 8) = v12 + v13;
  if ( a3 )
  {
    v15 = *(unsigned int *)(a3 + 104);
    if ( (_DWORD)v15 )
    {
      LODWORD(v16) = v15 - 1;
      v17 = *(_QWORD *)(a3 + 88);
      v18 = (v15 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v19 = v17 + 56LL * v18;
      v20 = *(_QWORD *)v19;
      if ( a2 == *(_QWORD *)v19 )
      {
LABEL_9:
        if ( v19 != v17 + 56 * v15 )
        {
          v21 = *(__int64 **)(v19 + 8);
          for ( i = &v21[*(unsigned int *)(v19 + 16)]; i != v21; *(_DWORD *)(a1 + 8) = v19 )
          {
            while ( 1 )
            {
              v27 = *v21;
              v28 = *v21 & 0xFFFFFFFFFFFFFFF8LL;
              v34[0] = v28;
              if ( (v27 & 4) == 0 )
                break;
              ++v21;
              sub_1E05890(a1, v34, v28, v19, v16, a6);
              if ( i == v21 )
                return a1;
            }
            v23 = (const void *)(*(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8));
            v24 = sub_1E06C90(*(_QWORD **)a1, (__int64)v23, v34);
            v25 = *(_QWORD *)a1;
            v16 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8) - (_QWORD)v23;
            if ( v23 != (const void *)(*(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8)) )
            {
              v32 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8) - (_QWORD)v23;
              v26 = memmove(v24, v23, v32);
              v16 = v32;
              v24 = v26;
              v25 = *(_QWORD *)a1;
            }
            ++v21;
            v19 = (__int64)((__int64)v24 + v16 - v25) >> 3;
          }
        }
      }
      else
      {
        v30 = 1;
        while ( v20 != -8 )
        {
          a6 = v30 + 1;
          v18 = v16 & (v30 + v18);
          v19 = v17 + 56LL * v18;
          v20 = *(_QWORD *)v19;
          if ( a2 == *(_QWORD *)v19 )
            goto LABEL_9;
          v30 = a6;
        }
      }
    }
  }
  return a1;
}
