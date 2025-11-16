// Function: sub_1AC6520
// Address: 0x1ac6520
//
__int64 __fastcall sub_1AC6520(_QWORD *a1, _QWORD *a2, __int64 a3, __int64 a4)
{
  unsigned int v6; // r14d
  __int64 v7; // rax
  unsigned __int64 v8; // r14
  __int64 *v9; // r15
  int v10; // r8d
  int v11; // r9d
  __int64 v12; // r12
  __int64 v13; // rax
  __int64 *v14; // rdi
  __int64 v15; // rdx
  __int64 v16; // rsi
  __int64 v17; // rax
  int v18; // ecx
  __int64 v19; // r10
  int v20; // ecx
  unsigned int v21; // r9d
  __int64 **v22; // rax
  __int64 *v23; // r12
  int v25; // eax
  int v26; // r11d
  __int64 v27; // [rsp-40h] [rbp-40h]
  __int64 *v28; // [rsp-40h] [rbp-40h]

  if ( !a3 )
    return 0;
  v27 = *(_QWORD *)(a3 + 24);
  v6 = *(_DWORD *)(v27 + 12) - 1;
  if ( v6 > (unsigned int)sub_14DA610(a2) )
    return 0;
  v7 = *(_QWORD *)(v27 + 16);
  v8 = (*a2 & 0xFFFFFFFFFFFFFFF8LL) - 24LL * (*(_DWORD *)((*a2 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF);
  v9 = (__int64 *)(v7 + 8);
  v28 = (__int64 *)(v7 + 8LL * *(unsigned int *)(v27 + 12));
  if ( (__int64 *)(v7 + 8) != v28 )
  {
    do
    {
      v14 = *(__int64 **)v8;
      v15 = a1[80];
      v16 = *v9;
      if ( *(_BYTE *)(*(_QWORD *)v8 + 16LL) > 0x10u )
      {
        v17 = a1[6];
        if ( v17 == a1[7] )
          v17 = *(_QWORD *)(a1[9] - 8LL) + 512LL;
        v18 = *(_DWORD *)(v17 - 8);
        if ( v18 )
        {
          v19 = *(_QWORD *)(v17 - 24);
          v20 = v18 - 1;
          v21 = v20 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
          v22 = (__int64 **)(v19 + 16LL * v21);
          v23 = *v22;
          if ( v14 == *v22 )
          {
LABEL_14:
            v12 = sub_14D66F0(v22[1], v16, v15);
            if ( !v12 )
              return 0;
            goto LABEL_6;
          }
          v25 = 1;
          while ( v23 != (__int64 *)-8LL )
          {
            v26 = v25 + 1;
            v21 = v20 & (v25 + v21);
            v22 = (__int64 **)(v19 + 16LL * v21);
            v23 = *v22;
            if ( v14 == *v22 )
              goto LABEL_14;
            v25 = v26;
          }
        }
        v14 = 0;
      }
      v12 = sub_14D66F0(v14, v16, v15);
      if ( !v12 )
        return 0;
LABEL_6:
      v13 = *(unsigned int *)(a4 + 8);
      if ( (unsigned int)v13 >= *(_DWORD *)(a4 + 12) )
      {
        sub_16CD150(a4, (const void *)(a4 + 16), 0, 8, v10, v11);
        v13 = *(unsigned int *)(a4 + 8);
      }
      v8 += 24LL;
      ++v9;
      *(_QWORD *)(*(_QWORD *)a4 + 8 * v13) = v12;
      ++*(_DWORD *)(a4 + 8);
    }
    while ( v28 != v9 );
  }
  return 1;
}
