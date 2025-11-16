// Function: sub_1F216F0
// Address: 0x1f216f0
//
__int64 __fastcall sub_1F216F0(__int64 a1, __int64 a2, __int64 a3, _BYTE *a4, __int64 a5, int a6)
{
  int v8; // edx
  int v9; // eax
  unsigned int v10; // r8d
  __int64 v11; // r14
  __int64 v12; // rax
  __int64 v13; // rdx
  char v15; // al
  __int64 v16; // rdx
  __int64 v17; // r8
  __int64 v18; // rbx
  int v19; // r14d
  __int64 v20; // rdx
  char v21; // r10
  __int64 v22; // rax
  __int64 v23; // [rsp+0h] [rbp-40h]
  char v24; // [rsp+0h] [rbp-40h]
  unsigned int v25; // [rsp+8h] [rbp-38h]
  __int64 v26; // [rsp+8h] [rbp-38h]

  v8 = **(unsigned __int16 **)(a2 + 16);
  if ( (unsigned int)(v8 - 17) > 1 )
  {
    if ( !byte_4FCA8E0 )
      return 0;
    v15 = byte_4FCA9C0 | ((unsigned __int16)(v8 - 12) <= 1u);
    if ( v15 )
      return 0;
    v16 = *(_QWORD *)(a2 + 32);
    v17 = v16 + 40LL * *(unsigned int *)(a2 + 40);
    if ( v16 == v17 )
      return 0;
    v18 = *(_QWORD *)(a2 + 32);
    do
    {
      if ( *(_BYTE *)v18 == 5 )
      {
        v19 = *(_DWORD *)(v18 + 24);
        if ( v19 >= 0 )
        {
          v20 = (unsigned int)v19 >> 6;
          if ( (*(_QWORD *)(*(_QWORD *)(a1 + 1512) + 8 * v20) & (1LL << v19)) != 0 )
          {
            v21 = byte_4FCA8E0;
            if ( byte_4FCA8E0 )
            {
              if ( !byte_4FCA9C0 && (*(_QWORD *)(*(_QWORD *)(a1 + 1536) + 8 * v20) & (1LL << v19)) == 0 )
              {
                v22 = *(unsigned int *)(a3 + 8);
                if ( (unsigned int)v22 >= *(_DWORD *)(a3 + 12) )
                {
                  v24 = byte_4FCA8E0;
                  v26 = v17;
                  sub_16CD150(a3, (const void *)(a3 + 16), 0, 4, v17, 1);
                  v22 = *(unsigned int *)(a3 + 8);
                  v21 = v24;
                  v17 = v26;
                }
                *(_DWORD *)(*(_QWORD *)a3 + 4 * v22) = v19;
                v15 = v21;
                ++*(_DWORD *)(a3 + 8);
              }
            }
          }
        }
      }
      v18 += 40;
    }
    while ( v17 != v18 );
    if ( !v15 )
      return 0;
LABEL_26:
    *a4 = 1;
    return 1;
  }
  v9 = *(_DWORD *)(*(_QWORD *)(a2 + 32) + 24LL);
  if ( v9 < 0 )
    return 0;
  v10 = *(_DWORD *)(*(_QWORD *)(a2 + 32) + 24LL);
  v11 = 1LL << v9;
  v12 = (unsigned int)v9 >> 6;
  if ( (*(_QWORD *)(*(_QWORD *)(a1 + 1512) + 8 * v12) & (1LL << v10)) == 0 )
    return 0;
  v13 = *(unsigned int *)(a3 + 8);
  if ( (unsigned int)v13 >= *(_DWORD *)(a3 + 12) )
  {
    v23 = v10 >> 6;
    v25 = *(_DWORD *)(*(_QWORD *)(a2 + 32) + 24LL);
    sub_16CD150(a3, (const void *)(a3 + 16), 0, 4, v10, a6);
    v13 = *(unsigned int *)(a3 + 8);
    v12 = v23;
    v10 = v25;
  }
  *(_DWORD *)(*(_QWORD *)a3 + 4 * v13) = v10;
  ++*(_DWORD *)(a3 + 8);
  if ( **(_WORD **)(a2 + 16) != 18 )
  {
    if ( byte_4FCA8E0 && !byte_4FCA9C0 && (*(_QWORD *)(*(_QWORD *)(a1 + 1536) + 8 * v12) & v11) == 0 )
      return 0;
    goto LABEL_26;
  }
  *a4 = 0;
  return 1;
}
