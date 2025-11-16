// Function: sub_16345B0
// Address: 0x16345b0
//
__int64 __fastcall sub_16345B0(__int64 a1, const void *a2, size_t a3, __int64 a4)
{
  __int64 result; // rax
  _QWORD *v5; // r15
  __int64 *v6; // rbx
  __int64 *v7; // r14
  __int64 v8; // r11
  size_t v9; // rdx
  __int64 v10; // r15
  __int64 v11; // r12
  int v12; // r13d
  int v13; // eax
  unsigned int v14; // esi
  __int64 v15; // r8
  unsigned int v16; // edi
  _QWORD *v17; // rax
  __int64 v18; // rcx
  int v19; // r10d
  _QWORD *v20; // r11
  int v21; // ecx
  int v22; // esi
  int v23; // esi
  __int64 v24; // r9
  unsigned int v25; // edi
  __int64 v26; // r8
  _QWORD *v27; // r11
  int v28; // eax
  int v29; // r11d
  __int64 v30; // r9
  _QWORD *v31; // r8
  unsigned int v32; // esi
  __int64 v33; // rdi
  __int64 v36; // [rsp+18h] [rbp-58h]
  unsigned int v37; // [rsp+24h] [rbp-4Ch]
  size_t v38; // [rsp+28h] [rbp-48h]
  size_t v39; // [rsp+28h] [rbp-48h]
  size_t v40; // [rsp+28h] [rbp-48h]
  _QWORD *v41; // [rsp+30h] [rbp-40h]

  result = a1 + 8;
  v36 = a1 + 8;
  if ( a1 + 8 == *(_QWORD *)(a1 + 24) )
    return result;
  v5 = *(_QWORD **)(a1 + 24);
  do
  {
    v6 = (__int64 *)v5[7];
    v7 = (__int64 *)v5[8];
    v8 = v5[4];
    if ( v7 == v6 )
      goto LABEL_15;
    v41 = v5;
    v9 = a3;
    v10 = v5[4];
    v37 = 37 * v8;
    do
    {
      while ( 1 )
      {
        v11 = *v6;
        if ( *v6 )
        {
          v12 = *(_DWORD *)(v11 + 8);
          if ( v12 == 1 && *(_QWORD *)(v11 + 32) == v9 )
          {
            if ( !v9 )
              break;
            v38 = v9;
            v13 = memcmp(*(const void **)(v11 + 24), a2, v9);
            v9 = v38;
            if ( !v13 )
              break;
          }
        }
        if ( v7 == ++v6 )
          goto LABEL_14;
      }
      v14 = *(_DWORD *)(a4 + 24);
      if ( v14 )
      {
        v15 = *(_QWORD *)(a4 + 8);
        v16 = (v14 - 1) & v37;
        v17 = (_QWORD *)(v15 + 16LL * v16);
        v18 = *v17;
        if ( v10 == *v17 )
          goto LABEL_13;
        v19 = 1;
        v20 = 0;
        while ( v18 != -1 )
        {
          if ( !v20 && v18 == -2 )
            v20 = v17;
          v16 = (v14 - 1) & (v19 + v16);
          v17 = (_QWORD *)(v15 + 16LL * v16);
          v18 = *v17;
          if ( v10 == *v17 )
            goto LABEL_13;
          ++v19;
        }
        if ( v20 )
          v17 = v20;
        ++*(_QWORD *)a4;
        v21 = *(_DWORD *)(a4 + 16) + 1;
        if ( 4 * v21 < 3 * v14 )
        {
          if ( v14 - *(_DWORD *)(a4 + 20) - v21 <= v14 >> 3 )
          {
            v40 = v9;
            sub_16343F0(a4, v14);
            v28 = *(_DWORD *)(a4 + 24);
            if ( !v28 )
            {
LABEL_55:
              ++*(_DWORD *)(a4 + 16);
              BUG();
            }
            v29 = v28 - 1;
            v30 = *(_QWORD *)(a4 + 8);
            v31 = 0;
            v9 = v40;
            v32 = (v28 - 1) & v37;
            v21 = *(_DWORD *)(a4 + 16) + 1;
            v17 = (_QWORD *)(v30 + 16LL * v32);
            v33 = *v17;
            if ( v10 != *v17 )
            {
              while ( v33 != -1 )
              {
                if ( !v31 && v33 == -2 )
                  v31 = v17;
                v32 = v29 & (v12 + v32);
                v17 = (_QWORD *)(v30 + 16LL * v32);
                v33 = *v17;
                if ( v10 == *v17 )
                  goto LABEL_23;
                ++v12;
              }
              if ( v31 )
                v17 = v31;
            }
          }
          goto LABEL_23;
        }
      }
      else
      {
        ++*(_QWORD *)a4;
      }
      v39 = v9;
      sub_16343F0(a4, 2 * v14);
      v22 = *(_DWORD *)(a4 + 24);
      if ( !v22 )
        goto LABEL_55;
      v23 = v22 - 1;
      v24 = *(_QWORD *)(a4 + 8);
      v9 = v39;
      v25 = v23 & v37;
      v21 = *(_DWORD *)(a4 + 16) + 1;
      v17 = (_QWORD *)(v24 + 16LL * (v23 & v37));
      v26 = *v17;
      if ( v10 != *v17 )
      {
        v27 = 0;
        while ( v26 != -1 )
        {
          if ( !v27 && v26 == -2 )
            v27 = v17;
          v25 = v23 & (v12 + v25);
          v17 = (_QWORD *)(v24 + 16LL * v25);
          v26 = *v17;
          if ( v10 == *v17 )
            goto LABEL_23;
          ++v12;
        }
        if ( v27 )
          v17 = v27;
      }
LABEL_23:
      *(_DWORD *)(a4 + 16) = v21;
      if ( *v17 != -1 )
        --*(_DWORD *)(a4 + 20);
      *v17 = v10;
      v17[1] = 0;
LABEL_13:
      ++v6;
      v17[1] = v11;
    }
    while ( v7 != v6 );
LABEL_14:
    v5 = v41;
LABEL_15:
    result = sub_220EF30(v5);
    v5 = (_QWORD *)result;
  }
  while ( v36 != result );
  return result;
}
