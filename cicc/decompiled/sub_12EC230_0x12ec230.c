// Function: sub_12EC230
// Address: 0x12ec230
//
__int64 __fastcall sub_12EC230(__int64 a1, __int64 a2)
{
  int *v2; // r12
  int *v3; // r14
  __int64 result; // rax
  __int64 v7; // r9
  unsigned int v8; // r8d
  int *v9; // rax
  int v10; // edi
  __int64 v11; // rdx
  unsigned int v12; // esi
  int v13; // r15d
  int v14; // ecx
  int v15; // ecx
  __int64 v16; // r8
  unsigned int v17; // esi
  int v18; // eax
  int *v19; // rdx
  int v20; // edi
  int v21; // r11d
  int v22; // eax
  int v23; // esi
  int v24; // esi
  __int64 v25; // r8
  int *v26; // r9
  int v27; // r10d
  unsigned int v28; // ecx
  int v29; // edi
  int v30; // edx
  int *v31; // r11
  int v32; // r10d
  int v33; // [rsp+0h] [rbp-40h]
  __int64 v34; // [rsp+8h] [rbp-38h]

  v2 = *(int **)(a1 + 8);
  v3 = &v2[*(unsigned int *)(a1 + 16)];
  result = a1 + 1048;
  v34 = a1 + 1048;
  while ( v3 != v2 )
  {
    v12 = *(_DWORD *)(a1 + 1072);
    v13 = *v2;
    if ( v12 )
    {
      v7 = *(_QWORD *)(a1 + 1056);
      v8 = (v12 - 1) & (37 * v13);
      v9 = (int *)(v7 + 16LL * v8);
      v10 = *v9;
      if ( v13 == *v9 )
      {
        v11 = *((_QWORD *)v9 + 1);
        goto LABEL_5;
      }
      v21 = 1;
      v19 = 0;
      while ( v10 != 0x7FFFFFFF )
      {
        if ( v19 || v10 != 0x80000000 )
          v9 = v19;
        v30 = v21 + 1;
        v8 = (v12 - 1) & (v21 + v8);
        v31 = (int *)(v7 + 16LL * v8);
        v10 = *v31;
        if ( v13 == *v31 )
        {
          v11 = *((_QWORD *)v31 + 1);
          goto LABEL_5;
        }
        v21 = v30;
        v19 = v9;
        v9 = (int *)(v7 + 16LL * v8);
      }
      if ( !v19 )
        v19 = v9;
      v22 = *(_DWORD *)(a1 + 1064);
      ++*(_QWORD *)(a1 + 1048);
      v18 = v22 + 1;
      if ( 4 * v18 < 3 * v12 )
      {
        if ( v12 - *(_DWORD *)(a1 + 1068) - v18 > v12 >> 3 )
          goto LABEL_10;
        v33 = 37 * v13;
        sub_12EABE0(v34, v12);
        v23 = *(_DWORD *)(a1 + 1072);
        if ( !v23 )
        {
LABEL_44:
          ++*(_DWORD *)(a1 + 1064);
          BUG();
        }
        v24 = v23 - 1;
        v25 = *(_QWORD *)(a1 + 1056);
        v26 = 0;
        v27 = 1;
        v28 = v24 & v33;
        v18 = *(_DWORD *)(a1 + 1064) + 1;
        v19 = (int *)(v25 + 16LL * (v24 & (unsigned int)v33));
        v29 = *v19;
        if ( v13 == *v19 )
          goto LABEL_10;
        while ( v29 != 0x7FFFFFFF )
        {
          if ( !v26 && v29 == 0x80000000 )
            v26 = v19;
          v28 = v24 & (v27 + v28);
          v19 = (int *)(v25 + 16LL * v28);
          v29 = *v19;
          if ( v13 == *v19 )
            goto LABEL_10;
          ++v27;
        }
        goto LABEL_23;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 1048);
    }
    sub_12EABE0(v34, 2 * v12);
    v14 = *(_DWORD *)(a1 + 1072);
    if ( !v14 )
      goto LABEL_44;
    v15 = v14 - 1;
    v16 = *(_QWORD *)(a1 + 1056);
    v17 = v15 & (37 * v13);
    v18 = *(_DWORD *)(a1 + 1064) + 1;
    v19 = (int *)(v16 + 16LL * v17);
    v20 = *v19;
    if ( v13 == *v19 )
      goto LABEL_10;
    v32 = 1;
    v26 = 0;
    while ( v20 != 0x7FFFFFFF )
    {
      if ( !v26 && v20 == 0x80000000 )
        v26 = v19;
      v17 = v15 & (v32 + v17);
      v19 = (int *)(v16 + 16LL * v17);
      v20 = *v19;
      if ( v13 == *v19 )
        goto LABEL_10;
      ++v32;
    }
LABEL_23:
    if ( v26 )
      v19 = v26;
LABEL_10:
    *(_DWORD *)(a1 + 1064) = v18;
    if ( *v19 != 0x7FFFFFFF )
      --*(_DWORD *)(a1 + 1068);
    *v19 = v13;
    *((_QWORD *)v19 + 1) = 0;
    v11 = 0;
LABEL_5:
    result = sub_1C278E0(a1, a2, v11, byte_3F871B3, 0);
    ++v2;
  }
  return result;
}
