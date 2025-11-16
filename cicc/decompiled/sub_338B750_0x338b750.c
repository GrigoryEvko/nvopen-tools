// Function: sub_338B750
// Address: 0x338b750
//
__int64 __fastcall sub_338B750(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  unsigned int v4; // r8d
  __int64 v5; // r10
  __int64 v6; // r9
  int v7; // ebx
  __int64 *v8; // rcx
  unsigned int v9; // eax
  __int64 *v10; // rdx
  __int64 v11; // rdi
  __int64 result; // rax
  int v13; // eax
  int v14; // edx
  __int64 v15; // rbx
  __int64 v16; // rdx
  __int64 v17; // r14
  _QWORD *v18; // rax
  __int64 v19; // rsi
  int v20; // eax
  int v21; // esi
  __int64 v22; // r8
  unsigned int v23; // eax
  __int64 v24; // rdi
  int v25; // r11d
  __int64 *v26; // r10
  int v27; // eax
  int v28; // esi
  __int64 v29; // r8
  __int64 *v30; // r10
  int v31; // r11d
  unsigned int v32; // eax
  __int64 v33[5]; // [rsp+18h] [rbp-28h] BYREF

  v2 = a1 + 8;
  v4 = *(_DWORD *)(a1 + 32);
  v33[0] = a2;
  if ( !v4 )
  {
    ++*(_QWORD *)(a1 + 8);
    goto LABEL_22;
  }
  v5 = *(_QWORD *)(a1 + 16);
  v6 = a2;
  v7 = 1;
  v8 = 0;
  v9 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v10 = (__int64 *)(v5 + 24LL * v9);
  v11 = *v10;
  if ( a2 != *v10 )
  {
    while ( v11 != -4096 )
    {
      if ( !v8 && v11 == -8192 )
        v8 = v10;
      v9 = (v4 - 1) & (v7 + v9);
      v10 = (__int64 *)(v5 + 24LL * v9);
      v11 = *v10;
      if ( a2 == *v10 )
        goto LABEL_3;
      ++v7;
    }
    v13 = *(_DWORD *)(a1 + 24);
    if ( !v8 )
      v8 = v10;
    ++*(_QWORD *)(a1 + 8);
    v14 = v13 + 1;
    if ( 4 * (v13 + 1) < 3 * v4 )
    {
      if ( v4 - *(_DWORD *)(a1 + 28) - v14 > v4 >> 3 )
      {
LABEL_16:
        *(_DWORD *)(a1 + 24) = v14;
        if ( *v8 != -4096 )
          --*(_DWORD *)(a1 + 28);
        v8[1] = 0;
        *((_DWORD *)v8 + 4) = 0;
        *v8 = v6;
        a2 = v33[0];
        goto LABEL_19;
      }
      sub_337DA20(v2, v4);
      v27 = *(_DWORD *)(a1 + 32);
      if ( v27 )
      {
        v28 = v27 - 1;
        v29 = *(_QWORD *)(a1 + 16);
        v30 = 0;
        v31 = 1;
        v32 = (v27 - 1) & ((LODWORD(v33[0]) >> 9) ^ (LODWORD(v33[0]) >> 4));
        v8 = (__int64 *)(v29 + 24LL * v32);
        v14 = *(_DWORD *)(a1 + 24) + 1;
        v6 = *v8;
        if ( v33[0] != *v8 )
        {
          while ( v6 != -4096 )
          {
            if ( !v30 && v6 == -8192 )
              v30 = v8;
            v32 = v28 & (v31 + v32);
            v8 = (__int64 *)(v29 + 24LL * v32);
            v6 = *v8;
            if ( v33[0] == *v8 )
              goto LABEL_16;
            ++v31;
          }
          v6 = v33[0];
          if ( v30 )
            v8 = v30;
        }
        goto LABEL_16;
      }
LABEL_45:
      ++*(_DWORD *)(a1 + 24);
      BUG();
    }
LABEL_22:
    sub_337DA20(v2, 2 * v4);
    v20 = *(_DWORD *)(a1 + 32);
    if ( v20 )
    {
      v6 = v33[0];
      v21 = v20 - 1;
      v22 = *(_QWORD *)(a1 + 16);
      v23 = (v20 - 1) & ((LODWORD(v33[0]) >> 9) ^ (LODWORD(v33[0]) >> 4));
      v8 = (__int64 *)(v22 + 24LL * v23);
      v14 = *(_DWORD *)(a1 + 24) + 1;
      v24 = *v8;
      if ( *v8 != v33[0] )
      {
        v25 = 1;
        v26 = 0;
        while ( v24 != -4096 )
        {
          if ( !v26 && v24 == -8192 )
            v26 = v8;
          v23 = v21 & (v25 + v23);
          v8 = (__int64 *)(v22 + 24LL * v23);
          v24 = *v8;
          if ( v33[0] == *v8 )
            goto LABEL_16;
          ++v25;
        }
        if ( v26 )
          v8 = v26;
      }
      goto LABEL_16;
    }
    goto LABEL_45;
  }
LABEL_3:
  if ( v10[1] )
    return v10[1];
LABEL_19:
  result = sub_3380740(a1, (int *)a2, *(_QWORD *)(a2 + 8));
  if ( !result )
  {
    v15 = sub_3389ED0(a1, v33[0]);
    v17 = v16;
    v18 = sub_337DC20(v2, v33);
    *v18 = v15;
    v19 = v33[0];
    *((_DWORD *)v18 + 2) = v17;
    sub_3380540(a1, v19, v15, v17);
    return v15;
  }
  return result;
}
