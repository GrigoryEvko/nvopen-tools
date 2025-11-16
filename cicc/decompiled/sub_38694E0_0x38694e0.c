// Function: sub_38694E0
// Address: 0x38694e0
//
__int64 __fastcall sub_38694E0(__int64 a1, __int64 a2, __m128i a3, __m128i a4)
{
  __int64 v6; // rdi
  unsigned int v7; // esi
  __int64 v8; // rcx
  unsigned int v9; // edx
  __int64 *v10; // rax
  __int64 v11; // r9
  __int64 v12; // r14
  int v14; // r11d
  __int64 *v15; // r13
  int v16; // eax
  int v17; // edx
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // r15
  __int64 v21; // rbx
  __int64 v22; // rax
  unsigned __int64 *v23; // rdi
  int v24; // eax
  int v25; // ecx
  __int64 v26; // rdi
  unsigned int v27; // eax
  __int64 v28; // rsi
  int v29; // r9d
  __int64 *v30; // r8
  int v31; // eax
  int v32; // eax
  __int64 v33; // rsi
  __int64 *v34; // rdi
  unsigned int v35; // r14d
  int v36; // r8d
  __int64 v37; // rcx
  __int64 *v38; // [rsp+8h] [rbp-48h]
  __int64 v39; // [rsp+10h] [rbp-40h]
  __int64 v40; // [rsp+18h] [rbp-38h]

  v6 = a1 + 160;
  v7 = *(_DWORD *)(a1 + 184);
  if ( !v7 )
  {
    ++*(_QWORD *)(a1 + 160);
    goto LABEL_19;
  }
  v8 = *(_QWORD *)(a1 + 168);
  v9 = (v7 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v10 = (__int64 *)(v8 + 16LL * v9);
  v11 = *v10;
  if ( *v10 != a2 )
  {
    v14 = 1;
    v15 = 0;
    while ( v11 != -8 )
    {
      if ( v11 == -16 && !v15 )
        v15 = v10;
      v9 = (v7 - 1) & (v14 + v9);
      v10 = (__int64 *)(v8 + 16LL * v9);
      v11 = *v10;
      if ( *v10 == a2 )
        goto LABEL_3;
      ++v14;
    }
    if ( !v15 )
      v15 = v10;
    v16 = *(_DWORD *)(a1 + 176);
    ++*(_QWORD *)(a1 + 160);
    v17 = v16 + 1;
    if ( 4 * (v16 + 1) < 3 * v7 )
    {
      if ( v7 - *(_DWORD *)(a1 + 180) - v17 > v7 >> 3 )
      {
LABEL_11:
        *(_DWORD *)(a1 + 176) = v17;
        if ( *v15 != -8 )
          --*(_DWORD *)(a1 + 180);
        *v15 = a2;
        v15[1] = 0;
        goto LABEL_14;
      }
      sub_3866BA0(v6, v7);
      v31 = *(_DWORD *)(a1 + 184);
      if ( v31 )
      {
        v32 = v31 - 1;
        v33 = *(_QWORD *)(a1 + 168);
        v34 = 0;
        v35 = v32 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v36 = 1;
        v17 = *(_DWORD *)(a1 + 176) + 1;
        v15 = (__int64 *)(v33 + 16LL * v35);
        v37 = *v15;
        if ( *v15 != a2 )
        {
          while ( v37 != -8 )
          {
            if ( v37 == -16 && !v34 )
              v34 = v15;
            v35 = v32 & (v36 + v35);
            v15 = (__int64 *)(v33 + 16LL * v35);
            v37 = *v15;
            if ( *v15 == a2 )
              goto LABEL_11;
            ++v36;
          }
          if ( v34 )
            v15 = v34;
        }
        goto LABEL_11;
      }
LABEL_48:
      ++*(_DWORD *)(a1 + 176);
      BUG();
    }
LABEL_19:
    sub_3866BA0(v6, 2 * v7);
    v24 = *(_DWORD *)(a1 + 184);
    if ( v24 )
    {
      v25 = v24 - 1;
      v26 = *(_QWORD *)(a1 + 168);
      v27 = (v24 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v17 = *(_DWORD *)(a1 + 176) + 1;
      v15 = (__int64 *)(v26 + 16LL * v27);
      v28 = *v15;
      if ( *v15 != a2 )
      {
        v29 = 1;
        v30 = 0;
        while ( v28 != -8 )
        {
          if ( !v30 && v28 == -16 )
            v30 = v15;
          v27 = v25 & (v29 + v27);
          v15 = (__int64 *)(v26 + 16LL * v27);
          v28 = *v15;
          if ( *v15 == a2 )
            goto LABEL_11;
          ++v29;
        }
        if ( v30 )
          v15 = v30;
      }
      goto LABEL_11;
    }
    goto LABEL_48;
  }
LABEL_3:
  v12 = v10[1];
  if ( v12 )
    return v12;
  v15 = v10;
LABEL_14:
  v18 = *(_QWORD *)(a1 + 208);
  v19 = *(_QWORD *)(a1 + 216);
  v20 = *(_QWORD *)(a1 + 192);
  v38 = *(__int64 **)(a1 + 200);
  v21 = *(_QWORD *)(a1 + 224);
  v39 = v18;
  v40 = v19;
  v22 = sub_22077B0(0xC8u);
  v12 = v22;
  if ( v22 )
    sub_38692D0(v22, a2, v20, v38, v39, v40, a3, a4, v21);
  v23 = (unsigned __int64 *)v15[1];
  v15[1] = v12;
  if ( v23 )
  {
    sub_385CEA0(v23);
    return v15[1];
  }
  return v12;
}
