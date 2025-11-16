// Function: sub_1BEC0C0
// Address: 0x1bec0c0
//
__int64 __fastcall sub_1BEC0C0(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // rsi
  unsigned int v6; // ecx
  __int64 *v7; // rdx
  __int64 v8; // r8
  __int64 v9; // r12
  int v11; // edx
  __int64 v12; // rdx
  _BYTE *v13; // rsi
  __int64 v14; // rdx
  __int64 *v15; // rdi
  unsigned int v16; // esi
  __int64 v17; // rdi
  __int64 v18; // r8
  unsigned int v19; // ecx
  __int64 *v20; // rax
  __int64 v21; // rdx
  int v22; // r9d
  int v23; // r11d
  __int64 *v24; // r10
  int v25; // ecx
  int v26; // ecx
  int v27; // eax
  int v28; // esi
  __int64 v29; // r8
  unsigned int v30; // edx
  __int64 v31; // rdi
  int v32; // r10d
  __int64 *v33; // r9
  int v34; // eax
  int v35; // edx
  __int64 v36; // rdi
  int v37; // r9d
  unsigned int v38; // r14d
  __int64 *v39; // r8
  __int64 v40; // rsi
  _QWORD v41[2]; // [rsp+0h] [rbp-80h] BYREF
  _QWORD *v42; // [rsp+10h] [rbp-70h] BYREF
  __int16 v43; // [rsp+20h] [rbp-60h]
  __int64 v44[2]; // [rsp+30h] [rbp-50h] BYREF
  __int64 v45; // [rsp+40h] [rbp-40h] BYREF

  v4 = *(unsigned int *)(a1 + 72);
  if ( !(_DWORD)v4 )
    goto LABEL_8;
  v5 = *(_QWORD *)(a1 + 56);
  v6 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v7 = (__int64 *)(v5 + 16LL * v6);
  v8 = *v7;
  if ( *v7 != a2 )
  {
    v11 = 1;
    while ( v8 != -8 )
    {
      v22 = v11 + 1;
      v6 = (v4 - 1) & (v11 + v6);
      v7 = (__int64 *)(v5 + 16LL * v6);
      v8 = *v7;
      if ( *v7 == a2 )
        goto LABEL_3;
      v11 = v22;
    }
LABEL_8:
    v41[0] = sub_1649960(a2);
    v43 = 261;
    v41[1] = v12;
    v42 = v41;
    v9 = sub_22077B0(128);
    if ( v9 )
    {
      sub_16E2FC0(v44, (__int64)&v42);
      v13 = (_BYTE *)v44[0];
      *(_BYTE *)(v9 + 8) = 0;
      v14 = v44[1];
      *(_QWORD *)v9 = &unk_49F6D50;
      *(_QWORD *)(v9 + 16) = v9 + 32;
      sub_1BEAFD0((__int64 *)(v9 + 16), v13, (__int64)&v13[v14]);
      v15 = (__int64 *)v44[0];
      *(_QWORD *)(v9 + 56) = v9 + 72;
      *(_QWORD *)(v9 + 64) = 0x100000000LL;
      *(_QWORD *)(v9 + 88) = 0x100000000LL;
      *(_QWORD *)(v9 + 48) = 0;
      *(_QWORD *)(v9 + 80) = v9 + 96;
      *(_QWORD *)(v9 + 104) = 0;
      if ( v15 != &v45 )
        j_j___libc_free_0(v15, v45 + 1);
      *(_QWORD *)v9 = &unk_49F7110;
      *(_QWORD *)(v9 + 120) = v9 + 112;
      *(_QWORD *)(v9 + 112) = (v9 + 112) | 4;
    }
    v16 = *(_DWORD *)(a1 + 72);
    v17 = a1 + 48;
    if ( v16 )
    {
      v18 = *(_QWORD *)(a1 + 56);
      v19 = (v16 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v20 = (__int64 *)(v18 + 16LL * v19);
      v21 = *v20;
      if ( *v20 == a2 )
      {
LABEL_14:
        v20[1] = v9;
        *(_QWORD *)(v9 + 48) = *(_QWORD *)(a1 + 24);
        return v9;
      }
      v23 = 1;
      v24 = 0;
      while ( v21 != -8 )
      {
        if ( !v24 && v21 == -16 )
          v24 = v20;
        v19 = (v16 - 1) & (v23 + v19);
        v20 = (__int64 *)(v18 + 16LL * v19);
        v21 = *v20;
        if ( *v20 == a2 )
          goto LABEL_14;
        ++v23;
      }
      v25 = *(_DWORD *)(a1 + 64);
      if ( v24 )
        v20 = v24;
      ++*(_QWORD *)(a1 + 48);
      v26 = v25 + 1;
      if ( 4 * v26 < 3 * v16 )
      {
        if ( v16 - *(_DWORD *)(a1 + 68) - v26 > v16 >> 3 )
        {
LABEL_23:
          *(_DWORD *)(a1 + 64) = v26;
          if ( *v20 != -8 )
            --*(_DWORD *)(a1 + 68);
          *v20 = a2;
          v20[1] = 0;
          goto LABEL_14;
        }
        sub_1BEBF00(v17, v16);
        v34 = *(_DWORD *)(a1 + 72);
        if ( v34 )
        {
          v35 = v34 - 1;
          v36 = *(_QWORD *)(a1 + 56);
          v37 = 1;
          v38 = (v34 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
          v39 = 0;
          v26 = *(_DWORD *)(a1 + 64) + 1;
          v20 = (__int64 *)(v36 + 16LL * v38);
          v40 = *v20;
          if ( *v20 != a2 )
          {
            while ( v40 != -8 )
            {
              if ( !v39 && v40 == -16 )
                v39 = v20;
              v38 = v35 & (v37 + v38);
              v20 = (__int64 *)(v36 + 16LL * v38);
              v40 = *v20;
              if ( *v20 == a2 )
                goto LABEL_23;
              ++v37;
            }
            if ( v39 )
              v20 = v39;
          }
          goto LABEL_23;
        }
LABEL_55:
        ++*(_DWORD *)(a1 + 64);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 48);
    }
    sub_1BEBF00(v17, 2 * v16);
    v27 = *(_DWORD *)(a1 + 72);
    if ( v27 )
    {
      v28 = v27 - 1;
      v29 = *(_QWORD *)(a1 + 56);
      v30 = (v27 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v26 = *(_DWORD *)(a1 + 64) + 1;
      v20 = (__int64 *)(v29 + 16LL * v30);
      v31 = *v20;
      if ( *v20 != a2 )
      {
        v32 = 1;
        v33 = 0;
        while ( v31 != -8 )
        {
          if ( v31 == -16 && !v33 )
            v33 = v20;
          v30 = v28 & (v32 + v30);
          v20 = (__int64 *)(v29 + 16LL * v30);
          v31 = *v20;
          if ( *v20 == a2 )
            goto LABEL_23;
          ++v32;
        }
        if ( v33 )
          v20 = v33;
      }
      goto LABEL_23;
    }
    goto LABEL_55;
  }
LABEL_3:
  if ( v7 == (__int64 *)(v5 + 16 * v4) )
    goto LABEL_8;
  return v7[1];
}
