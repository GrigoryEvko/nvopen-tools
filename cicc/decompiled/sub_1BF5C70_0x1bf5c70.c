// Function: sub_1BF5C70
// Address: 0x1bf5c70
//
__int64 __fastcall sub_1BF5C70(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 v8; // r14
  unsigned int v9; // esi
  __int64 v10; // rcx
  int v11; // r10d
  __int64 *v12; // r13
  unsigned int v13; // edx
  __int64 *v14; // rax
  __int64 v15; // r8
  __int64 v16; // rax
  int v18; // eax
  int v19; // edx
  __int64 v20; // rcx
  __int64 v21; // rdx
  __int64 *v22; // r8
  __int64 v23; // r12
  char *v24; // rdi
  __int64 v25; // rax
  bool v26; // zf
  int v27; // eax
  __int64 v28; // rdi
  unsigned int v29; // eax
  __int64 v30; // rsi
  __int64 *v31; // r8
  int v32; // eax
  int v33; // eax
  __int64 v34; // rsi
  int v35; // r8d
  unsigned int v36; // r15d
  __int64 *v37; // rdi
  _QWORD v38[2]; // [rsp+0h] [rbp-E0h] BYREF
  __int64 v39; // [rsp+10h] [rbp-D0h]
  __int128 v40; // [rsp+18h] [rbp-C8h]
  __int64 v41; // [rsp+28h] [rbp-B8h]
  _BYTE *v42; // [rsp+30h] [rbp-B0h]
  __int64 v43; // [rsp+38h] [rbp-A8h]
  _BYTE v44[16]; // [rsp+40h] [rbp-A0h] BYREF
  __int64 v45; // [rsp+50h] [rbp-90h] BYREF
  _QWORD v46[2]; // [rsp+58h] [rbp-88h] BYREF
  __int64 v47; // [rsp+68h] [rbp-78h]
  int v48; // [rsp+70h] [rbp-70h]
  __int64 v49; // [rsp+78h] [rbp-68h]
  __int64 v50; // [rsp+80h] [rbp-60h]
  char *v51; // [rsp+88h] [rbp-58h] BYREF
  __int64 v52; // [rsp+90h] [rbp-50h]
  _BYTE v53[72]; // [rsp+98h] [rbp-48h] BYREF

  v8 = *a2;
  v9 = *(_DWORD *)(a1 + 24);
  if ( !v9 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_39;
  }
  v10 = *(_QWORD *)(a1 + 8);
  v11 = 1;
  v12 = 0;
  v13 = (v9 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
  v14 = (__int64 *)(v10 + 16LL * v13);
  v15 = *v14;
  if ( v8 == *v14 )
  {
LABEL_3:
    v16 = *((unsigned int *)v14 + 2);
    return *(_QWORD *)(a1 + 32) + 88 * v16 + 8;
  }
  while ( v15 != -8 )
  {
    if ( !v12 && v15 == -16 )
      v12 = v14;
    a6 = v11 + 1;
    v13 = (v9 - 1) & (v11 + v13);
    v14 = (__int64 *)(v10 + 16LL * v13);
    v15 = *v14;
    if ( v8 == *v14 )
      goto LABEL_3;
    ++v11;
  }
  if ( !v12 )
    v12 = v14;
  v18 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v19 = v18 + 1;
  if ( 4 * (v18 + 1) >= 3 * v9 )
  {
LABEL_39:
    sub_1A72540(a1, 2 * v9);
    v27 = *(_DWORD *)(a1 + 24);
    if ( v27 )
    {
      v20 = (unsigned int)(v27 - 1);
      v28 = *(_QWORD *)(a1 + 8);
      v29 = v20 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
      v19 = *(_DWORD *)(a1 + 16) + 1;
      v12 = (__int64 *)(v28 + 16LL * v29);
      v30 = *v12;
      if ( v8 != *v12 )
      {
        a6 = 1;
        v31 = 0;
        while ( v30 != -8 )
        {
          if ( !v31 && v30 == -16 )
            v31 = v12;
          v29 = v20 & (a6 + v29);
          v12 = (__int64 *)(v28 + 16LL * v29);
          v30 = *v12;
          if ( v8 == *v12 )
            goto LABEL_15;
          ++a6;
        }
        if ( v31 )
          v12 = v31;
      }
      goto LABEL_15;
    }
    goto LABEL_62;
  }
  v20 = v9 >> 3;
  if ( v9 - *(_DWORD *)(a1 + 20) - v19 <= (unsigned int)v20 )
  {
    sub_1A72540(a1, v9);
    v32 = *(_DWORD *)(a1 + 24);
    if ( v32 )
    {
      v33 = v32 - 1;
      v34 = *(_QWORD *)(a1 + 8);
      v35 = 1;
      v36 = v33 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
      v19 = *(_DWORD *)(a1 + 16) + 1;
      v37 = 0;
      v12 = (__int64 *)(v34 + 16LL * v36);
      v20 = *v12;
      if ( v8 != *v12 )
      {
        while ( v20 != -8 )
        {
          if ( v20 == -16 && !v37 )
            v37 = v12;
          a6 = v35 + 1;
          v36 = v33 & (v35 + v36);
          v12 = (__int64 *)(v34 + 16LL * v36);
          v20 = *v12;
          if ( v8 == *v12 )
            goto LABEL_15;
          ++v35;
        }
        if ( v37 )
          v12 = v37;
      }
      goto LABEL_15;
    }
LABEL_62:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
LABEL_15:
  *(_DWORD *)(a1 + 16) = v19;
  if ( *v12 != -8 )
    --*(_DWORD *)(a1 + 20);
  *v12 = v8;
  *((_DWORD *)v12 + 2) = 0;
  v21 = *a2;
  v22 = &v45;
  v41 = 0;
  v23 = *(_QWORD *)(a1 + 40);
  v38[0] = 6;
  v38[1] = 0;
  v39 = 0;
  v42 = v44;
  v43 = 0x200000000LL;
  v45 = v21;
  v46[0] = 6;
  v46[1] = 0;
  v47 = 0;
  v48 = 0;
  v49 = 0;
  v50 = 0;
  v51 = v53;
  v52 = 0x200000000LL;
  v40 = 0;
  if ( v23 == *(_QWORD *)(a1 + 48) )
  {
    sub_1BF5870((__int64 *)(a1 + 32), (char *)v23, (__int64)&v45, v20, (int)&v45);
    v24 = v51;
  }
  else
  {
    v24 = v53;
    if ( v23 )
    {
      *(_QWORD *)v23 = v21;
      *(_QWORD *)(v23 + 8) = 6;
      *(_QWORD *)(v23 + 16) = 0;
      v25 = v47;
      v26 = v47 == -8;
      *(_QWORD *)(v23 + 24) = v47;
      LOBYTE(v20) = !v26;
      LOBYTE(v21) = v25 != 0;
      if ( v25 != 0 && !v26 && v25 != -16 )
        sub_1649AC0((unsigned __int64 *)(v23 + 8), v46[0] & 0xFFFFFFFFFFFFFFF8LL);
      *(_DWORD *)(v23 + 32) = v48;
      *(_QWORD *)(v23 + 40) = v49;
      *(_QWORD *)(v23 + 48) = v50;
      *(_QWORD *)(v23 + 56) = v23 + 72;
      *(_QWORD *)(v23 + 64) = 0x200000000LL;
      if ( (_DWORD)v52 )
        sub_1BF0B80(v23 + 56, &v51, v21, v20, (int)v22, a6);
      v23 = *(_QWORD *)(a1 + 40);
      v24 = v51;
    }
    *(_QWORD *)(a1 + 40) = v23 + 88;
  }
  if ( v24 != v53 )
    _libc_free((unsigned __int64)v24);
  if ( v47 != 0 && v47 != -8 && v47 != -16 )
    sub_1649B30(v46);
  if ( v42 != v44 )
    _libc_free((unsigned __int64)v42);
  if ( v39 != 0 && v39 != -8 && v39 != -16 )
    sub_1649B30(v38);
  v16 = -1171354717 * (unsigned int)((__int64)(*(_QWORD *)(a1 + 40) - *(_QWORD *)(a1 + 32)) >> 3) - 1;
  *((_DWORD *)v12 + 2) = v16;
  return *(_QWORD *)(a1 + 32) + 88 * v16 + 8;
}
