// Function: sub_11D6B90
// Address: 0x11d6b90
//
__int64 __fastcall sub_11D6B90(__int64 *a1, __int64 a2)
{
  __int64 v4; // rbx
  unsigned int v5; // esi
  __int64 v6; // rdi
  int v7; // r11d
  __int64 *v8; // rdx
  unsigned int v9; // ecx
  __int64 *v10; // rax
  __int64 v11; // r9
  __int64 v12; // r12
  int v14; // eax
  int v15; // ecx
  __int64 v16; // rax
  __int64 v17; // rsi
  __int64 v18; // rax
  __int64 *v19; // r14
  __int64 *v20; // rbx
  __int64 *i; // rdx
  __int64 v22; // rdi
  unsigned int v23; // ecx
  __int64 *v24; // rbx
  __int64 *v25; // r13
  __int64 v26; // rdi
  int v27; // eax
  int v28; // esi
  __int64 v29; // r8
  unsigned int v30; // eax
  __int64 v31; // rdi
  int v32; // r10d
  __int64 *v33; // r9
  int v34; // eax
  int v35; // eax
  __int64 v36; // rdi
  int v37; // r9d
  unsigned int v38; // r12d
  __int64 *v39; // r8
  __int64 v40; // rsi
  _QWORD v41[4]; // [rsp+10h] [rbp-D0h] BYREF
  __int64 v42; // [rsp+30h] [rbp-B0h]
  __int64 v43; // [rsp+38h] [rbp-A8h]
  unsigned int v44; // [rsp+40h] [rbp-A0h]
  __int64 v45; // [rsp+48h] [rbp-98h]
  __int64 v46; // [rsp+50h] [rbp-90h]
  __int64 *v47; // [rsp+58h] [rbp-88h]
  __int64 v48; // [rsp+60h] [rbp-80h]
  _BYTE v49[32]; // [rsp+68h] [rbp-78h] BYREF
  __int64 *v50; // [rsp+88h] [rbp-58h]
  __int64 v51; // [rsp+90h] [rbp-50h]
  _QWORD v52[9]; // [rsp+98h] [rbp-48h] BYREF

  v4 = *a1;
  v5 = *(_DWORD *)(*a1 + 24);
  if ( !v5 )
  {
    ++*(_QWORD *)v4;
    goto LABEL_33;
  }
  v6 = *(_QWORD *)(v4 + 8);
  v7 = 1;
  v8 = 0;
  v9 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v10 = (__int64 *)(v6 + 16LL * v9);
  v11 = *v10;
  if ( *v10 == a2 )
  {
LABEL_3:
    v12 = v10[1];
    if ( v12 )
      return v12;
    goto LABEL_18;
  }
  while ( v11 != -4096 )
  {
    if ( v11 == -8192 && !v8 )
      v8 = v10;
    v9 = (v5 - 1) & (v7 + v9);
    v10 = (__int64 *)(v6 + 16LL * v9);
    v11 = *v10;
    if ( *v10 == a2 )
      goto LABEL_3;
    ++v7;
  }
  if ( !v8 )
    v8 = v10;
  v14 = *(_DWORD *)(v4 + 16);
  ++*(_QWORD *)v4;
  v15 = v14 + 1;
  if ( 4 * (v14 + 1) >= 3 * v5 )
  {
LABEL_33:
    sub_116E750(v4, 2 * v5);
    v27 = *(_DWORD *)(v4 + 24);
    if ( v27 )
    {
      v28 = v27 - 1;
      v29 = *(_QWORD *)(v4 + 8);
      v30 = (v27 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v15 = *(_DWORD *)(v4 + 16) + 1;
      v8 = (__int64 *)(v29 + 16LL * v30);
      v31 = *v8;
      if ( *v8 != a2 )
      {
        v32 = 1;
        v33 = 0;
        while ( v31 != -4096 )
        {
          if ( !v33 && v31 == -8192 )
            v33 = v8;
          v30 = v28 & (v32 + v30);
          v8 = (__int64 *)(v29 + 16LL * v30);
          v31 = *v8;
          if ( *v8 == a2 )
            goto LABEL_15;
          ++v32;
        }
        if ( v33 )
          v8 = v33;
      }
      goto LABEL_15;
    }
    goto LABEL_56;
  }
  if ( v5 - *(_DWORD *)(v4 + 20) - v15 <= v5 >> 3 )
  {
    sub_116E750(v4, v5);
    v34 = *(_DWORD *)(v4 + 24);
    if ( v34 )
    {
      v35 = v34 - 1;
      v36 = *(_QWORD *)(v4 + 8);
      v37 = 1;
      v38 = v35 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v39 = 0;
      v15 = *(_DWORD *)(v4 + 16) + 1;
      v8 = (__int64 *)(v36 + 16LL * v38);
      v40 = *v8;
      if ( *v8 != a2 )
      {
        while ( v40 != -4096 )
        {
          if ( v40 == -8192 && !v39 )
            v39 = v8;
          v38 = v35 & (v37 + v38);
          v8 = (__int64 *)(v36 + 16LL * v38);
          v40 = *v8;
          if ( *v8 == a2 )
            goto LABEL_15;
          ++v37;
        }
        if ( v39 )
          v8 = v39;
      }
      goto LABEL_15;
    }
LABEL_56:
    ++*(_DWORD *)(v4 + 16);
    BUG();
  }
LABEL_15:
  *(_DWORD *)(v4 + 16) = v15;
  if ( *v8 != -4096 )
    --*(_DWORD *)(v4 + 20);
  *v8 = a2;
  v8[1] = 0;
LABEL_18:
  v16 = a1[6];
  v17 = a2;
  v41[0] = a1;
  v41[2] = v16;
  v47 = (__int64 *)v49;
  v41[1] = v4;
  v41[3] = 0;
  v42 = 0;
  v43 = 0;
  v44 = 0;
  v45 = 0;
  v46 = 0;
  v48 = 0x400000000LL;
  v50 = v52;
  v51 = 0;
  v52[0] = 0;
  v52[1] = 1;
  v18 = sub_11D5FF0((__int64)v41, a2);
  v19 = v47;
  v12 = v18;
  v20 = &v47[(unsigned int)v48];
  if ( v47 != v20 )
  {
    for ( i = v47; ; i = v47 )
    {
      v22 = *v19;
      v23 = (unsigned int)(v19 - i) >> 7;
      v17 = 4096LL << v23;
      if ( v23 >= 0x1E )
        v17 = 0x40000000000LL;
      ++v19;
      sub_C7D6A0(v22, v17, 16);
      if ( v20 == v19 )
        break;
    }
  }
  v24 = v50;
  v25 = &v50[2 * (unsigned int)v51];
  if ( v50 != v25 )
  {
    do
    {
      v17 = v24[1];
      v26 = *v24;
      v24 += 2;
      sub_C7D6A0(v26, v17, 16);
    }
    while ( v25 != v24 );
    v25 = v50;
  }
  if ( v25 != v52 )
    _libc_free(v25, v17);
  if ( v47 != (__int64 *)v49 )
    _libc_free(v47, v17);
  sub_C7D6A0(v42, 16LL * v44, 8);
  return v12;
}
