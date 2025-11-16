// Function: sub_1858630
// Address: 0x1858630
//
void __fastcall sub_1858630(__int64 a1, __int64 a2)
{
  _BYTE *v3; // rax
  __int64 v5; // rbx
  _QWORD *v6; // rax
  _QWORD *v7; // r13
  _QWORD *v8; // rax
  __int64 v9; // r12
  _QWORD *v10; // rbx
  __int64 v11; // rcx
  unsigned int v12; // esi
  __int64 v13; // r8
  unsigned int v14; // edi
  __int64 *v15; // rax
  __int64 v16; // rcx
  __int64 *v17; // rdx
  __int64 v18; // r8
  _QWORD *v19; // rax
  int v20; // edx
  int v21; // edx
  __int64 v22; // r10
  unsigned int v23; // esi
  int v24; // ecx
  __int64 v25; // rdi
  int v26; // r9d
  __int64 *v27; // r8
  __int64 *v28; // rdi
  unsigned int v29; // r9d
  __int64 *v30; // rsi
  int v31; // r11d
  __int64 *v32; // r10
  int v33; // ecx
  _BYTE *v34; // rdx
  int v35; // esi
  int v36; // esi
  __int64 v37; // r10
  int v38; // r9d
  unsigned int v39; // edx
  __int64 v40; // rdi
  unsigned int v41; // [rsp+4h] [rbp-ACh]
  __int64 v42; // [rsp+8h] [rbp-A8h]
  __int64 v43; // [rsp+10h] [rbp-A0h] BYREF
  _BYTE *v44; // [rsp+18h] [rbp-98h]
  _BYTE *v45; // [rsp+20h] [rbp-90h]
  __int64 v46; // [rsp+28h] [rbp-88h]
  int v47; // [rsp+30h] [rbp-80h]
  _BYTE v48[120]; // [rsp+38h] [rbp-78h] BYREF

  v3 = v48;
  v5 = *(_QWORD *)(a2 + 8);
  v43 = 0;
  v44 = v48;
  v45 = v48;
  v46 = 8;
  v47 = 0;
  if ( !v5 )
  {
    v11 = 0;
    v34 = v48;
    goto LABEL_71;
  }
  do
  {
    v6 = sub_1648700(v5);
    sub_1857B40(a1, (unsigned __int64)v6, (__int64)&v43);
    v5 = *(_QWORD *)(v5 + 8);
  }
  while ( v5 );
  v7 = v44;
  if ( v45 == v44 )
  {
    v11 = HIDWORD(v46);
    v34 = &v44[v11 * 8];
    v3 = &v44[v11 * 8];
    if ( v44 != &v44[v11 * 8] )
    {
      while ( a2 != *v7 )
      {
        if ( ++v7 == (_QWORD *)v34 )
          goto LABEL_71;
      }
      goto LABEL_60;
    }
LABEL_71:
    v7 = v34;
    if ( v3 == v34 )
      goto LABEL_74;
LABEL_15:
    *v7 = -2;
    v8 = v45;
    ++v47;
    if ( v45 != v44 )
      goto LABEL_6;
    v11 = HIDWORD(v46);
    goto LABEL_17;
  }
  v7 = sub_16CC9F0((__int64)&v43, a2);
  v8 = v45;
  if ( a2 != *v7 )
  {
    if ( v45 != v44 )
    {
LABEL_6:
      v7 = &v8[(unsigned int)v46];
      goto LABEL_9;
    }
    v11 = HIDWORD(v46);
    goto LABEL_74;
  }
  if ( v45 != v44 )
  {
    if ( v7 == (_QWORD *)&v45[8 * (unsigned int)v46] )
      goto LABEL_9;
    goto LABEL_15;
  }
  v11 = HIDWORD(v46);
  v3 = &v45[v11 * 8];
LABEL_60:
  if ( v3 != (_BYTE *)v7 )
    goto LABEL_15;
LABEL_74:
  v8 = v45;
LABEL_17:
  v7 = &v8[v11];
  while ( 1 )
  {
LABEL_9:
    if ( v7 == v8 )
      goto LABEL_10;
    v9 = *v8;
    v10 = v8;
    if ( *v8 < 0xFFFFFFFFFFFFFFFELL )
      break;
    ++v8;
  }
  if ( v7 != v8 )
  {
    v12 = *(_DWORD *)(a1 + 320);
    v42 = a1 + 296;
    if ( !v12 )
      goto LABEL_29;
LABEL_20:
    v13 = *(_QWORD *)(a1 + 304);
    v14 = (v12 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
    v15 = (__int64 *)(v13 + 80LL * v14);
    v16 = *v15;
    if ( v9 == *v15 )
      goto LABEL_21;
    v31 = 1;
    v32 = 0;
    while ( v16 != -8 )
    {
      if ( v16 == -16 && !v32 )
        v32 = v15;
      v14 = (v12 - 1) & (v31 + v14);
      v15 = (__int64 *)(v13 + 80LL * v14);
      v16 = *v15;
      if ( v9 == *v15 )
      {
LABEL_21:
        v17 = (__int64 *)v15[2];
        v18 = (__int64)(v15 + 1);
        if ( (__int64 *)v15[3] == v17 )
        {
          v28 = &v17[*((unsigned int *)v15 + 9)];
          v29 = *((_DWORD *)v15 + 9);
          if ( v17 != v28 )
          {
            v30 = 0;
            while ( a2 != *v17 )
            {
              if ( *v17 == -2 )
                v30 = v17;
              if ( ++v17 == v28 )
              {
                if ( v30 )
                {
                  *v30 = a2;
                  --*((_DWORD *)v15 + 10);
                  ++v15[1];
                  goto LABEL_23;
                }
                goto LABEL_54;
              }
            }
            goto LABEL_23;
          }
          goto LABEL_54;
        }
LABEL_22:
        sub_16CCBA0(v18, a2);
        goto LABEL_23;
      }
      ++v31;
    }
    v33 = *(_DWORD *)(a1 + 312);
    if ( v32 )
      v15 = v32;
    ++*(_QWORD *)(a1 + 296);
    v24 = v33 + 1;
    if ( 4 * v24 >= 3 * v12 )
    {
      while ( 1 )
      {
        sub_1858440(v42, 2 * v12);
        v20 = *(_DWORD *)(a1 + 320);
        if ( !v20 )
          goto LABEL_86;
        v21 = v20 - 1;
        v22 = *(_QWORD *)(a1 + 304);
        v23 = v21 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
        v24 = *(_DWORD *)(a1 + 312) + 1;
        v15 = (__int64 *)(v22 + 80LL * v23);
        v25 = *v15;
        if ( *v15 != v9 )
          break;
LABEL_51:
        *(_DWORD *)(a1 + 312) = v24;
        if ( *v15 != -8 )
          --*(_DWORD *)(a1 + 316);
        v28 = v15 + 6;
        *v15 = v9;
        v18 = (__int64)(v15 + 1);
        v29 = 0;
        v15[1] = 0;
        v15[2] = (__int64)(v15 + 6);
        v15[3] = (__int64)(v15 + 6);
        v15[4] = 4;
        *((_DWORD *)v15 + 10) = 0;
LABEL_54:
        if ( v29 >= *((_DWORD *)v15 + 8) )
          goto LABEL_22;
        *((_DWORD *)v15 + 9) = v29 + 1;
        *v28 = a2;
        ++v15[1];
LABEL_23:
        v19 = v10 + 1;
        if ( v10 + 1 == v7 )
          goto LABEL_10;
        v9 = *v19;
        for ( ++v10; *v19 >= 0xFFFFFFFFFFFFFFFELL; v10 = v19 )
        {
          if ( v7 == ++v19 )
            goto LABEL_10;
          v9 = *v19;
        }
        if ( v7 == v10 )
          goto LABEL_10;
        v12 = *(_DWORD *)(a1 + 320);
        if ( v12 )
          goto LABEL_20;
LABEL_29:
        ++*(_QWORD *)(a1 + 296);
      }
      v26 = 1;
      v27 = 0;
      while ( v25 != -8 )
      {
        if ( v25 == -16 && !v27 )
          v27 = v15;
        v23 = v21 & (v26 + v23);
        v15 = (__int64 *)(v22 + 80LL * v23);
        v25 = *v15;
        if ( v9 == *v15 )
          goto LABEL_51;
        ++v26;
      }
    }
    else
    {
      if ( v12 - *(_DWORD *)(a1 + 316) - v24 > v12 >> 3 )
        goto LABEL_51;
      v41 = ((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4);
      sub_1858440(v42, v12);
      v35 = *(_DWORD *)(a1 + 320);
      if ( !v35 )
      {
LABEL_86:
        ++*(_DWORD *)(a1 + 312);
        BUG();
      }
      v36 = v35 - 1;
      v27 = 0;
      v37 = *(_QWORD *)(a1 + 304);
      v38 = 1;
      v39 = v36 & v41;
      v24 = *(_DWORD *)(a1 + 312) + 1;
      v15 = (__int64 *)(v37 + 80LL * (v36 & v41));
      v40 = *v15;
      if ( v9 == *v15 )
        goto LABEL_51;
      while ( v40 != -8 )
      {
        if ( v40 == -16 && !v27 )
          v27 = v15;
        v39 = v36 & (v38 + v39);
        v15 = (__int64 *)(v37 + 80LL * v39);
        v40 = *v15;
        if ( v9 == *v15 )
          goto LABEL_51;
        ++v38;
      }
    }
    if ( v27 )
      v15 = v27;
    goto LABEL_51;
  }
LABEL_10:
  if ( v45 != v44 )
    _libc_free((unsigned __int64)v45);
}
