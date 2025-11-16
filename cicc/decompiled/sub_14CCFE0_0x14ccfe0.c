// Function: sub_14CCFE0
// Address: 0x14ccfe0
//
_QWORD *__fastcall sub_14CCFE0(__int64 a1, __int64 a2)
{
  unsigned int v3; // r8d
  __int64 v4; // rcx
  unsigned int v5; // eax
  __int64 v6; // r14
  __int64 v7; // rdx
  _QWORD *v8; // r14
  __int64 v10; // rdi
  int v11; // edx
  __int64 v12; // rax
  _QWORD *v13; // r15
  int v14; // ecx
  __int64 v15; // rdx
  _QWORD *v16; // r12
  __int64 v17; // r12
  __int64 v18; // rsi
  unsigned int v19; // edx
  _QWORD *v20; // r14
  __int64 v21; // r10
  __int64 v22; // r15
  _BYTE *v23; // r12
  __int64 v24; // rax
  __int64 v25; // r15
  _BYTE *v26; // r12
  __int64 v27; // rax
  int v28; // ecx
  __int64 v29; // r9
  unsigned int v30; // edx
  int v31; // edi
  _QWORD *v32; // rsi
  __int64 v33; // r8
  int v34; // r9d
  int v35; // r11d
  int v36; // esi
  int v37; // edx
  int v38; // ecx
  __int64 v39; // r8
  _QWORD *v40; // r9
  unsigned int v41; // edx
  int v42; // esi
  __int64 v43; // rdi
  _QWORD v44[2]; // [rsp+8h] [rbp-E8h] BYREF
  __int64 v45; // [rsp+18h] [rbp-D8h]
  __int64 v46; // [rsp+20h] [rbp-D0h]
  _BYTE *v47; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v48; // [rsp+38h] [rbp-B8h]
  _BYTE v49[32]; // [rsp+40h] [rbp-B0h] BYREF
  void *v50; // [rsp+60h] [rbp-90h]
  __int64 v51; // [rsp+68h] [rbp-88h] BYREF
  __int64 v52; // [rsp+70h] [rbp-80h]
  __int64 v53; // [rsp+78h] [rbp-78h]
  __int64 v54; // [rsp+80h] [rbp-70h]
  _BYTE *v55; // [rsp+88h] [rbp-68h] BYREF
  __int64 v56; // [rsp+90h] [rbp-60h]
  _BYTE v57[88]; // [rsp+98h] [rbp-58h] BYREF

  v3 = *(_DWORD *)(a1 + 176);
  if ( v3 )
  {
    v4 = *(_QWORD *)(a1 + 160);
    v5 = (v3 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v6 = v4 + 88LL * v5;
    v7 = *(_QWORD *)(v6 + 24);
    if ( v7 == a2 )
    {
LABEL_10:
      if ( v6 != v4 + 88LL * v3 )
        return (_QWORD *)(v6 + 40);
    }
    else
    {
      v34 = 1;
      while ( v7 != -8 )
      {
        v5 = (v3 - 1) & (v34 + v5);
        v6 = v4 + 88LL * v5;
        v7 = *(_QWORD *)(v6 + 24);
        if ( v7 == a2 )
          goto LABEL_10;
        ++v34;
      }
    }
  }
  v45 = a2;
  v44[0] = 2;
  v44[1] = 0;
  if ( a2 == -8 || a2 == 0 || a2 == -16 )
  {
    v53 = a2;
    v46 = a1;
    v47 = v49;
    v48 = 0x100000000LL;
    v51 = 2;
    v52 = 0;
LABEL_14:
    v54 = a1;
    v50 = &unk_49ECBD0;
    v55 = v57;
    v56 = 0x100000000LL;
    goto LABEL_15;
  }
  sub_164C220(v44);
  v46 = a1;
  v47 = v49;
  v48 = 0x100000000LL;
  v53 = v45;
  v51 = v44[0] & 6;
  v52 = 0;
  if ( v45 == -8 || v45 == -16 || !v45 )
  {
    v3 = *(_DWORD *)(a1 + 176);
    goto LABEL_14;
  }
  sub_1649AC0(&v51, v44[0] & 0xFFFFFFFFFFFFFFF8LL);
  v50 = &unk_49ECBD0;
  v55 = v57;
  v54 = v46;
  v56 = 0x100000000LL;
  if ( (_DWORD)v48 )
    sub_14CB800((__int64)&v55, (__int64 *)&v47);
  v3 = *(_DWORD *)(a1 + 176);
LABEL_15:
  v10 = a1 + 152;
  if ( !v3 )
  {
    ++*(_QWORD *)(a1 + 152);
    goto LABEL_17;
  }
  v12 = v53;
  v18 = *(_QWORD *)(a1 + 160);
  v19 = (v3 - 1) & (((unsigned int)v53 >> 9) ^ ((unsigned int)v53 >> 4));
  v20 = (_QWORD *)(v18 + 88LL * v19);
  v21 = v20[3];
  if ( v53 != v21 )
  {
    v35 = 1;
    v13 = 0;
    while ( v21 != -8 )
    {
      if ( !v13 && v21 == -16 )
        v13 = v20;
      v19 = (v3 - 1) & (v35 + v19);
      v20 = (_QWORD *)(v18 + 88LL * v19);
      v21 = v20[3];
      if ( v53 == v21 )
        goto LABEL_31;
      ++v35;
    }
    v36 = *(_DWORD *)(a1 + 168);
    if ( !v13 )
      v13 = v20;
    ++*(_QWORD *)(a1 + 152);
    v14 = v36 + 1;
    if ( 4 * (v36 + 1) < 3 * v3 )
    {
      if ( v3 - *(_DWORD *)(a1 + 172) - v14 > v3 >> 3 )
      {
LABEL_20:
        *(_DWORD *)(a1 + 168) = v14;
        if ( v13[3] == -8 )
        {
          v16 = v13 + 1;
          if ( v12 != -8 )
          {
LABEL_25:
            v13[3] = v12;
            if ( v12 != 0 && v12 != -8 && v12 != -16 )
              sub_1649AC0(v16, v51 & 0xFFFFFFFFFFFFFFF8LL);
          }
        }
        else
        {
          --*(_DWORD *)(a1 + 172);
          v15 = v13[3];
          if ( v12 != v15 )
          {
            v16 = v13 + 1;
            if ( v15 != 0 && v15 != -8 && v15 != -16 )
            {
              sub_1649B30(v13 + 1);
              v12 = v53;
            }
            goto LABEL_25;
          }
        }
        v8 = v13 + 5;
        v13[4] = v54;
        v13[5] = v13 + 7;
        v13[6] = 0x100000000LL;
        v17 = (unsigned int)v56;
        if ( (_DWORD)v56 )
        {
          sub_14CB800((__int64)(v13 + 5), (__int64 *)&v55);
          v17 = (unsigned int)v56;
        }
        goto LABEL_32;
      }
      sub_14CCB30(v10, v3);
      v37 = *(_DWORD *)(a1 + 176);
      if ( v37 )
      {
        v12 = v53;
        v38 = v37 - 1;
        v39 = *(_QWORD *)(a1 + 160);
        v40 = 0;
        v41 = (v37 - 1) & (((unsigned int)v53 >> 9) ^ ((unsigned int)v53 >> 4));
        v13 = (_QWORD *)(v39 + 88LL * v41);
        v42 = 1;
        v43 = v13[3];
        if ( v53 != v43 )
        {
          while ( v43 != -8 )
          {
            if ( v43 == -16 && !v40 )
              v40 = v13;
            v41 = v38 & (v42 + v41);
            v13 = (_QWORD *)(v39 + 88LL * v41);
            v43 = v13[3];
            if ( v53 == v43 )
              goto LABEL_19;
            ++v42;
          }
          if ( v40 )
            v13 = v40;
        }
        goto LABEL_19;
      }
      goto LABEL_18;
    }
LABEL_17:
    sub_14CCB30(v10, 2 * v3);
    v11 = *(_DWORD *)(a1 + 176);
    if ( v11 )
    {
      v12 = v53;
      v28 = v11 - 1;
      v29 = *(_QWORD *)(a1 + 160);
      v30 = (v11 - 1) & (((unsigned int)v53 >> 9) ^ ((unsigned int)v53 >> 4));
      v31 = 1;
      v13 = (_QWORD *)(v29 + 88LL * v30);
      v32 = 0;
      v33 = v13[3];
      if ( v33 != v53 )
      {
        while ( v33 != -8 )
        {
          if ( !v32 && v33 == -16 )
            v32 = v13;
          v30 = v28 & (v31 + v30);
          v13 = (_QWORD *)(v29 + 88LL * v30);
          v33 = v13[3];
          if ( v53 == v33 )
            goto LABEL_19;
          ++v31;
        }
        if ( v32 )
          v13 = v32;
      }
      goto LABEL_19;
    }
LABEL_18:
    v12 = v53;
    v13 = 0;
LABEL_19:
    v14 = *(_DWORD *)(a1 + 168) + 1;
    goto LABEL_20;
  }
LABEL_31:
  v17 = (unsigned int)v56;
  v8 = v20 + 5;
LABEL_32:
  v22 = (__int64)v55;
  v23 = &v55[32 * v17];
  if ( v55 != v23 )
  {
    do
    {
      v24 = *((_QWORD *)v23 - 2);
      v23 -= 32;
      if ( v24 != 0 && v24 != -8 && v24 != -16 )
        sub_1649B30(v23);
    }
    while ( (_BYTE *)v22 != v23 );
    v23 = v55;
  }
  if ( v23 != v57 )
    _libc_free((unsigned __int64)v23);
  v50 = &unk_49EE2B0;
  if ( v53 != 0 && v53 != -8 && v53 != -16 )
    sub_1649B30(&v51);
  v25 = (__int64)v47;
  v26 = &v47[32 * (unsigned int)v48];
  if ( v47 != v26 )
  {
    do
    {
      v27 = *((_QWORD *)v26 - 2);
      v26 -= 32;
      if ( v27 != 0 && v27 != -8 && v27 != -16 )
        sub_1649B30(v26);
    }
    while ( (_BYTE *)v25 != v26 );
    v26 = v47;
  }
  if ( v26 != v49 )
    _libc_free((unsigned __int64)v26);
  if ( v45 != 0 && v45 != -8 && v45 != -16 )
    sub_1649B30(v44);
  return v8;
}
