// Function: sub_1DFE8A0
// Address: 0x1dfe8a0
//
void __fastcall sub_1DFE8A0(__int64 a1, __int64 a2)
{
  __int64 v4; // r13
  __int64 v5; // rdx
  __int64 v6; // rax
  __int64 v7; // rsi
  unsigned int v8; // ecx
  __int64 *v9; // rdx
  __int64 v10; // r8
  __int64 *v11; // rax
  __int64 v12; // r13
  __int64 v13; // rbx
  __int64 v14; // rdx
  __int64 v15; // rsi
  unsigned int v16; // ecx
  __int64 *v17; // rax
  __int64 v18; // r8
  __int64 v19; // r13
  int v20; // ecx
  __int64 v21; // r10
  unsigned int v22; // edi
  unsigned __int64 v23; // r8
  __int64 v24; // rdx
  int v26; // r11d
  int v27; // r14d
  unsigned int v29; // r15d
  __int64 v30; // rax
  int v31; // r11d
  unsigned int v32; // eax
  int v33; // edx
  unsigned int v34; // r9d
  unsigned int v35; // esi
  __int64 v36; // r10
  int v37; // ecx
  unsigned __int64 v38; // r8
  __int64 v39; // rdx
  unsigned __int64 v40; // r8
  int v43; // edx
  int v44; // r9d
  int v45; // eax
  int v46; // r9d
  __int64 v47; // [rsp+8h] [rbp-48h] BYREF
  __int64 v48; // [rsp+10h] [rbp-40h] BYREF
  __int64 v49[7]; // [rsp+18h] [rbp-38h] BYREF

  v4 = *(_QWORD *)(a1 + 16);
  v47 = a2;
  sub_1E06620(v4);
  v5 = *(_QWORD *)(v4 + 1312);
  v6 = *(unsigned int *)(v5 + 48);
  if ( !(_DWORD)v6 )
    return;
  v7 = *(_QWORD *)(v5 + 32);
  v8 = (v6 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v9 = (__int64 *)(v7 + 16LL * v8);
  v10 = *v9;
  if ( a2 == *v9 )
  {
LABEL_3:
    if ( v9 == (__int64 *)(v7 + 16 * v6) || !v9[1] )
      return;
    v11 = sub_1DFD350(a1 + 112, &v47);
    v12 = v47;
    v13 = v11[1];
    v48 = v47;
    sub_1DF9290(a1 + 112, &v48, v49);
    v14 = *(unsigned int *)(a1 + 136);
    v15 = *(_QWORD *)(a1 + 120);
    if ( (_DWORD)v14 )
    {
      v16 = (v14 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
      v17 = (__int64 *)(v15 + 16LL * v16);
      v18 = *v17;
      if ( v12 == *v17 )
        goto LABEL_7;
      v45 = 1;
      while ( v18 != -8 )
      {
        v46 = v45 + 1;
        v16 = (v14 - 1) & (v45 + v16);
        v17 = (__int64 *)(v15 + 16LL * v16);
        v18 = *v17;
        if ( v12 == *v17 )
          goto LABEL_7;
        v45 = v46;
      }
    }
    v17 = (__int64 *)(v15 + 16 * v14);
LABEL_7:
    v19 = v17[1];
    v20 = *(_DWORD *)(v19 + 40);
    if ( !v20 )
      goto LABEL_13;
    v21 = *(_QWORD *)(v19 + 24);
    v22 = (unsigned int)(v20 - 1) >> 6;
    v23 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v20;
    v24 = 0;
    while ( 1 )
    {
      _RCX = *(_QWORD *)(v21 + 8 * v24);
      if ( v22 == (_DWORD)v24 )
        _RCX = v23 & *(_QWORD *)(v21 + 8 * v24);
      if ( _RCX )
        break;
      if ( v22 + 1 == ++v24 )
        goto LABEL_13;
    }
    __asm { tzcnt   rcx, rcx }
    v29 = ((_DWORD)v24 << 6) + _RCX;
    if ( v29 == -1 )
    {
LABEL_13:
      v26 = *(_DWORD *)(v13 + 8);
      v27 = *(_DWORD *)(v13 + 12);
    }
    else
    {
      do
      {
        v30 = sub_21EA570(
                *(unsigned int *)(*(_QWORD *)(a1 + 88) + 4LL * v29),
                *(_QWORD *)(a1 + 192),
                *(_QWORD *)(a1 + 176));
        v31 = v30;
        v32 = v29 + 1;
        v26 = *(_DWORD *)(v13 + 8) + v31;
        v27 = *(_DWORD *)(v13 + 12) + HIDWORD(v30);
        *(_DWORD *)(v13 + 8) = v26;
        *(_DWORD *)(v13 + 12) = v27;
        v33 = *(_DWORD *)(v19 + 40);
        if ( v33 == v29 + 1 )
          break;
        v34 = v32 >> 6;
        v35 = (unsigned int)(v33 - 1) >> 6;
        if ( v32 >> 6 > v35 )
          break;
        v36 = *(_QWORD *)(v19 + 24);
        v37 = 64 - (v32 & 0x3F);
        v38 = 0xFFFFFFFFFFFFFFFFLL >> v37;
        v39 = v34;
        if ( v37 == 64 )
          v38 = 0;
        v40 = ~v38;
        while ( 1 )
        {
          _RAX = *(_QWORD *)(v36 + 8 * v39);
          if ( v34 == (_DWORD)v39 )
            _RAX = v40 & *(_QWORD *)(v36 + 8 * v39);
          if ( (_DWORD)v39 == v35 )
            _RAX &= 0xFFFFFFFFFFFFFFFFLL >> -(char)*(_DWORD *)(v19 + 40);
          if ( _RAX )
            break;
          if ( v35 < (unsigned int)++v39 )
            goto LABEL_14;
        }
        __asm { tzcnt   rax, rax }
        v29 = ((_DWORD)v39 << 6) + _RAX;
      }
      while ( v29 != -1 );
    }
LABEL_14:
    if ( *(_DWORD *)(a1 + 36) >= v27 )
      v27 = *(_DWORD *)(a1 + 36);
    if ( *(_DWORD *)(a1 + 32) >= v26 )
      v26 = *(_DWORD *)(a1 + 32);
    *(_DWORD *)(a1 + 36) = v27;
    *(_DWORD *)(a1 + 32) = v26;
    *(_QWORD *)v13 = *(_QWORD *)(v13 + 8);
    sub_1DFDF40(a1, v47);
    return;
  }
  v43 = 1;
  while ( v10 != -8 )
  {
    v44 = v43 + 1;
    v8 = (v6 - 1) & (v43 + v8);
    v9 = (__int64 *)(v7 + 16LL * v8);
    v10 = *v9;
    if ( a2 == *v9 )
      goto LABEL_3;
    v43 = v44;
  }
}
