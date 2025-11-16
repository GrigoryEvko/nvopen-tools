// Function: sub_2E58AA0
// Address: 0x2e58aa0
//
void __fastcall sub_2E58AA0(__int64 a1, __int64 a2)
{
  __int64 v3; // rdx
  __int64 v4; // rcx
  unsigned int v5; // eax
  __int64 *v6; // rax
  __int64 v7; // rsi
  unsigned int v8; // ecx
  __int64 v9; // rbx
  unsigned int v10; // edx
  __int64 *v11; // rax
  __int64 v12; // r9
  __int64 v13; // r13
  int v14; // ecx
  __int64 v15; // r10
  unsigned int v16; // edi
  unsigned __int64 v17; // r8
  __int64 v18; // rdx
  int v20; // r11d
  int v21; // r14d
  unsigned int v23; // r15d
  __int64 v24; // rax
  int v25; // r11d
  unsigned int v26; // eax
  int v27; // edx
  unsigned int v28; // r9d
  unsigned int v29; // esi
  __int64 v30; // r10
  int v31; // ecx
  unsigned __int64 v32; // r8
  __int64 v33; // rdx
  unsigned __int64 v34; // r8
  int v37; // eax
  int v38; // r10d
  __int64 v39[7]; // [rsp+8h] [rbp-38h] BYREF

  v39[0] = a2;
  v3 = *(_QWORD *)(a1 + 16);
  if ( a2 )
  {
    v4 = (unsigned int)(*(_DWORD *)(a2 + 24) + 1);
    v5 = *(_DWORD *)(a2 + 24) + 1;
  }
  else
  {
    v4 = 0;
    v5 = 0;
  }
  if ( v5 >= *(_DWORD *)(v3 + 32) || !*(_QWORD *)(*(_QWORD *)(v3 + 24) + 8 * v4) )
    return;
  v6 = sub_2E57C80(a1 + 112, v39);
  v7 = *(_QWORD *)(a1 + 120);
  v8 = *(_DWORD *)(a1 + 136);
  v9 = *v6;
  if ( v8 )
  {
    v10 = (v8 - 1) & ((LODWORD(v39[0]) >> 9) ^ (LODWORD(v39[0]) >> 4));
    v11 = (__int64 *)(v7 + 16LL * v10);
    v12 = *v11;
    if ( v39[0] == *v11 )
      goto LABEL_7;
    v37 = 1;
    while ( v12 != -4096 )
    {
      v38 = v37 + 1;
      v10 = (v8 - 1) & (v37 + v10);
      v11 = (__int64 *)(v7 + 16LL * v10);
      v12 = *v11;
      if ( v39[0] == *v11 )
        goto LABEL_7;
      v37 = v38;
    }
  }
  v11 = (__int64 *)(v7 + 16LL * v8);
LABEL_7:
  v13 = v11[1];
  v14 = *(_DWORD *)(v13 + 88);
  if ( !v14 )
    goto LABEL_13;
  v15 = *(_QWORD *)(v13 + 24);
  v16 = (unsigned int)(v14 - 1) >> 6;
  v17 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v14;
  v18 = 0;
  while ( 1 )
  {
    _RCX = *(_QWORD *)(v15 + 8 * v18);
    if ( v16 == (_DWORD)v18 )
      _RCX = v17 & *(_QWORD *)(v15 + 8 * v18);
    if ( _RCX )
      break;
    if ( ++v18 == v16 + 1 )
      goto LABEL_13;
  }
  __asm { tzcnt   rcx, rcx }
  v23 = ((_DWORD)v18 << 6) + _RCX;
  if ( v23 == -1 )
  {
LABEL_13:
    v20 = *(_DWORD *)(v9 + 8);
    v21 = *(_DWORD *)(v9 + 12);
  }
  else
  {
    do
    {
      v24 = sub_307B990(
              *(unsigned int *)(*(_QWORD *)(a1 + 88) + 4LL * v23),
              *(_QWORD *)(a1 + 192),
              *(_QWORD *)(a1 + 176));
      v25 = v24;
      v26 = v23 + 1;
      v20 = *(_DWORD *)(v9 + 8) + v25;
      v21 = *(_DWORD *)(v9 + 12) + HIDWORD(v24);
      *(_DWORD *)(v9 + 8) = v20;
      *(_DWORD *)(v9 + 12) = v21;
      v27 = *(_DWORD *)(v13 + 88);
      if ( v27 == v23 + 1 )
        break;
      v28 = v26 >> 6;
      v29 = (unsigned int)(v27 - 1) >> 6;
      if ( v26 >> 6 > v29 )
        break;
      v30 = *(_QWORD *)(v13 + 24);
      v31 = 64 - (v26 & 0x3F);
      v32 = 0xFFFFFFFFFFFFFFFFLL >> v31;
      v33 = v28;
      if ( v31 == 64 )
        v32 = 0;
      v34 = ~v32;
      while ( 1 )
      {
        _RAX = *(_QWORD *)(v30 + 8 * v33);
        if ( v28 == (_DWORD)v33 )
          _RAX = v34 & *(_QWORD *)(v30 + 8 * v33);
        if ( (_DWORD)v33 == v29 )
          _RAX &= 0xFFFFFFFFFFFFFFFFLL >> -(char)*(_DWORD *)(v13 + 88);
        if ( _RAX )
          break;
        if ( v29 < (unsigned int)++v33 )
          goto LABEL_14;
      }
      __asm { tzcnt   rax, rax }
      v23 = ((_DWORD)v33 << 6) + _RAX;
    }
    while ( v23 != -1 );
  }
LABEL_14:
  if ( *(_DWORD *)(a1 + 36) >= v21 )
    v21 = *(_DWORD *)(a1 + 36);
  if ( *(_DWORD *)(a1 + 32) >= v20 )
    v20 = *(_DWORD *)(a1 + 32);
  *(_DWORD *)(a1 + 36) = v21;
  *(_DWORD *)(a1 + 32) = v20;
  *(_QWORD *)v9 = *(_QWORD *)(v9 + 8);
  sub_2E58200(a1, v39[0]);
}
