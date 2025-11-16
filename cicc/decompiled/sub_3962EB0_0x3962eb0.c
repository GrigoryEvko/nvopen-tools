// Function: sub_3962EB0
// Address: 0x3962eb0
//
__int64 __fastcall sub_3962EB0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v6; // rax
  __int64 v7; // rax
  int v8; // ecx
  __int64 v9; // r10
  unsigned int v10; // edi
  unsigned __int64 v11; // r8
  __int64 v12; // rdx
  int v14; // ecx
  __int64 v15; // r10
  unsigned int v16; // edi
  unsigned __int64 v17; // r8
  __int64 v18; // rdx
  unsigned int i; // r15d
  _BYTE *v23; // rax
  int v24; // edx
  unsigned int v25; // eax
  unsigned int v26; // r8d
  unsigned int v27; // esi
  __int64 v28; // r9
  int v29; // ecx
  unsigned __int64 v30; // rdi
  int v31; // ecx
  __int64 v32; // rdx
  unsigned __int64 v33; // rdi
  unsigned int j; // r15d
  _BYTE *v38; // rax
  int v39; // edx
  unsigned int v40; // eax
  unsigned int v41; // r9d
  unsigned int v42; // esi
  __int64 v43; // r10
  int v44; // ecx
  unsigned __int64 v45; // r8
  int v46; // ecx
  __int64 v47; // rdx
  unsigned __int64 v48; // r8
  __int64 v51; // rax
  int v52[14]; // [rsp+8h] [rbp-38h] BYREF

  v6 = sub_1263B40(a2, "RP: ");
  *(_QWORD *)v52 = *(_QWORD *)a3;
  sub_1DF8250(v52, v6);
  v7 = sub_1263B40(a2, " Live-in RP: ");
  *(_QWORD *)v52 = *(_QWORD *)(a3 + 8);
  sub_1DF8250(v52, v7);
  if ( *(_BYTE *)(a1 + 48) )
  {
    v51 = sub_1263B40(a2, " Register Target: ");
    *(_QWORD *)v52 = *(_QWORD *)(a3 + 16);
    sub_1DF8250(v52, v51);
  }
  sub_1263B40(a2, "\n");
  sub_1263B40(a2, "Live-in values begin\n");
  v8 = *(_DWORD *)(a3 + 40);
  if ( v8 )
  {
    v9 = *(_QWORD *)(a3 + 24);
    v10 = (unsigned int)(v8 - 1) >> 6;
    v11 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v8;
    v12 = 0;
    while ( 1 )
    {
      _RCX = *(_QWORD *)(v9 + 8 * v12);
      if ( v10 == (_DWORD)v12 )
        _RCX = v11 & *(_QWORD *)(v9 + 8 * v12);
      if ( _RCX )
        break;
      if ( v10 + 1 == ++v12 )
        goto LABEL_9;
    }
    __asm { tzcnt   rcx, rcx }
    for ( i = ((_DWORD)v12 << 6) + _RCX; i != -1; i = _RAX + ((_DWORD)v32 << 6) )
    {
      sub_155C2B0(*(_QWORD *)(*(_QWORD *)(a1 + 88) + 8LL * i), a2, 0);
      v23 = *(_BYTE **)(a2 + 24);
      if ( *(_BYTE **)(a2 + 16) == v23 )
      {
        sub_16E7EE0(a2, "\n", 1u);
      }
      else
      {
        *v23 = 10;
        ++*(_QWORD *)(a2 + 24);
      }
      v24 = *(_DWORD *)(a3 + 40);
      v25 = i + 1;
      if ( v24 == i + 1 )
        break;
      v26 = v25 >> 6;
      v27 = (unsigned int)(v24 - 1) >> 6;
      if ( v25 >> 6 > v27 )
        break;
      v28 = *(_QWORD *)(a3 + 24);
      v29 = 64 - (v25 & 0x3F);
      v30 = 0xFFFFFFFFFFFFFFFFLL >> v29;
      if ( v29 == 64 )
        v30 = 0;
      v31 = -v24;
      v32 = v26;
      v33 = ~v30;
      while ( 1 )
      {
        _RAX = *(_QWORD *)(v28 + 8 * v32);
        if ( v26 == (_DWORD)v32 )
          _RAX = v33 & *(_QWORD *)(v28 + 8 * v32);
        if ( v27 == (_DWORD)v32 )
          _RAX &= 0xFFFFFFFFFFFFFFFFLL >> v31;
        if ( _RAX )
          break;
        if ( v27 < (unsigned int)++v32 )
          goto LABEL_9;
      }
      __asm { tzcnt   rax, rax }
    }
  }
LABEL_9:
  sub_1263B40(a2, "Live-in values end\n");
  sub_1263B40(a2, "Live-out values begin\n");
  v14 = *(_DWORD *)(a3 + 64);
  if ( v14 )
  {
    v15 = *(_QWORD *)(a3 + 48);
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
      if ( v16 + 1 == ++v18 )
        return sub_1263B40(a2, "Live-out values end\n");
    }
    __asm { tzcnt   rcx, rcx }
    for ( j = ((_DWORD)v18 << 6) + _RCX; j != -1; j = ((_DWORD)v47 << 6) + _RAX )
    {
      sub_155C2B0(*(_QWORD *)(*(_QWORD *)(a1 + 88) + 8LL * j), a2, 0);
      v38 = *(_BYTE **)(a2 + 24);
      if ( *(_BYTE **)(a2 + 16) == v38 )
      {
        sub_16E7EE0(a2, "\n", 1u);
      }
      else
      {
        *v38 = 10;
        ++*(_QWORD *)(a2 + 24);
      }
      v39 = *(_DWORD *)(a3 + 64);
      v40 = j + 1;
      if ( v39 == j + 1 )
        break;
      v41 = v40 >> 6;
      v42 = (unsigned int)(v39 - 1) >> 6;
      if ( v40 >> 6 > v42 )
        break;
      v43 = *(_QWORD *)(a3 + 48);
      v44 = 64 - (v40 & 0x3F);
      v45 = 0xFFFFFFFFFFFFFFFFLL >> v44;
      if ( v44 == 64 )
        v45 = 0;
      v46 = -v39;
      v47 = v41;
      v48 = ~v45;
      while ( 1 )
      {
        _RAX = *(_QWORD *)(v43 + 8 * v47);
        if ( v41 == (_DWORD)v47 )
          _RAX = v48 & *(_QWORD *)(v43 + 8 * v47);
        if ( v42 == (_DWORD)v47 )
          _RAX &= 0xFFFFFFFFFFFFFFFFLL >> v46;
        if ( _RAX )
          break;
        if ( v42 < (unsigned int)++v47 )
          return sub_1263B40(a2, "Live-out values end\n");
      }
      __asm { tzcnt   rax, rax }
    }
  }
  return sub_1263B40(a2, "Live-out values end\n");
}
