// Function: sub_DFB770
// Address: 0xdfb770
//
__int64 __fastcall sub_DFB770(unsigned __int8 *a1)
{
  char v1; // al
  char v2; // bl
  _BYTE *v3; // rax
  _BYTE *v4; // r12
  unsigned __int8 v5; // dl
  __int64 v6; // rax
  __int64 v8; // rax
  unsigned int v9; // ebx
  int v11; // r12d
  unsigned __int64 v12; // rax
  int v14; // eax
  __int64 v15; // r13
  unsigned int v16; // r13d
  int v18; // edx
  int v19; // r12d
  unsigned int v20; // r14d
  char v21; // r13
  unsigned __int8 v22; // bl
  __int64 v23; // rax
  unsigned int v24; // esi
  __int64 v25; // r9
  int v27; // edx
  unsigned __int64 v28; // rdx
  char v30; // al
  unsigned int v31; // edx
  int v32; // ebx
  int v34; // eax
  unsigned __int64 v35; // rax
  __int64 v36; // [rsp+10h] [rbp-40h]
  __int64 v37; // [rsp+10h] [rbp-40h]
  __int64 v38; // [rsp+18h] [rbp-38h]
  int v39; // [rsp+18h] [rbp-38h]

  v1 = *a1;
  if ( *a1 == 17 )
  {
    v9 = *((_DWORD *)a1 + 8);
    if ( v9 > 0x40 )
    {
      v15 = (__int64)(a1 + 24);
      v8 = sub_C44630((__int64)(a1 + 24));
      if ( (_DWORD)v8 == 1 )
        return (v8 << 32) | 2;
      v8 = 0;
      if ( (*(_QWORD *)(*((_QWORD *)a1 + 3) + 8LL * ((v9 - 1) >> 6)) & (1LL << ((unsigned __int8)v9 - 1))) == 0 )
        return (v8 << 32) | 2;
      v11 = sub_C44500(v15);
      v14 = sub_C44590(v15);
    }
    else
    {
      _RDX = *((_QWORD *)a1 + 3);
      v8 = 0;
      if ( !_RDX )
        return (v8 << 32) | 2;
      v8 = 1;
      if ( (_RDX & (_RDX - 1)) == 0 )
        return (v8 << 32) | 2;
      if ( !_bittest64(&_RDX, v9 - 1) )
        goto LABEL_9;
      if ( v9 )
      {
        v11 = 64;
        if ( _RDX << (64 - (unsigned __int8)v9) != -1 )
        {
          _BitScanReverse64(&v12, ~(_RDX << (64 - (unsigned __int8)v9)));
          v11 = v12 ^ 0x3F;
        }
      }
      else
      {
        v11 = 0;
      }
      __asm { tzcnt   rdx, rdx }
      v14 = _RDX;
      if ( (unsigned int)_RDX > v9 )
        v14 = *((_DWORD *)a1 + 8);
    }
    v8 = 2 * (unsigned int)(v9 == v11 + v14);
    return (v8 << 32) | 2;
  }
  if ( v1 != 18 )
  {
    v2 = 0;
    if ( v1 == 92 )
    {
      v31 = *((_DWORD *)a1 + 20);
      v2 = *(_DWORD *)(*(_QWORD *)(*((_QWORD *)a1 - 8) + 8LL) + 32LL) == v31
        && (unsigned __int8)sub_B4EE20(*((int **)a1 + 9), v31, v31) != 0;
    }
    v3 = (_BYTE *)sub_9B7920((char *)a1);
    v4 = v3;
    if ( v3 )
    {
      v5 = *v3;
      v6 = 0;
      if ( v5 == 22 || v5 <= 3u )
      {
        v2 = 1;
        return (v6 << 32) | v2 & 3;
      }
      if ( v5 > 0x15u )
        return (v6 << 32) | v2 & 3;
      v2 = 2;
      if ( v5 != 17 )
        return (v6 << 32) | v2 & 3;
      v16 = *((_DWORD *)v4 + 8);
      if ( v16 > 0x40 )
      {
        v2 = 2;
        v6 = sub_C44630((__int64)(v4 + 24));
        if ( (_DWORD)v6 == 1 )
          return (v6 << 32) | v2 & 3;
        v6 = 0;
        if ( (*(_QWORD *)(*((_QWORD *)v4 + 3) + 8LL * ((v16 - 1) >> 6)) & (1LL << ((unsigned __int8)v16 - 1))) == 0 )
          return (v6 << 32) | v2 & 3;
        v32 = sub_C44500((__int64)(v4 + 24));
        LODWORD(_RAX) = sub_C44590((__int64)(v4 + 24));
      }
      else
      {
        _RDX = *((_QWORD *)v4 + 3);
        if ( !_RDX )
          return (v6 << 32) | v2 & 3;
        if ( (_RDX & (_RDX - 1)) == 0 )
        {
          v6 = 1;
          v2 = 2;
          return (v6 << 32) | v2 & 3;
        }
        if ( !_bittest64(&_RDX, v16 - 1) )
        {
          v2 = 2;
          return (v6 << 32) | v2 & 3;
        }
        if ( v16 )
        {
          v32 = 64;
          if ( _RDX << (64 - (unsigned __int8)v16) != -1 )
          {
            _BitScanReverse64(&v35, ~(_RDX << (64 - (unsigned __int8)v16)));
            v32 = v35 ^ 0x3F;
          }
        }
        else
        {
          v32 = 0;
        }
        __asm { tzcnt   rax, rdx }
        if ( (unsigned int)_RAX > v16 )
          LODWORD(_RAX) = *((_DWORD *)v4 + 8);
      }
      v34 = v32 + _RAX;
      v2 = 2;
      v6 = 2 * (unsigned int)(v16 == v34);
      return (v6 << 32) | v2 & 3;
    }
    v18 = *a1;
    if ( (unsigned int)(v18 - 15) > 1 )
    {
      v6 = 0;
      if ( (_BYTE)v18 == 11 )
      {
LABEL_50:
        v6 = 0;
        v2 = 3;
      }
      return (v6 << 32) | v2 & 3;
    }
    v19 = sub_AC5290((__int64)a1);
    if ( !v19 )
    {
LABEL_48:
      v6 = 2;
      v2 = 3;
      return (v6 << 32) | v2 & 3;
    }
    v20 = 0;
    v21 = 1;
    v22 = 1;
    while ( 1 )
    {
      v23 = sub_AD68C0((__int64)a1, v20);
      if ( *(_BYTE *)v23 != 17 )
        goto LABEL_50;
      v24 = *(_DWORD *)(v23 + 32);
      v25 = 1LL << ((unsigned __int8)v24 - 1);
      if ( v24 > 0x40 )
      {
        v38 = v23 + 24;
        v36 = v23;
        v22 &= (unsigned int)sub_C44630(v23 + 24) == 1;
        if ( (*(_QWORD *)(*(_QWORD *)(v36 + 24) + 8LL * ((v24 - 1) >> 6)) & (1LL << ((unsigned __int8)v24 - 1))) == 0 )
        {
LABEL_63:
          v30 = v22;
          v21 = 0;
          goto LABEL_45;
        }
        v37 = v38;
        v39 = sub_C44500(v38);
        LODWORD(_RAX) = sub_C44590(v37);
        v27 = v39;
      }
      else
      {
        _RAX = *(_QWORD *)(v23 + 24);
        if ( !_RAX )
          goto LABEL_50;
        if ( (_RAX & (_RAX - 1)) != 0 )
        {
          if ( (_RAX & v25) == 0 )
            goto LABEL_50;
          v22 = 0;
        }
        else if ( (_RAX & v25) == 0 )
        {
          goto LABEL_63;
        }
        if ( v24 )
        {
          v27 = 64;
          if ( _RAX << (64 - (unsigned __int8)v24) != -1 )
          {
            _BitScanReverse64(&v28, ~(_RAX << (64 - (unsigned __int8)v24)));
            v27 = v28 ^ 0x3F;
          }
        }
        else
        {
          v27 = 0;
        }
        __asm { tzcnt   rax, rax }
        if ( (unsigned int)_RAX > v24 )
          LODWORD(_RAX) = v24;
      }
      v21 &= v27 + (_DWORD)_RAX == v24;
      v30 = v22 | v21;
LABEL_45:
      if ( !v30 )
        goto LABEL_50;
      if ( v19 == ++v20 )
      {
        if ( v21 )
          goto LABEL_48;
        v6 = v22;
        v2 = 3;
        return (v6 << 32) | v2 & 3;
      }
    }
  }
LABEL_9:
  v8 = 0;
  return (v8 << 32) | 2;
}
