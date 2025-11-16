// Function: sub_2B23E00
// Address: 0x2b23e00
//
void __fastcall sub_2B23E00(__int64 a1, unsigned int a2)
{
  __int64 v3; // rax
  unsigned int v4; // r10d
  __int64 v5; // r10
  unsigned __int64 v6; // rdx
  unsigned __int64 v7; // r12
  unsigned __int64 v8; // r13
  unsigned __int64 v13; // rcx
  unsigned __int64 v15; // rcx
  _QWORD *v17; // rax
  unsigned __int64 v18; // r8
  _QWORD *v19; // rcx
  unsigned __int64 v20; // r12
  int v21; // ecx
  __int64 v22; // rdi
  __int64 v23; // rax
  int v26; // ecx
  unsigned __int64 v27; // r10
  unsigned int v28; // edi
  unsigned __int64 v29; // r9
  __int64 v30; // rdx
  int v32; // edx
  unsigned int v33; // r11d
  unsigned int v34; // r10d
  unsigned __int64 v35; // r14
  int v36; // ecx
  unsigned __int64 v37; // rsi
  int v38; // ecx
  __int64 v39; // rdx
  unsigned __int64 v40; // rsi
  int v42; // r15d
  unsigned int v43; // r11d
  unsigned int v44; // edi
  unsigned __int64 v45; // r14
  unsigned __int64 v46; // rdx
  unsigned __int64 v47; // rcx
  __int64 v48; // rdx
  unsigned __int64 v52; // [rsp+8h] [rbp-68h]
  unsigned __int64 v53; // [rsp+10h] [rbp-60h]
  unsigned __int64 v54; // [rsp+18h] [rbp-58h]
  __int64 v55; // [rsp+20h] [rbp-50h]
  char v56; // [rsp+2Eh] [rbp-42h]
  char v57; // [rsp+2Fh] [rbp-41h]
  unsigned __int64 v58; // [rsp+30h] [rbp-40h] BYREF
  unsigned __int64 v59[7]; // [rsp+38h] [rbp-38h] BYREF

  sub_B48880((__int64 *)&v58, a2, 1u);
  sub_B48880((__int64 *)v59, a2, 0);
  if ( a2 )
  {
    v3 = 0;
    while ( 1 )
    {
      v4 = *(_DWORD *)(a1 + 4 * v3);
      if ( v4 < a2 )
        break;
      if ( (v59[0] & 1) != 0 )
      {
        v5 = ~(-1LL << (v59[0] >> 58));
        v6 = v5 & (v59[0] >> 1) | (1LL << v3++);
        v59[0] = 2 * ((v59[0] >> 58 << 57) | v5 & v6) + 1;
        if ( a2 == v3 )
          goto LABEL_9;
      }
      else
      {
        *(_QWORD *)(*(_QWORD *)v59[0] + 8LL * ((unsigned int)v3 >> 6)) |= 1LL << v3;
LABEL_5:
        if ( a2 == ++v3 )
          goto LABEL_9;
      }
    }
    if ( (v58 & 1) != 0 )
      v58 = 2 * ((v58 >> 58 << 57) | ~(1LL << v4) & ~(-1LL << (v58 >> 58)) & (v58 >> 1)) + 1;
    else
      *(_QWORD *)(*(_QWORD *)v58 + 8LL * (v4 >> 6)) &= ~(1LL << v4);
    goto LABEL_5;
  }
LABEL_9:
  v7 = v59[0];
  v57 = v59[0] & 1;
  if ( (v59[0] & 1) != 0 )
  {
    if ( ((v59[0] >> 1) & ~(-1LL << (v59[0] >> 58))) != 0 )
      goto LABEL_11;
  }
  else
  {
    v17 = sub_2B0B280(*(_QWORD **)v59[0], *(_QWORD *)v59[0] + 8LL * *(unsigned int *)(v59[0] + 8));
    if ( v19 != v17 )
    {
LABEL_11:
      v8 = v58;
      v56 = v58 & 1;
      if ( (v58 & 1) != 0 )
      {
        if ( (v58 >> 1) & ~(-1LL << (v58 >> 58)) )
        {
          __asm { tzcnt   r8, r8 }
          goto LABEL_14;
        }
      }
      else
      {
        v21 = *(_DWORD *)(v58 + 64);
        if ( v21 )
        {
          v22 = (unsigned int)(v21 - 1) >> 6;
          v23 = 0;
          while ( 1 )
          {
            _RDX = *(_QWORD *)(*(_QWORD *)v58 + 8 * v23);
            if ( v22 == v23 )
              break;
            if ( _RDX )
              goto LABEL_45;
            if ( (_DWORD)v22 + 1 == ++v23 )
              goto LABEL_87;
          }
          _RDX &= 0xFFFFFFFFFFFFFFFFLL >> -(char)v21;
          if ( _RDX )
          {
LABEL_45:
            __asm { tzcnt   rdx, rdx }
            LODWORD(_R8) = _RDX + ((_DWORD)v23 << 6);
            if ( v57 )
            {
LABEL_15:
              v54 = v7 >> 58;
              _RAX = (v7 >> 1) & ~(-1LL << (v7 >> 58));
              v55 = _RAX;
              if ( _RAX )
              {
                __asm { tzcnt   rax, rax }
LABEL_17:
                v52 = v58 >> 58;
                v53 = (v58 >> 1) & ~(-1LL << (v58 >> 58));
                while ( 1 )
                {
                  v13 = (unsigned int)(_R8 + 1);
                  *(_DWORD *)(a1 + 4LL * (int)_RAX) = _R8;
                  if ( !v56 )
                    break;
                  _R8 = v53 & (-1LL << ((unsigned __int8)_R8 + 1));
                  if ( !_R8 || v52 <= v13 )
                    goto LABEL_86;
                  __asm { tzcnt   r8, r8 }
LABEL_22:
                  v15 = (unsigned int)(_RAX + 1);
                  if ( v57 )
                  {
                    _RAX = v55 & (-1LL << ((unsigned __int8)_RAX + 1));
                    if ( !_RAX || v15 >= v54 )
                      goto LABEL_56;
                    __asm { tzcnt   rax, rax }
                  }
                  else
                  {
                    v32 = *(_DWORD *)(v7 + 64);
                    if ( (_DWORD)v15 == v32 )
                      goto LABEL_52;
                    v33 = (unsigned int)v15 >> 6;
                    v34 = (unsigned int)(v32 - 1) >> 6;
                    if ( (unsigned int)v15 >> 6 > v34 )
                      goto LABEL_52;
                    v35 = *(_QWORD *)v7;
                    v36 = 64 - (((_BYTE)_RAX + 1) & 0x3F);
                    v37 = 0xFFFFFFFFFFFFFFFFLL >> v36;
                    if ( v36 == 64 )
                      v37 = 0;
                    v38 = -v32;
                    v39 = v33;
                    v40 = ~v37;
                    while ( 1 )
                    {
                      _RAX = *(_QWORD *)(v35 + 8 * v39);
                      if ( v33 == (_DWORD)v39 )
                        _RAX = v40 & *(_QWORD *)(v35 + 8 * v39);
                      if ( v34 == (_DWORD)v39 )
                        _RAX &= 0xFFFFFFFFFFFFFFFFLL >> v38;
                      if ( _RAX )
                        break;
                      if ( v34 < (unsigned int)++v39 )
                        goto LABEL_52;
                    }
                    __asm { tzcnt   rax, rax }
                    LODWORD(_RAX) = ((_DWORD)v39 << 6) + _RAX;
                    if ( (int)_RAX < 0 )
                      goto LABEL_52;
                  }
                }
                v42 = *(_DWORD *)(v8 + 64);
                if ( (_DWORD)v13 != v42 )
                {
                  v43 = (unsigned int)v13 >> 6;
                  v44 = (unsigned int)(v42 - 1) >> 6;
                  if ( (unsigned int)v13 >> 6 <= v44 )
                  {
                    v45 = *(_QWORD *)v8;
                    v46 = 0xFFFFFFFFFFFFFFFFLL >> (64 - ((_R8 + 1) & 0x3F));
                    if ( (((_BYTE)_R8 + 1) & 0x3F) == 0 )
                      v46 = 0;
                    v47 = ~v46;
                    v48 = v43;
                    while ( 1 )
                    {
                      _RSI = *(_QWORD *)(v45 + 8 * v48);
                      if ( v43 == (_DWORD)v48 )
                        _RSI = v47 & *(_QWORD *)(v45 + 8 * v48);
                      if ( v44 == (_DWORD)v48 )
                        break;
                      if ( _RSI )
                        goto LABEL_85;
                      if ( v44 < (unsigned int)++v48 )
                        goto LABEL_86;
                    }
                    _RSI &= 0xFFFFFFFFFFFFFFFFLL >> -(char)v42;
                    if ( !_RSI )
                      goto LABEL_86;
LABEL_85:
                    __asm { tzcnt   rsi, rsi }
                    LODWORD(_R8) = _RSI + ((_DWORD)v48 << 6);
                    goto LABEL_22;
                  }
                }
LABEL_86:
                LODWORD(_R8) = -1;
                goto LABEL_22;
              }
LABEL_56:
              if ( !v56 && v8 )
              {
                if ( *(_QWORD *)v8 != v8 + 16 )
                  _libc_free(*(_QWORD *)v8);
                j_j___libc_free_0(v8);
              }
              return;
            }
LABEL_46:
            v26 = *(_DWORD *)(v7 + 64);
            if ( v26 )
            {
              v27 = *(_QWORD *)v7;
              v28 = (unsigned int)(v26 - 1) >> 6;
              v29 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v26;
              v30 = 0;
              while ( 1 )
              {
                _RCX = *(_QWORD *)(v27 + 8 * v30);
                if ( v28 == (_DWORD)v30 )
                  _RCX = v29 & *(_QWORD *)(v27 + 8 * v30);
                if ( _RCX )
                  break;
                if ( v28 + 1 == ++v30 )
                  goto LABEL_52;
              }
              __asm { tzcnt   rcx, rcx }
              LODWORD(_RAX) = _RCX + ((_DWORD)v30 << 6);
              if ( (int)_RAX >= 0 )
              {
                v54 = v7 >> 58;
                v55 = (v7 >> 1) & ~(-1LL << (v7 >> 58));
                goto LABEL_17;
              }
            }
LABEL_52:
            if ( v7 )
            {
              if ( *(_QWORD *)v7 != v7 + 16 )
                _libc_free(*(_QWORD *)v7);
              j_j___libc_free_0(v7);
              v8 = v58;
              v56 = v58 & 1;
            }
            goto LABEL_56;
          }
        }
      }
LABEL_87:
      LODWORD(_R8) = -1;
LABEL_14:
      if ( v57 )
        goto LABEL_15;
      goto LABEL_46;
    }
    if ( v7 )
    {
      if ( v18 != v7 + 16 )
        _libc_free(v18);
      j_j___libc_free_0(v7);
    }
  }
  v20 = v58;
  if ( (v58 & 1) == 0 && v58 )
  {
    if ( *(_QWORD *)v58 != v58 + 16 )
      _libc_free(*(_QWORD *)v58);
    j_j___libc_free_0(v20);
  }
}
