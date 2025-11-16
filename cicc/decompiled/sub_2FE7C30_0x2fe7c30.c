// Function: sub_2FE7C30
// Address: 0x2fe7c30
//
_QWORD *__fastcall sub_2FE7C30(__int64 a1, __int64 a2, unsigned __int16 a3)
{
  _QWORD *v3; // r14
  __int64 v6; // r8
  int v7; // r12d
  int v8; // ebx
  __int64 v9; // r9
  size_t v10; // rdx
  _BYTE *v11; // rdi
  _WORD *v12; // rdx
  __int64 v13; // r9
  _WORD *v14; // rax
  __int64 v15; // rsi
  _WORD *v16; // r8
  unsigned int v17; // ecx
  unsigned int v18; // ebx
  __int64 v19; // rdx
  __int64 v20; // r11
  __int64 v21; // rax
  _QWORD *v22; // rdi
  __int64 v23; // rcx
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // r11
  int v27; // ecx
  __int64 v28; // rbx
  unsigned __int64 v29; // rdx
  unsigned int v30; // esi
  unsigned int i; // r12d
  __int64 v35; // rcx
  __int64 v36; // rbx
  int v37; // eax
  unsigned int v38; // eax
  unsigned int v39; // r10d
  unsigned int v40; // esi
  int v41; // ecx
  unsigned __int64 v42; // r9
  __int64 v43; // rdx
  unsigned __int64 v44; // r9
  unsigned int v47; // [rsp+Ch] [rbp-94h]
  unsigned int v48; // [rsp+10h] [rbp-90h]
  __int64 v49; // [rsp+10h] [rbp-90h]
  void *v50; // [rsp+20h] [rbp-80h] BYREF
  __int64 v51; // [rsp+28h] [rbp-78h]
  _BYTE v52[48]; // [rsp+30h] [rbp-70h] BYREF
  int v53; // [rsp+60h] [rbp-40h]

  v3 = *(_QWORD **)(a1 + 8LL * a3 + 112);
  if ( v3 )
  {
    v6 = (__int64)(*(_QWORD *)(a2 + 288) - *(_QWORD *)(a2 + 280)) >> 3;
    v50 = v52;
    v7 = v6;
    v8 = v6;
    v9 = (unsigned int)(v6 + 63) >> 6;
    v51 = 0x600000000LL;
    if ( (unsigned int)v9 > 6 )
    {
      v47 = (unsigned int)(v6 + 63) >> 6;
      v49 = (unsigned int)v9;
      sub_C8D5F0((__int64)&v50, v52, (unsigned int)v9, 8u, v6, v9);
      memset(v50, 0, 8 * v49);
      v11 = v50;
      LODWORD(v51) = v47;
      v9 = (__int64)(*(_QWORD *)(a2 + 288) - *(_QWORD *)(a2 + 280)) >> 3;
    }
    else
    {
      if ( (_DWORD)v9 )
      {
        v10 = 8LL * (unsigned int)v9;
        if ( v10 )
        {
          v48 = (unsigned int)(v6 + 63) >> 6;
          memset(v52, 0, v10);
          LODWORD(v9) = v48;
        }
      }
      v11 = v52;
      LODWORD(v51) = v9;
      LODWORD(v9) = v7;
    }
    v53 = v7;
    v12 = (_WORD *)v3[2];
    v13 = 4LL * ((unsigned int)(v9 + 31) >> 5);
    v14 = v12 + 1;
    v15 = v13 + v3[1];
    if ( !*v12 )
      v14 = 0;
    while ( 1 )
    {
      v16 = v14;
      if ( !v14 )
        break;
      while ( 1 )
      {
        v17 = v8 + 31;
        v18 = (unsigned int)(v8 + 31) >> 5;
        if ( v17 <= 0x3F )
        {
          v24 = v15;
          v20 = 0;
        }
        else
        {
          v19 = 0;
          v20 = ((v18 - 2) >> 1) + 1;
          v21 = 8 * v20;
          while ( 1 )
          {
            v22 = &v11[v19];
            v23 = *(_QWORD *)(v15 + v19) | *v22;
            v19 += 8;
            *v22 = v23;
            if ( v21 == v19 )
              break;
            v11 = v50;
          }
          v11 = v50;
          v24 = v15 + v21;
          v18 &= 1u;
        }
        if ( v18 )
        {
          v25 = v24 + 4;
          v26 = 8 * v20;
          v27 = 0;
          v28 = v25;
          while ( 1 )
          {
            v29 = (unsigned __int64)*(unsigned int *)(v25 - 4) << v27;
            v27 += 32;
            *(_QWORD *)&v11[v26] |= v29;
            v11 = v50;
            if ( v25 == v28 )
              break;
            v25 += 4;
          }
        }
        v8 = v53;
        if ( (v53 & 0x3F) != 0 )
        {
          *(_QWORD *)&v11[8 * (unsigned int)v51 - 8] &= ~(-1LL << (v53 & 0x3F));
          v8 = v53;
          v11 = v50;
        }
        ++v16;
        v15 += v13;
        v14 = 0;
        if ( !*(v16 - 1) )
          break;
        if ( !v16 )
          goto LABEL_23;
      }
    }
LABEL_23:
    if ( v8 )
    {
      v30 = (unsigned int)(v8 - 1) >> 6;
      while ( 1 )
      {
        _RDX = *(_QWORD *)&v11[8 * (_QWORD)v16];
        if ( v30 == (_DWORD)v16 )
          _RDX = (0xFFFFFFFFFFFFFFFFLL >> -(char)v8) & *(_QWORD *)&v11[8 * (_QWORD)v16];
        if ( _RDX )
          break;
        v16 = (_WORD *)((char *)v16 + 1);
        if ( (_WORD *)(v30 + 1) == v16 )
          goto LABEL_29;
      }
      __asm { tzcnt   rdx, rdx }
      for ( i = ((_DWORD)v16 << 6) + _RDX; i != -1; i = ((_DWORD)v43 << 6) + _RAX )
      {
        v35 = *(_QWORD *)(a2 + 280);
        v36 = *(_QWORD *)(v35 + 8LL * i);
        v37 = *(_DWORD *)(a2 + 328) * ((*(_QWORD *)(a2 + 288) - v35) >> 3);
        if ( *(_DWORD *)(*(_QWORD *)(a2 + 312) + 16LL * (v37 + (unsigned int)*(unsigned __int16 *)(*v3 + 24LL)) + 4) >> 3 < *(_DWORD *)(*(_QWORD *)(a2 + 312) + 16LL * ((unsigned int)*(unsigned __int16 *)(*(_QWORD *)v36 + 24LL) + v37) + 4) >> 3
          && (unsigned __int8)sub_2FE7BB0(a1, a2, v36) )
        {
          v3 = (_QWORD *)v36;
        }
        v38 = i + 1;
        v11 = v50;
        if ( v53 == i + 1 )
          break;
        v39 = v38 >> 6;
        v40 = (unsigned int)(v53 - 1) >> 6;
        if ( v38 >> 6 > v40 )
          break;
        v41 = 64 - (v38 & 0x3F);
        v42 = 0xFFFFFFFFFFFFFFFFLL >> v41;
        v43 = v39;
        if ( v41 == 64 )
          v42 = 0;
        v44 = ~v42;
        while ( 1 )
        {
          _RAX = *((_QWORD *)v50 + v43);
          if ( v39 == (_DWORD)v43 )
            _RAX = v44 & *((_QWORD *)v50 + v43);
          if ( (_DWORD)v43 == v40 )
            _RAX &= 0xFFFFFFFFFFFFFFFFLL >> -(char)v53;
          if ( _RAX )
            break;
          if ( v40 < (unsigned int)++v43 )
            goto LABEL_29;
        }
        __asm { tzcnt   rax, rax }
      }
    }
LABEL_29:
    if ( v11 != v52 )
      _libc_free((unsigned __int64)v11);
  }
  return v3;
}
