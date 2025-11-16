// Function: sub_2C71280
// Address: 0x2c71280
//
__int64 __fastcall sub_2C71280(__int64 a1, __int64 **a2)
{
  __int64 *v4; // rbx
  __int64 *v5; // rsi
  int v6; // r15d
  __int64 v7; // rdx
  unsigned int v8; // r8d
  __int64 v9; // r10
  unsigned int v13; // eax
  __int64 **v14; // rdi
  __int64 **v15; // r12
  __int64 *v16; // rdx
  unsigned int v17; // eax
  unsigned int v18; // r9d
  unsigned int v19; // esi
  __int64 v20; // r10
  __int64 v21; // rdx
  int v22; // ecx
  unsigned __int64 v23; // r8
  unsigned __int64 v24; // r8
  __int64 **v27; // r14
  __int64 v28; // rbx
  unsigned __int64 v29; // rax
  __int64 **v30; // r13
  __int64 *v31; // r14
  const char *v32; // rax
  size_t v33; // rdx
  size_t v34; // rbx
  const char *v35; // r15
  const char *v36; // rax
  size_t v37; // rdx
  size_t v38; // r12
  bool v39; // cc
  size_t v40; // rdx
  int v41; // eax
  __int64 *v42; // rax
  __int64 v43; // rdx
  unsigned int v44; // edi
  __int64 v45; // r10
  unsigned int v48; // r12d
  _BYTE *v49; // rax
  __int64 v50; // rdi
  int v51; // edx
  unsigned int v52; // eax
  unsigned int v53; // r9d
  unsigned int v54; // esi
  __int64 v55; // r10
  int v56; // ecx
  unsigned __int64 v57; // r8
  __int64 v58; // rdx
  unsigned __int64 v59; // r8
  __int64 **v62; // rbx
  __int64 v63; // rdi
  __int64 *v64; // r14
  _BYTE *v65; // rax
  __int64 v66; // [rsp+8h] [rbp-78h]
  __int64 **v67; // [rsp+10h] [rbp-70h]
  __int64 **v68; // [rsp+18h] [rbp-68h]
  unsigned int v69; // [rsp+18h] [rbp-68h]
  __int64 *v70; // [rsp+28h] [rbp-58h] BYREF
  __int64 **v71; // [rsp+30h] [rbp-50h] BYREF
  __int64 **v72; // [rsp+38h] [rbp-48h]
  __int64 **v73; // [rsp+40h] [rbp-40h]

  v4 = *a2;
  v5 = a2[1];
  v6 = *((_DWORD *)v4 + 16);
  if ( v5 )
  {
    v71 = 0;
    v72 = 0;
    v73 = 0;
    if ( v6 )
    {
      v7 = 0;
      v8 = (unsigned int)(v6 - 1) >> 6;
      v9 = *v4;
      while ( 1 )
      {
        _RCX = *(_QWORD *)(v9 + 8 * v7);
        if ( v8 == (_DWORD)v7 )
          _RCX = (0xFFFFFFFFFFFFFFFFLL >> -(char)v6) & *(_QWORD *)(v9 + 8 * v7);
        if ( _RCX )
          break;
        if ( v8 + 1 == ++v7 )
          return a1;
      }
      __asm { tzcnt   rcx, rcx }
      v13 = _RCX + ((_DWORD)v7 << 6);
      if ( v13 != -1 )
      {
        v14 = 0;
        v15 = 0;
        while ( 1 )
        {
          v16 = *(__int64 **)(v5[5] + 8LL * v13);
          v70 = v16;
          if ( v14 == v15 )
          {
            v69 = v13;
            sub_2C6F5C0((__int64)&v71, v15, &v70);
            v15 = v72;
            v6 = *((_DWORD *)v4 + 16);
            v13 = v69;
          }
          else
          {
            if ( v15 )
            {
              *v15 = v16;
              v15 = v72;
              v6 = *((_DWORD *)v4 + 16);
            }
            v72 = ++v15;
          }
          v17 = v13 + 1;
          if ( v6 == v17 )
            break;
          v18 = v17 >> 6;
          v19 = (unsigned int)(v6 - 1) >> 6;
          if ( v17 >> 6 > v19 )
            break;
          v20 = *v4;
          v21 = v18;
          v22 = 64 - (v17 & 0x3F);
          v23 = 0xFFFFFFFFFFFFFFFFLL >> v22;
          if ( v22 == 64 )
            v23 = 0;
          v24 = ~v23;
          while ( 1 )
          {
            _RAX = *(_QWORD *)(v20 + 8 * v21);
            if ( v18 == (_DWORD)v21 )
              _RAX = v24 & *(_QWORD *)(v20 + 8 * v21);
            if ( v19 == (_DWORD)v21 )
              _RAX &= 0xFFFFFFFFFFFFFFFFLL >> -(char)v6;
            if ( _RAX )
              break;
            if ( v19 < (unsigned int)++v21 )
              goto LABEL_28;
          }
          __asm { tzcnt   rax, rax }
          v13 = ((_DWORD)v21 << 6) + _RAX;
          if ( v13 == -1 )
            break;
          v5 = a2[1];
          v14 = v73;
        }
LABEL_28:
        v27 = v71;
        if ( v15 != v71 )
        {
          v28 = (char *)v15 - (char *)v71;
          _BitScanReverse64(&v29, v15 - v71);
          sub_2C71050(v71, v15, 2LL * (int)(63 - (v29 ^ 0x3F)));
          if ( v28 <= 128 )
          {
            sub_2C6F750(v27, v15);
          }
          else
          {
            sub_2C6F750(v27, v27 + 16);
            v68 = v27 + 16;
            if ( v27 + 16 != v15 )
            {
              v67 = v15;
              v66 = a1;
              while ( 1 )
              {
                v30 = v68;
                v31 = *v68;
                while ( 1 )
                {
                  v32 = sub_BD5D20(**(v30 - 1));
                  v34 = v33;
                  v35 = v32;
                  v36 = sub_BD5D20(*v31);
                  v38 = v37;
                  v39 = v37 <= v34;
                  v40 = v34;
                  if ( v39 )
                    v40 = v38;
                  if ( !v40 )
                    break;
                  v41 = memcmp(v36, v35, v40);
                  if ( !v41 )
                    break;
                  if ( v41 >= 0 )
                    goto LABEL_32;
LABEL_40:
                  v42 = *--v30;
                  v30[1] = v42;
                }
                if ( v38 != v34 && v38 < v34 )
                  goto LABEL_40;
LABEL_32:
                ++v68;
                *v30 = v31;
                if ( v67 == v68 )
                {
                  a1 = v66;
                  break;
                }
              }
            }
          }
          v62 = v72;
          v15 = v71;
          if ( v72 != v71 )
          {
            do
            {
              v64 = *v15;
              v65 = *(_BYTE **)(a1 + 32);
              if ( *(_BYTE **)(a1 + 24) == v65 )
              {
                v63 = sub_CB6200(a1, (unsigned __int8 *)" ", 1u);
              }
              else
              {
                *v65 = 32;
                v63 = a1;
                ++*(_QWORD *)(a1 + 32);
              }
              ++v15;
              sub_2C6EAD0(v63, v64);
            }
            while ( v62 != v15 );
            v15 = v71;
          }
        }
        if ( v15 )
          j_j___libc_free_0((unsigned __int64)v15);
      }
    }
  }
  else if ( v6 )
  {
    v43 = 0;
    v44 = (unsigned int)(v6 - 1) >> 6;
    v45 = *v4;
    while ( 1 )
    {
      _RCX = *(_QWORD *)(v45 + 8 * v43);
      if ( v44 == (_DWORD)v43 )
        _RCX = (0xFFFFFFFFFFFFFFFFLL >> -(char)v6) & *(_QWORD *)(v45 + 8 * v43);
      if ( _RCX )
        break;
      if ( v44 + 1 == ++v43 )
        return a1;
    }
    __asm { tzcnt   rcx, rcx }
    v48 = ((_DWORD)v43 << 6) + _RCX;
    if ( v48 != -1 )
    {
      v49 = *(_BYTE **)(a1 + 32);
      if ( *(_BYTE **)(a1 + 24) == v49 )
        goto LABEL_68;
LABEL_54:
      *v49 = 32;
      v50 = a1;
      ++*(_QWORD *)(a1 + 32);
      while ( 1 )
      {
        sub_CB59D0(v50, v48);
        v51 = *((_DWORD *)v4 + 16);
        v52 = v48 + 1;
        if ( v51 == v48 + 1 )
          break;
        v53 = v52 >> 6;
        v54 = (unsigned int)(v51 - 1) >> 6;
        if ( v52 >> 6 > v54 )
          break;
        v55 = *v4;
        v56 = 64 - (v52 & 0x3F);
        v57 = 0xFFFFFFFFFFFFFFFFLL >> v56;
        v58 = v53;
        if ( v56 == 64 )
          v57 = 0;
        v59 = ~v57;
        while ( 1 )
        {
          _RAX = *(_QWORD *)(v55 + 8 * v58);
          if ( v53 == (_DWORD)v58 )
            _RAX = v59 & *(_QWORD *)(v55 + 8 * v58);
          if ( (_DWORD)v58 == v54 )
            _RAX &= 0xFFFFFFFFFFFFFFFFLL >> -(char)*((_DWORD *)v4 + 16);
          if ( _RAX )
            break;
          if ( v54 < (unsigned int)++v58 )
            return a1;
        }
        __asm { tzcnt   rax, rax }
        v48 = ((_DWORD)v58 << 6) + _RAX;
        if ( v48 == -1 )
          break;
        v49 = *(_BYTE **)(a1 + 32);
        if ( *(_BYTE **)(a1 + 24) != v49 )
          goto LABEL_54;
LABEL_68:
        v50 = sub_CB6200(a1, (unsigned __int8 *)" ", 1u);
      }
    }
  }
  return a1;
}
