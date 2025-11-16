// Function: sub_DBE140
// Address: 0xdbe140
//
__int64 __fastcall sub_DBE140(__int64 *a1, __int64 a2, unsigned int a3, char a4)
{
  __int16 v5; // ax
  __int64 v6; // rbx
  unsigned int v7; // r12d
  __int64 result; // rax
  int v10; // ebx
  unsigned __int64 v11; // rbx
  __int64 v13; // rdi
  __int64 v14; // rax
  __int64 *v15; // r14
  __int64 v16; // rbx
  __int64 v17; // rax
  __int16 v18; // cx
  __int64 v19; // r13
  unsigned int v20; // ebx
  __int64 v22; // rax
  __int64 *v23; // r13
  __int16 v24; // cx
  __int64 v25; // rbx
  unsigned int v26; // r9d
  int v28; // ebx
  unsigned __int64 v29; // rcx
  __int64 v31; // rax
  __int16 v32; // cx
  __int64 v33; // rbx
  unsigned int v34; // r9d
  int v36; // ebx
  unsigned __int64 v37; // rcx
  __int64 v39; // rax
  __int16 v40; // cx
  __int64 v41; // r15
  unsigned int v42; // ebx
  int v44; // r15d
  unsigned __int64 v45; // rcx
  __int64 v47; // rdi
  int v48; // ebx
  __int64 v49; // rdi
  int v50; // ebx
  int v51; // r15d
  int v52; // r13d
  unsigned __int64 v53; // rcx
  __int64 v55; // rdi
  __int64 *v56; // [rsp+8h] [rbp-68h]
  int v59; // [rsp+20h] [rbp-50h]
  int v60; // [rsp+20h] [rbp-50h]
  __int64 v61; // [rsp+20h] [rbp-50h]
  __int64 *v62; // [rsp+28h] [rbp-48h]
  __int64 *v63; // [rsp+30h] [rbp-40h] BYREF
  char v64; // [rsp+38h] [rbp-38h]

  v5 = *(_WORD *)(a2 + 24);
  if ( !v5 )
  {
    v6 = *(_QWORD *)(a2 + 32);
    v7 = *(_DWORD *)(v6 + 32);
    if ( v7 <= 0x40 )
    {
      _RAX = *(_QWORD *)(v6 + 24);
      if ( _RAX )
      {
        if ( (_RAX & (_RAX - 1)) == 0 )
          return 1;
        if ( a4 && _bittest64(&_RAX, v7 - 1) )
        {
          if ( v7 )
          {
            v10 = 64;
            if ( _RAX << (64 - (unsigned __int8)v7) != -1 )
            {
              _BitScanReverse64(&v11, ~(_RAX << (64 - (unsigned __int8)v7)));
              v10 = v11 ^ 0x3F;
            }
          }
          else
          {
            v10 = 0;
          }
          __asm { tzcnt   rax, rax }
          if ( (unsigned int)_RAX > v7 )
            LODWORD(_RAX) = v7;
LABEL_19:
          if ( v7 == v10 + (_DWORD)_RAX )
            return 1;
        }
      }
    }
    else
    {
      if ( (unsigned int)sub_C44630(v6 + 24) == 1 )
        return 1;
      if ( a4 && (*(_QWORD *)(*(_QWORD *)(v6 + 24) + 8LL * ((v7 - 1) >> 6)) & (1LL << ((unsigned __int8)v7 - 1))) != 0 )
      {
        v13 = v6 + 24;
        v10 = sub_C44500(v6 + 24);
        LODWORD(_RAX) = sub_C44590(v13);
        goto LABEL_19;
      }
    }
    return 0;
  }
  if ( v5 == 1 )
  {
    if ( (unsigned __int8)sub_B2D610(*a1, 96) )
      return 1;
    v5 = *(_WORD *)(a2 + 24);
  }
  if ( v5 != 6 )
    return 0;
  v14 = *(_QWORD *)(a2 + 40);
  v15 = *(__int64 **)(a2 + 32);
  v64 = a4;
  v56 = &v15[v14];
  v63 = a1;
  v14 *= 8;
  v16 = v14 >> 5;
  v17 = v14 >> 3;
  if ( v16 > 0 )
  {
    v62 = &v15[4 * v16];
    while ( 1 )
    {
      v18 = *(_WORD *)(*v15 + 24);
      if ( v18 )
      {
        if ( v18 != 1 || !(unsigned __int8)sub_B2D610(*v63, 96) )
          goto LABEL_28;
      }
      else
      {
        v19 = *(_QWORD *)(*v15 + 32);
        v20 = *(_DWORD *)(v19 + 32);
        if ( v20 > 0x40 )
        {
          if ( (unsigned int)sub_C44630(v19 + 24) == 1 )
            goto LABEL_32;
          if ( !v64
            || (*(_QWORD *)(*(_QWORD *)(v19 + 24) + 8LL * ((v20 - 1) >> 6)) & (1LL << ((unsigned __int8)v20 - 1))) == 0 )
          {
            goto LABEL_28;
          }
          v55 = v19 + 24;
          v52 = sub_C44500(v19 + 24);
          LODWORD(_RAX) = sub_C44590(v55);
        }
        else
        {
          _RAX = *(_QWORD *)(v19 + 24);
          if ( !_RAX )
            goto LABEL_28;
          if ( (_RAX & (_RAX - 1)) == 0 )
            goto LABEL_32;
          if ( !v64 || (_RAX & (1LL << ((unsigned __int8)v20 - 1))) == 0 )
            goto LABEL_28;
          if ( v20 )
          {
            v52 = 64;
            if ( _RAX << (64 - (unsigned __int8)v20) != -1 )
            {
              _BitScanReverse64(&v53, ~(_RAX << (64 - (unsigned __int8)v20)));
              v52 = v53 ^ 0x3F;
            }
          }
          else
          {
            v52 = 0;
          }
          __asm { tzcnt   rax, rax }
          if ( (unsigned int)_RAX > v20 )
            LODWORD(_RAX) = v20;
        }
        if ( v20 != v52 + (_DWORD)_RAX )
          goto LABEL_28;
      }
LABEL_32:
      v22 = v15[1];
      v23 = v15 + 1;
      v24 = *(_WORD *)(v22 + 24);
      if ( v24 )
      {
        if ( v24 != 1 || !(unsigned __int8)sub_B2D610(*v63, 96) )
          goto LABEL_76;
      }
      else
      {
        v25 = *(_QWORD *)(v22 + 32);
        v26 = *(_DWORD *)(v25 + 32);
        if ( v26 > 0x40 )
        {
          v59 = *(_DWORD *)(v25 + 32);
          if ( (unsigned int)sub_C44630(v25 + 24) != 1 )
          {
            if ( !v64 )
              goto LABEL_76;
            if ( (*(_QWORD *)(*(_QWORD *)(v25 + 24) + 8LL * ((unsigned int)(v59 - 1) >> 6))
                & (1LL << ((unsigned __int8)v59 - 1))) == 0 )
              goto LABEL_76;
            v47 = v25 + 24;
            v48 = sub_C44500(v25 + 24);
            if ( v59 != v48 + (unsigned int)sub_C44590(v47) )
              goto LABEL_76;
          }
        }
        else
        {
          _RAX = *(_QWORD *)(v25 + 24);
          if ( !_RAX )
            goto LABEL_76;
          if ( (_RAX & (_RAX - 1)) != 0 )
          {
            if ( !v64 || (_RAX & (1LL << ((unsigned __int8)v26 - 1))) == 0 )
              goto LABEL_76;
            if ( v26 )
            {
              v28 = 64;
              if ( _RAX << (64 - (unsigned __int8)v26) != -1 )
              {
                _BitScanReverse64(&v29, ~(_RAX << (64 - (unsigned __int8)v26)));
                v28 = v29 ^ 0x3F;
              }
            }
            else
            {
              v28 = 0;
            }
            __asm { tzcnt   rax, rax }
            if ( (unsigned int)_RAX > v26 )
              LODWORD(_RAX) = v26;
            if ( v26 != v28 + (_DWORD)_RAX )
            {
LABEL_76:
              v15 = v23;
              goto LABEL_28;
            }
          }
        }
      }
      v31 = v15[2];
      v23 = v15 + 2;
      v32 = *(_WORD *)(v31 + 24);
      if ( v32 )
      {
        if ( v32 != 1 || !(unsigned __int8)sub_B2D610(*v63, 96) )
          goto LABEL_76;
      }
      else
      {
        v33 = *(_QWORD *)(v31 + 32);
        v34 = *(_DWORD *)(v33 + 32);
        if ( v34 > 0x40 )
        {
          v60 = *(_DWORD *)(v33 + 32);
          if ( (unsigned int)sub_C44630(v33 + 24) != 1 )
          {
            if ( !v64 )
              goto LABEL_76;
            if ( (*(_QWORD *)(*(_QWORD *)(v33 + 24) + 8LL * ((unsigned int)(v60 - 1) >> 6))
                & (1LL << ((unsigned __int8)v60 - 1))) == 0 )
              goto LABEL_76;
            v49 = v33 + 24;
            v50 = sub_C44500(v33 + 24);
            if ( v60 != v50 + (unsigned int)sub_C44590(v49) )
              goto LABEL_76;
          }
        }
        else
        {
          _RAX = *(_QWORD *)(v33 + 24);
          if ( !_RAX )
            goto LABEL_76;
          if ( (_RAX & (_RAX - 1)) != 0 )
          {
            if ( !v64 || (_RAX & (1LL << ((unsigned __int8)v34 - 1))) == 0 )
              goto LABEL_76;
            if ( v34 )
            {
              v36 = 64;
              if ( _RAX << (64 - (unsigned __int8)v34) != -1 )
              {
                _BitScanReverse64(&v37, ~(_RAX << (64 - (unsigned __int8)v34)));
                v36 = v37 ^ 0x3F;
              }
            }
            else
            {
              v36 = 0;
            }
            __asm { tzcnt   rax, rax }
            if ( (unsigned int)_RAX > v34 )
              LODWORD(_RAX) = v34;
            if ( v34 != v36 + (_DWORD)_RAX )
              goto LABEL_76;
          }
        }
      }
      v39 = v15[3];
      v23 = v15 + 3;
      v40 = *(_WORD *)(v39 + 24);
      if ( v40 )
      {
        if ( v40 != 1 || !(unsigned __int8)sub_B2D610(*v63, 96) )
          goto LABEL_76;
        goto LABEL_68;
      }
      v41 = *(_QWORD *)(v39 + 32);
      v42 = *(_DWORD *)(v41 + 32);
      if ( v42 > 0x40 )
      {
        v61 = v41 + 24;
        if ( (unsigned int)sub_C44630(v41 + 24) == 1 )
          goto LABEL_68;
        if ( !v64 )
          goto LABEL_76;
        if ( (*(_QWORD *)(*(_QWORD *)(v41 + 24) + 8LL * ((v42 - 1) >> 6)) & (1LL << ((unsigned __int8)v42 - 1))) == 0 )
          goto LABEL_76;
        v51 = sub_C44500(v41 + 24);
        if ( v42 != v51 + (unsigned int)sub_C44590(v61) )
          goto LABEL_76;
        v15 += 4;
        if ( v62 == v15 )
        {
LABEL_69:
          v17 = v56 - v15;
          break;
        }
      }
      else
      {
        _RAX = *(_QWORD *)(v41 + 24);
        if ( !_RAX )
          goto LABEL_76;
        if ( (_RAX & (_RAX - 1)) != 0 )
        {
          if ( !v64 || (_RAX & (1LL << ((unsigned __int8)v42 - 1))) == 0 )
            goto LABEL_76;
          if ( v42 )
          {
            v44 = 64;
            if ( _RAX << (64 - (unsigned __int8)v42) != -1 )
            {
              _BitScanReverse64(&v45, ~(_RAX << (64 - (unsigned __int8)v42)));
              v44 = v45 ^ 0x3F;
            }
          }
          else
          {
            v44 = 0;
          }
          __asm { tzcnt   rax, rax }
          if ( (unsigned int)_RAX > v42 )
            LODWORD(_RAX) = v42;
          if ( v42 != v44 + (_DWORD)_RAX )
            goto LABEL_76;
        }
LABEL_68:
        v15 += 4;
        if ( v62 == v15 )
          goto LABEL_69;
      }
    }
  }
  if ( v17 != 2 )
  {
    if ( v17 != 3 )
    {
      if ( v17 != 1 )
        goto LABEL_29;
      goto LABEL_73;
    }
    if ( !sub_D92AA0((__int64)&v63, *v15) )
      goto LABEL_28;
    ++v15;
  }
  if ( !sub_D92AA0((__int64)&v63, *v15) )
    goto LABEL_28;
  ++v15;
LABEL_73:
  if ( !sub_D92AA0((__int64)&v63, *v15) )
  {
LABEL_28:
    result = 0;
    if ( v56 != v15 )
      return result;
  }
LABEL_29:
  result = a3;
  if ( !(_BYTE)a3 )
    return sub_DBE090((__int64)a1, a2);
  return result;
}
