// Function: sub_2B2C810
// Address: 0x2b2c810
//
unsigned __int64 __fastcall sub_2B2C810(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v3; // rax
  _BOOL8 v4; // rbx
  __int64 v5; // r12
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rax
  bool v9; // of
  __int64 v10; // rax
  unsigned __int64 v13; // r13
  __int64 v15; // rax
  _QWORD *v16; // r15
  __int64 v17; // rbx
  __int64 v18; // rax
  __int64 v19; // rbx
  _QWORD *v20; // rbx
  _BYTE *v21; // rdi
  unsigned int v22; // eax
  __int64 v23; // rdi
  _QWORD *v24; // r14
  __int64 v27; // rdi
  __int64 v30; // rdi
  _BYTE *v33; // rax
  _BYTE *v36; // rax
  _BYTE *v38; // rax
  __int64 v43; // [rsp+20h] [rbp-70h]
  __int64 v44; // [rsp+28h] [rbp-68h]
  __int64 v45; // [rsp+30h] [rbp-60h]
  __int64 v46; // [rsp+40h] [rbp-50h]
  _QWORD *v47; // [rsp+48h] [rbp-48h]
  __int64 v48; // [rsp+50h] [rbp-40h]
  __int64 v49; // [rsp+58h] [rbp-38h]

  v2 = a1;
  v45 = a2;
  v44 = *(_QWORD *)(a1 + 40);
  v3 = *(_QWORD *)(a1 + 48);
  v46 = *(_QWORD *)(v3 + 240);
  if ( *(_DWORD *)a1 != 28 )
    goto LABEL_2;
  v49 = *(_QWORD *)(a1 + 24);
  if ( v49 == *(_QWORD *)(v44 + 3528) + 24LL * *(unsigned int *)(v44 + 3544) )
    goto LABEL_2;
  v43 = *(unsigned int *)(v3 + 248);
  if ( !*(_DWORD *)(v3 + 248) )
    goto LABEL_2;
  v48 = 0;
  do
  {
    v15 = v46 + 80LL * (unsigned int)v48;
    v16 = *(_QWORD **)v15;
    v17 = 8LL * *(unsigned int *)(v15 + 8);
    v47 = (_QWORD *)(*(_QWORD *)v15 + v17);
    v18 = v17 >> 3;
    v19 = v17 >> 5;
    if ( v19 )
    {
      v20 = &v16[4 * v19];
      while ( 1 )
      {
        v21 = (_BYTE *)*v16;
        if ( *(_BYTE *)*v16 != 17 )
          goto LABEL_11;
        if ( *((_DWORD *)v21 + 8) <= 0x40u )
        {
          _RDX = ~*((_QWORD *)v21 + 3);
          v13 = *(_QWORD *)(v49 + 8);
          __asm { tzcnt   rax, rdx }
          _RAX = (int)_RAX;
          if ( *((_QWORD *)v21 + 3) == -1 )
            _RAX = 64;
          if ( v13 > _RAX )
            goto LABEL_11;
        }
        else
        {
          v22 = sub_C445E0((__int64)(v21 + 24));
          v13 = *(_QWORD *)(v49 + 8);
          if ( v13 > v22 )
            goto LABEL_11;
        }
        v23 = v16[1];
        v24 = v16 + 1;
        if ( *(_BYTE *)v23 != 17 )
          goto LABEL_44;
        if ( *(_DWORD *)(v23 + 32) > 0x40u )
        {
          _RAX = (unsigned int)sub_C445E0(v23 + 24);
        }
        else
        {
          _RDX = ~*(_QWORD *)(v23 + 24);
          __asm { tzcnt   rax, rdx }
          _RAX = (int)_RAX;
          if ( *(_QWORD *)(v23 + 24) == -1 )
            _RAX = 64;
        }
        if ( v13 > _RAX )
          goto LABEL_44;
        v27 = v16[2];
        v24 = v16 + 2;
        if ( *(_BYTE *)v27 != 17 )
          goto LABEL_44;
        if ( *(_DWORD *)(v27 + 32) > 0x40u )
        {
          _RAX = (unsigned int)sub_C445E0(v27 + 24);
        }
        else
        {
          _RDX = ~*(_QWORD *)(v27 + 24);
          __asm { tzcnt   rax, rdx }
          _RAX = (int)_RAX;
          if ( *(_QWORD *)(v27 + 24) == -1 )
            _RAX = 64;
        }
        if ( v13 > _RAX )
          goto LABEL_44;
        v30 = v16[3];
        v24 = v16 + 3;
        if ( *(_BYTE *)v30 != 17 )
          goto LABEL_44;
        if ( *(_DWORD *)(v30 + 32) > 0x40u )
        {
          _RAX = (unsigned int)sub_C445E0(v30 + 24);
        }
        else
        {
          _RDX = ~*(_QWORD *)(v30 + 24);
          __asm { tzcnt   rax, rdx }
          _RAX = (int)_RAX;
          if ( *(_QWORD *)(v30 + 24) == -1 )
            _RAX = 64;
        }
        if ( v13 > _RAX )
        {
LABEL_44:
          v16 = v24;
          goto LABEL_11;
        }
        v16 += 4;
        if ( v20 == v16 )
        {
          v18 = v47 - v16;
          break;
        }
      }
    }
    if ( v18 != 2 )
    {
      if ( v18 != 3 )
      {
        if ( v18 != 1 )
          return v45;
        v33 = (_BYTE *)*v16;
        if ( *(_BYTE *)*v16 != 17 )
          goto LABEL_11;
        goto LABEL_39;
      }
      v36 = (_BYTE *)*v16;
      if ( *(_BYTE *)*v16 != 17 )
        goto LABEL_11;
      if ( *((_DWORD *)v36 + 8) <= 0x40u )
      {
        _RCX = ~*((_QWORD *)v36 + 3);
        __asm { tzcnt   rax, rcx }
        _RAX = (int)_RAX;
        if ( !_RCX )
          _RAX = 64;
      }
      else
      {
        _RAX = (unsigned int)sub_C445E0((__int64)(v36 + 24));
      }
      if ( _RAX < *(_QWORD *)(v49 + 8) )
        goto LABEL_11;
      ++v16;
    }
    v38 = (_BYTE *)*v16;
    if ( *(_BYTE *)*v16 != 17 )
      goto LABEL_11;
    if ( *((_DWORD *)v38 + 8) > 0x40u )
    {
      _RAX = (unsigned int)sub_C445E0((__int64)(v38 + 24));
    }
    else
    {
      _RCX = ~*((_QWORD *)v38 + 3);
      __asm { tzcnt   rax, rcx }
      _RAX = (int)_RAX;
      if ( !_RCX )
        _RAX = 64;
    }
    if ( _RAX < *(_QWORD *)(v49 + 8) )
      goto LABEL_11;
    v33 = (_BYTE *)v16[1];
    ++v16;
    if ( *v33 != 17 )
      goto LABEL_11;
LABEL_39:
    if ( *((_DWORD *)v33 + 8) > 0x40u )
    {
      _RAX = (unsigned int)sub_C445E0((__int64)(v33 + 24));
    }
    else
    {
      _RCX = ~*((_QWORD *)v33 + 3);
      __asm { tzcnt   rax, rcx }
      _RAX = (int)_RAX;
      if ( !_RCX )
        _RAX = 64;
    }
    if ( _RAX >= *(_QWORD *)(v49 + 8) )
      return v45;
LABEL_11:
    if ( v47 == v16 )
      return v45;
    ++v48;
  }
  while ( v43 != v48 );
  v2 = a1;
LABEL_2:
  v4 = **(_BYTE **)(v2 + 56) != 41;
  v5 = sub_2B2BBE0(v44, *(char **)v46, *(unsigned int *)(v46 + 8));
  v6 = *(_QWORD *)(*(_QWORD *)(v2 + 48) + 240LL) + 80 * v4;
  v7 = sub_2B2BBE0(*(_QWORD *)(v2 + 40), *(char **)v6, *(unsigned int *)(v6 + 8));
  v8 = sub_DFD800(
         *(_QWORD *)(*(_QWORD *)(v2 + 40) + 3296LL),
         *(_DWORD *)v2,
         *(_QWORD *)(v2 + 64),
         *(_DWORD *)(v2 + 72),
         v5,
         v7,
         0,
         0,
         0,
         *(__int64 **)(*(_QWORD *)(v2 + 40) + 3304LL));
  v9 = __OFADD__(a2, v8);
  v10 = a2 + v8;
  if ( v9 )
  {
    v10 = 0x7FFFFFFFFFFFFFFFLL;
    if ( a2 <= 0 )
      return 0x8000000000000000LL;
  }
  return v10;
}
