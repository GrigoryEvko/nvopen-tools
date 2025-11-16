// Function: sub_158D430
// Address: 0x158d430
//
__int64 __fastcall sub_158D430(__int64 a1, __int64 a2, unsigned int a3)
{
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // rcx
  unsigned int v9; // eax
  unsigned __int64 v10; // rdx
  unsigned int v11; // r14d
  unsigned __int64 v12; // rdx
  int v13; // eax
  unsigned int v14; // ebx
  unsigned __int64 v15; // rdx
  unsigned int v16; // eax
  __int64 v17; // rdx
  char v18; // cl
  unsigned __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rcx
  int v23; // [rsp+8h] [rbp-B8h]
  unsigned __int64 v24; // [rsp+10h] [rbp-B0h] BYREF
  unsigned int v25; // [rsp+18h] [rbp-A8h]
  void *s; // [rsp+20h] [rbp-A0h] BYREF
  unsigned int v27; // [rsp+28h] [rbp-98h]
  unsigned __int64 v28; // [rsp+30h] [rbp-90h] BYREF
  unsigned int v29; // [rsp+38h] [rbp-88h]
  __int64 v30; // [rsp+40h] [rbp-80h] BYREF
  unsigned int v31; // [rsp+48h] [rbp-78h]
  unsigned __int64 v32; // [rsp+50h] [rbp-70h] BYREF
  unsigned int v33; // [rsp+58h] [rbp-68h]
  __int64 v34; // [rsp+60h] [rbp-60h]
  unsigned int v35; // [rsp+68h] [rbp-58h]
  unsigned __int64 v36; // [rsp+70h] [rbp-50h] BYREF
  unsigned int v37; // [rsp+78h] [rbp-48h]
  __int64 v38; // [rsp+80h] [rbp-40h]
  unsigned int v39; // [rsp+88h] [rbp-38h]

  if ( sub_158A120(a2) )
  {
    sub_15897D0(a1, a3, 0);
    return a1;
  }
  if ( sub_158A0B0(a2) )
  {
    sub_15897D0(a1, a3, 1);
    return a1;
  }
  v25 = *(_DWORD *)(a2 + 8);
  if ( v25 > 0x40 )
    sub_16A4FD0(&v24, a2);
  else
    v24 = *(_QWORD *)a2;
  v27 = *(_DWORD *)(a2 + 24);
  if ( v27 > 0x40 )
    sub_16A4FD0(&s, a2 + 16);
  else
    s = *(void **)(a2 + 16);
  sub_15897D0((__int64)&v32, a3, 0);
  if ( !sub_158A670(a2) )
  {
    v11 = v25;
    if ( v25 > 0x40 )
      goto LABEL_65;
    v12 = v24;
LABEL_42:
    if ( !v12 )
    {
LABEL_45:
      v14 = v27;
      if ( v27 > 0x40 )
      {
        v16 = v14 - sub_16A57B0(&s);
      }
      else
      {
        if ( !s )
        {
LABEL_69:
          sub_16A5A50(&v30, &s);
          sub_16A5A50(&v28, &v24);
          sub_15898E0((__int64)&v36, (__int64)&v28, &v30);
          sub_158C3A0(a1, (__int64)&v36, (__int64)&v32);
          if ( v39 > 0x40 && v38 )
            j_j___libc_free_0_0(v38);
          if ( v37 > 0x40 && v36 )
            j_j___libc_free_0_0(v36);
          if ( v29 > 0x40 && v28 )
            j_j___libc_free_0_0(v28);
          if ( v31 > 0x40 && v30 )
            j_j___libc_free_0_0(v30);
          goto LABEL_51;
        }
        _BitScanReverse64(&v15, (unsigned __int64)s);
        v16 = 64 - (v15 ^ 0x3F);
      }
      if ( a3 >= v16 )
        goto LABEL_69;
      if ( a3 + 1 == v16 )
      {
        v21 = ~(1LL << a3);
        if ( v14 > 0x40 )
          *((_QWORD *)s + (a3 >> 6)) &= v21;
        else
          s = (void *)((unsigned __int64)s & v21);
        if ( (int)sub_16A9900(&s, &v24) < 0 )
          goto LABEL_69;
      }
LABEL_50:
      sub_15897D0(a1, a3, 1);
LABEL_51:
      if ( v35 > 0x40 && v34 )
        j_j___libc_free_0_0(v34);
      if ( v33 > 0x40 && v32 )
        j_j___libc_free_0_0(v32);
      goto LABEL_57;
    }
    _BitScanReverse64(&v12, v12);
    v13 = (v12 ^ 0x3F) + v11 - 64;
LABEL_44:
    if ( a3 >= v11 - v13 )
      goto LABEL_45;
    v17 = *(unsigned int *)(a2 + 8);
    v37 = v17;
    if ( (unsigned int)v17 > 0x40 )
    {
      sub_16A4EF0(&v36, 0, 0);
      v17 = v37;
      if ( a3 == v37 )
        goto LABEL_96;
    }
    else
    {
      v36 = 0;
      if ( a3 == (_DWORD)v17 )
      {
        v19 = 0;
        goto LABEL_86;
      }
    }
    if ( a3 <= 0x3F && (unsigned int)v17 <= 0x40 )
    {
      v18 = a3 + 64 - v17;
      LODWORD(v17) = v37;
      v19 = v36 | (0xFFFFFFFFFFFFFFFFLL >> v18 << a3);
      goto LABEL_86;
    }
    sub_16A5260(&v36, a3, v17);
    LODWORD(v17) = v37;
LABEL_96:
    v19 = v36;
    if ( (unsigned int)v17 > 0x40 )
    {
      sub_16A8890(&v36, &v24);
      LODWORD(v17) = v37;
      v20 = v36;
      goto LABEL_87;
    }
LABEL_86:
    v20 = v24 & v19;
LABEL_87:
    v31 = v17;
    v30 = v20;
    sub_16A7590(&v24, &v30);
    sub_16A7590(&s, &v30);
    if ( v31 > 0x40 && v30 )
      j_j___libc_free_0_0(v30);
    goto LABEL_45;
  }
  if ( *(_DWORD *)(a2 + 24) > 0x40u )
  {
    v23 = *(_DWORD *)(a2 + 24);
    if ( a3 < v23 - (unsigned int)sub_16A57B0(a2 + 16) )
      goto LABEL_50;
    LODWORD(_RDX) = sub_16A58F0(a2 + 16);
  }
  else
  {
    v5 = *(_QWORD *)(a2 + 16);
    if ( !v5 )
    {
      _RAX = -1;
LABEL_15:
      __asm { tzcnt   rdx, rax }
      goto LABEL_16;
    }
    _BitScanReverse64(&v6, v5);
    if ( a3 < 64 - ((unsigned int)v6 ^ 0x3F) )
      goto LABEL_50;
    LODWORD(_RDX) = 64;
    _RAX = ~v5;
    if ( _RAX )
      goto LABEL_15;
  }
LABEL_16:
  if ( a3 == (_DWORD)_RDX )
    goto LABEL_50;
  sub_16A5A50(&v30, a2 + 16);
  v29 = a3;
  if ( a3 > 0x40 )
    sub_16A4EF0(&v28, -1, 1);
  else
    v28 = 0xFFFFFFFFFFFFFFFFLL >> -(char)a3;
  sub_15898E0((__int64)&v36, (__int64)&v28, &v30);
  if ( v33 > 0x40 && v32 )
    j_j___libc_free_0_0(v32);
  v32 = v36;
  v9 = v37;
  v37 = 0;
  v33 = v9;
  if ( v35 > 0x40 && v34 )
  {
    j_j___libc_free_0_0(v34);
    v34 = v38;
    v35 = v39;
    if ( v37 > 0x40 && v36 )
      j_j___libc_free_0_0(v36);
  }
  else
  {
    v34 = v38;
    v35 = v39;
  }
  if ( v29 > 0x40 && v28 )
    j_j___libc_free_0_0(v28);
  if ( v31 > 0x40 && v30 )
    j_j___libc_free_0_0(v30);
  if ( v27 <= 0x40 )
  {
    s = (void *)-1LL;
    v10 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v27;
LABEL_35:
    s = (void *)(v10 & (unsigned __int64)s);
    goto LABEL_36;
  }
  memset(s, -1, 8 * (((unsigned __int64)v27 + 63) >> 6));
  v10 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v27;
  if ( v27 <= 0x40 )
    goto LABEL_35;
  v22 = (unsigned int)(((unsigned __int64)v27 + 63) >> 6) - 1;
  *((_QWORD *)s + v22) &= v10;
LABEL_36:
  v11 = v25;
  if ( v25 <= 0x40 )
  {
    v12 = v24;
    if ( (void *)v24 == s )
      goto LABEL_38;
    goto LABEL_42;
  }
  if ( !(unsigned __int8)sub_16A5220(&v24, &s) )
  {
LABEL_65:
    v13 = sub_16A57B0(&v24);
    goto LABEL_44;
  }
LABEL_38:
  *(_DWORD *)(a1 + 8) = v33;
  *(_QWORD *)a1 = v32;
  *(_DWORD *)(a1 + 24) = v35;
  *(_QWORD *)(a1 + 16) = v34;
LABEL_57:
  if ( v27 > 0x40 && s )
    j_j___libc_free_0_0(s);
  if ( v25 > 0x40 && v24 )
    j_j___libc_free_0_0(v24);
  return a1;
}
