// Function: sub_1767AD0
// Address: 0x1767ad0
//
_QWORD *__fastcall sub_1767AD0(
        __int64 *a1,
        __int64 a2,
        __int64 *a3,
        __int64 a4,
        __int64 a5,
        __m128 a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        double a12,
        __m128 a13)
{
  __int64 v15; // r12
  unsigned int v16; // ebx
  bool v17; // al
  unsigned int v18; // r13d
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rbx
  __int16 v22; // r13
  int v23; // eax
  _QWORD *v24; // r12
  _QWORD **v25; // rax
  _QWORD *v26; // r14
  __int64 *v27; // rax
  __int64 v28; // rsi
  unsigned int v30; // r9d
  int v32; // r9d
  int v33; // eax
  __int64 v34; // rax
  __int64 v35; // rbx
  __int64 v36; // r13
  const void *v40; // rax
  __int64 v41; // r15
  _QWORD *v42; // rax
  double v43; // xmm4_8
  double v44; // xmm5_8
  __int16 v45; // r14
  __int64 v46; // r13
  int v47; // eax
  _QWORD *v48; // rax
  bool v49; // al
  int v50; // [rsp+Ch] [rbp-64h]
  unsigned int v51; // [rsp+Ch] [rbp-64h]
  __int64 v52; // [rsp+18h] [rbp-58h]
  const void *v53; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v54; // [rsp+28h] [rbp-48h]
  __int16 v55; // [rsp+30h] [rbp-40h]

  v15 = a4;
  v16 = *(_DWORD *)(a5 + 8);
  v52 = a2;
  if ( v16 <= 0x40 )
    v17 = *(_QWORD *)a5 == 0;
  else
    v17 = v16 == (unsigned int)sub_16A57B0(a5);
  if ( v17 )
    return 0;
  if ( v16 <= 0x40 )
  {
    _RAX = *(const void **)a5;
    v18 = 64;
    __asm { tzcnt   rdx, rax }
    if ( *(_QWORD *)a5 )
      v18 = _RDX;
    if ( v18 > v16 )
      v18 = v16;
  }
  else
  {
    v18 = sub_16A58A0(a5);
  }
  v19 = *(unsigned int *)(v15 + 8);
  LOBYTE(a4) = v18 != 0;
  if ( (unsigned int)v19 <= 0x40 )
  {
    _RAX = *(const void **)v15;
    if ( *(_QWORD *)v15 || !v18 )
    {
      if ( _RAX == *(const void **)a5 )
        goto LABEL_10;
      v30 = 64;
      __asm { tzcnt   rcx, rax }
      if ( _RAX )
        v30 = _RCX;
      if ( (unsigned int)v19 <= v30 )
        v30 = *(_DWORD *)(v15 + 8);
      v32 = v30 - v18;
      if ( v32 <= 0 )
        goto LABEL_24;
      goto LABEL_34;
    }
LABEL_47:
    v45 = 35;
    v46 = sub_15A0680(*a3, v16 - v18, 0);
    v47 = *(unsigned __int16 *)(a2 + 18);
    BYTE1(v47) &= ~0x80u;
    if ( v47 == 33 )
      v45 = sub_15FF0F0(35);
    v55 = 257;
    v48 = sub_1648A60(56, 2u);
    v24 = v48;
    if ( v48 )
      sub_17582E0((__int64)v48, v45, (__int64)a3, v46, (__int64)&v53);
    return v24;
  }
  v50 = *(_DWORD *)(v15 + 8);
  if ( v50 == (unsigned int)sub_16A57B0(v15) && v18 )
    goto LABEL_47;
  a2 = a5;
  if ( sub_16A5220(v15, (const void **)a5) )
  {
LABEL_10:
    v20 = sub_15A06D0((__int64 **)*a3, a2, v19, a4);
LABEL_11:
    v21 = v20;
    v22 = 32;
    v23 = *(unsigned __int16 *)(v52 + 18);
    BYTE1(v23) &= ~0x80u;
    if ( v23 == 33 )
      v22 = sub_15FF0F0(32);
    v55 = 257;
    v24 = sub_1648A60(56, 2u);
    if ( v24 )
    {
      v25 = (_QWORD **)*a3;
      if ( *(_BYTE *)(*a3 + 8) == 16 )
      {
        v26 = v25[4];
        v27 = (__int64 *)sub_1643320(*v25);
        v28 = (__int64)sub_16463B0(v27, (unsigned int)v26);
      }
      else
      {
        v28 = sub_1643320(*v25);
      }
      sub_15FEC10((__int64)v24, v28, 51, v22, (__int64)a3, v21, (__int64)&v53, 0);
    }
    return v24;
  }
  v32 = sub_16A58A0(v15) - v18;
  if ( v32 <= 0 )
    goto LABEL_24;
LABEL_34:
  v54 = v16;
  if ( v16 > 0x40 )
  {
    v51 = v32;
    sub_16A4FD0((__int64)&v53, (const void **)a5);
    v16 = v54;
    v32 = v51;
    if ( v54 > 0x40 )
    {
      sub_16A7DC0((__int64 *)&v53, v51);
      v32 = v51;
      if ( v54 <= 0x40 )
      {
        if ( *(const void **)v15 == v53 )
        {
LABEL_39:
          v20 = sub_15A0680(*a3, v32, 0);
          goto LABEL_11;
        }
      }
      else
      {
        v49 = sub_16A5220((__int64)&v53, (const void **)v15);
        v32 = v51;
        if ( v49 )
        {
          if ( v53 )
          {
            j_j___libc_free_0_0(v53);
            v32 = v51;
          }
          goto LABEL_39;
        }
        if ( v53 )
          j_j___libc_free_0_0(v53);
      }
      goto LABEL_24;
    }
  }
  else
  {
    v53 = *(const void **)a5;
  }
  v40 = 0;
  if ( v32 != v16 )
    v40 = (const void *)(((_QWORD)v53 << v32) & (0xFFFFFFFFFFFFFFFFLL >> -(char)v16));
  v53 = v40;
  if ( v40 == *(const void **)v15 )
    goto LABEL_39;
LABEL_24:
  v33 = *(unsigned __int16 *)(v52 + 18);
  v24 = (_QWORD *)v52;
  BYTE1(v33) &= ~0x80u;
  v34 = sub_15A0680(*(_QWORD *)v52, v33 == 33, 0);
  v35 = *(_QWORD *)(v52 + 8);
  v36 = v34;
  if ( !v35 )
    return 0;
  v41 = *a1;
  do
  {
    v42 = sub_1648700(v35);
    sub_170B990(v41, (__int64)v42);
    v35 = *(_QWORD *)(v35 + 8);
  }
  while ( v35 );
  if ( v52 == v36 )
    v36 = sub_1599EF0(*(__int64 ***)v52);
  sub_164D160(v52, v36, a6, a7, a8, a9, v43, v44, a12, a13);
  return v24;
}
