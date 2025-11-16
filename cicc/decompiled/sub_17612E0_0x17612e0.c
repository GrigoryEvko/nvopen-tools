// Function: sub_17612E0
// Address: 0x17612e0
//
_QWORD *__fastcall sub_17612E0(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4, double a5, double a6, double a7)
{
  __int64 v7; // r9
  _BYTE *v9; // r14
  _BYTE *v10; // r13
  int v11; // eax
  __int64 v12; // rax
  unsigned int v13; // r15d
  unsigned int v14; // r15d
  int v15; // eax
  unsigned int v16; // r15d
  __int64 v17; // rcx
  unsigned __int8 *v18; // rax
  unsigned __int8 *v19; // rcx
  _QWORD *v20; // rax
  __int64 *v21; // r15
  __int64 v22; // rdi
  unsigned __int8 *v23; // r13
  int v24; // eax
  _BOOL4 v25; // r12d
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // rax
  __int64 v29; // r14
  __int16 v30; // r12
  _QWORD *v31; // rax
  _QWORD *v32; // r15
  __int64 v34; // r15
  __int64 v35; // rdx
  bool v36; // al
  unsigned int v37; // ecx
  unsigned __int64 v38; // rdx
  unsigned int v39; // eax
  unsigned __int64 v40; // r15
  int v41; // eax
  int v42; // eax
  __int64 *v43; // r13
  __int16 v44; // r12
  __int64 v45; // rax
  __int16 v46; // r12
  __int64 v47; // rax
  __int64 v48; // rbx
  _QWORD **v49; // rax
  _QWORD *v50; // r13
  __int64 *v51; // rax
  __int64 v52; // rsi
  unsigned __int64 v53; // rax
  __int64 v54; // [rsp+8h] [rbp-78h]
  __int64 v55; // [rsp+8h] [rbp-78h]
  __int64 v56; // [rsp+8h] [rbp-78h]
  __int64 v58; // [rsp+10h] [rbp-70h]
  bool v59; // [rsp+10h] [rbp-70h]
  __int64 v60; // [rsp+10h] [rbp-70h]
  unsigned __int64 v61; // [rsp+20h] [rbp-60h] BYREF
  unsigned int v62; // [rsp+28h] [rbp-58h]
  unsigned __int64 *v63; // [rsp+30h] [rbp-50h] BYREF
  unsigned int v64; // [rsp+38h] [rbp-48h]
  __int16 v65; // [rsp+40h] [rbp-40h]

  v7 = a4;
  v9 = (_BYTE *)*(a3 - 6);
  v10 = (_BYTE *)*(a3 - 3);
  if ( v9[16] == 54 )
  {
    v34 = *((_QWORD *)v9 - 3);
    if ( *(_BYTE *)(v34 + 16) == 56 )
    {
      v35 = *(_QWORD *)(v34 - 24LL * (*(_DWORD *)(v34 + 20) & 0xFFFFFFF));
      if ( *(_BYTE *)(v35 + 16) == 3 && (*(_BYTE *)(v35 + 80) & 1) != 0 )
      {
        v55 = *(_QWORD *)(v34 - 24LL * (*(_DWORD *)(v34 + 20) & 0xFFFFFFF));
        v36 = sub_15E4F60(v55);
        v7 = a4;
        if ( !v36 )
          __asm { jmp     rax }
      }
    }
  }
  v11 = *(unsigned __int16 *)(a2 + 18);
  BYTE1(v11) &= ~0x80u;
  if ( (unsigned int)(v11 - 32) > 1 )
    return 0;
  if ( v10 == *(_BYTE **)(a2 - 24) )
  {
    v37 = *(_DWORD *)(v7 + 8);
    v62 = v37;
    if ( v37 > 0x40 )
    {
      v60 = v7;
      sub_16A4FD0((__int64)&v61, (const void **)v7);
      LOBYTE(v37) = v62;
      v7 = v60;
      if ( v62 > 0x40 )
      {
        sub_16A8F40((__int64 *)&v61);
        v7 = v60;
LABEL_32:
        v58 = v7;
        sub_16A7400((__int64)&v61);
        v39 = v62;
        v40 = v61;
        v62 = 0;
        v7 = v58;
        v64 = v39;
        v63 = (unsigned __int64 *)v61;
        if ( v39 > 0x40 )
        {
          v56 = v58;
          v41 = sub_16A5940((__int64)&v63);
          v7 = v58;
          v59 = v41 == 1;
          if ( v40 )
          {
            j_j___libc_free_0_0(v40);
            v7 = v56;
            if ( v62 > 0x40 )
            {
              if ( v61 )
              {
                j_j___libc_free_0_0(v61);
                v7 = v56;
              }
            }
          }
          if ( !v59 )
            goto LABEL_4;
        }
        else if ( !v61 || (v61 & (v61 - 1)) != 0 )
        {
          goto LABEL_4;
        }
        v42 = *(unsigned __int16 *)(a2 + 18);
        v43 = *(__int64 **)(a2 - 24);
        BYTE1(v42) &= ~0x80u;
        v44 = v42 != 32;
        v45 = sub_15A0680(*v43, 1, 0);
        v46 = 3 * v44 + 34;
        v47 = sub_15A2B60(v43, v45, 0, 0, a5, a6, a7);
        v65 = 257;
        v48 = v47;
        v32 = sub_1648A60(56, 2u);
        if ( v32 )
        {
          v49 = *(_QWORD ***)v9;
          if ( *(_BYTE *)(*(_QWORD *)v9 + 8LL) == 16 )
          {
            v50 = v49[4];
            v51 = (__int64 *)sub_1643320(*v49);
            v52 = (__int64)sub_16463B0(v51, (unsigned int)v50);
          }
          else
          {
            v52 = sub_1643320(*v49);
          }
          sub_15FEC10((__int64)v32, v52, 51, v46, (__int64)v9, v48, (__int64)&v63, 0);
        }
        return v32;
      }
      v38 = v61;
    }
    else
    {
      v38 = *(_QWORD *)v7;
    }
    v61 = ~v38 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v37);
    goto LABEL_32;
  }
LABEL_4:
  v12 = a3[1];
  if ( !v12 || *(_QWORD *)(v12 + 8) )
    return 0;
  v13 = *(_DWORD *)(v7 + 8);
  if ( v13 <= 0x40 )
  {
    if ( *(_QWORD *)v7 )
      return 0;
  }
  else if ( v13 != (unsigned int)sub_16A57B0(v7) )
  {
    return 0;
  }
  v63 = &v61;
  if ( !(unsigned __int8)sub_13D2630(&v63, v10) )
    return 0;
  v14 = *(_DWORD *)(v61 + 8);
  if ( v14 <= 0x40 )
  {
    v53 = *(_QWORD *)v61;
    if ( !*(_QWORD *)v61 || (v53 & (v53 - 1)) != 0 )
      return 0;
    _BitScanReverse64(&v53, v53);
    v15 = v14 + (v53 ^ 0x3F) - 64;
    goto LABEL_12;
  }
  v54 = v61;
  if ( (unsigned int)sub_16A5940(v61) != 1 )
    return 0;
  v15 = sub_16A57B0(v54);
LABEL_12:
  if ( v14 == v15 )
    return 0;
  v16 = v14 - v15;
  v17 = *(_QWORD *)(a1 + 2664);
  v18 = *(unsigned __int8 **)(v17 + 24);
  v19 = &v18[*(unsigned int *)(v17 + 32)];
  if ( v18 == v19 )
    return 0;
  while ( v16 != (unsigned __int64)*v18 )
  {
    if ( v19 == ++v18 )
      return 0;
  }
  v20 = (_QWORD *)sub_16498A0(a2);
  v21 = (__int64 *)sub_1644900(v20, v16);
  if ( *(_BYTE *)(*a3 + 8LL) == 16 )
    v21 = sub_16463B0(v21, *(_QWORD *)(*a3 + 32LL));
  v22 = *(_QWORD *)(a1 + 8);
  v65 = 257;
  v23 = sub_1708970(v22, 36, (__int64)v9, (__int64 **)v21, (__int64 *)&v63);
  v24 = *(unsigned __int16 *)(a2 + 18);
  BYTE1(v24) &= ~0x80u;
  v25 = v24 != 32;
  v28 = sub_15A06D0((__int64 **)v21, 36, v26, v27);
  v65 = 257;
  v29 = v28;
  v30 = v25 + 39;
  v31 = sub_1648A60(56, 2u);
  v32 = v31;
  if ( v31 )
    sub_17582E0((__int64)v31, v30, (__int64)v23, v29, (__int64)&v63);
  return v32;
}
