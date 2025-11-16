// Function: sub_14C0CB0
// Address: 0x14c0cb0
//
__int64 __fastcall sub_14C0CB0(__int64 *a1, __int64 *a2, bool a3, __int64 a4, __int64 a5, unsigned int a6, __int64 *a7)
{
  unsigned int v10; // r10d
  unsigned int v11; // eax
  unsigned int v12; // r8d
  int v13; // r12d
  unsigned __int64 v14; // rax
  int v15; // eax
  unsigned __int64 v16; // rax
  unsigned int v17; // eax
  unsigned int v18; // eax
  __int64 v19; // r12
  unsigned __int64 v20; // r9
  unsigned __int64 v21; // r9
  unsigned __int64 v24; // r12
  unsigned __int64 v25; // r12
  unsigned int v27; // r9d
  int v31; // ecx
  int v32; // edx
  unsigned int v35; // r12d
  unsigned int v36; // r12d
  unsigned int v37; // eax
  unsigned int v38; // eax
  __int64 v39; // rdx
  __int64 v40; // rsi
  unsigned int v41; // eax
  unsigned __int64 v42; // rsi
  unsigned __int64 v43; // rdx
  __int64 result; // rax
  unsigned int v45; // esi
  unsigned int v46; // esi
  unsigned __int64 v47; // rdx
  int v48; // eax
  int v49; // eax
  unsigned int v50; // r9d
  __int64 v51; // rdx
  __int64 v52; // rdx
  unsigned int v53; // esi
  bool v54; // r11
  __int64 v55; // rdx
  bool v56; // dl
  bool v57; // si
  bool v58; // cl
  char v59; // al
  int v60; // eax
  unsigned int v61; // eax
  unsigned int v62; // eax
  unsigned int v63; // eax
  unsigned int v64; // esi
  unsigned __int64 v65; // rdx
  unsigned int v66; // ecx
  unsigned int v67; // esi
  char v68; // al
  int v69; // [rsp+0h] [rbp-D0h]
  unsigned int v70; // [rsp+0h] [rbp-D0h]
  unsigned __int64 v71; // [rsp+0h] [rbp-D0h]
  unsigned int v72; // [rsp+0h] [rbp-D0h]
  unsigned int v73; // [rsp+8h] [rbp-C8h]
  unsigned __int64 v74; // [rsp+10h] [rbp-C0h]
  __int64 v75; // [rsp+18h] [rbp-B8h]
  __int64 v77; // [rsp+20h] [rbp-B0h]
  unsigned int v78; // [rsp+20h] [rbp-B0h]
  unsigned int v79; // [rsp+28h] [rbp-A8h]
  bool v81; // [rsp+2Ch] [rbp-A4h]
  bool v82; // [rsp+33h] [rbp-9Dh]
  unsigned int v83; // [rsp+34h] [rbp-9Ch]
  unsigned int v84; // [rsp+34h] [rbp-9Ch]
  unsigned int v85; // [rsp+34h] [rbp-9Ch]
  unsigned int v86; // [rsp+34h] [rbp-9Ch]
  unsigned int v87; // [rsp+34h] [rbp-9Ch]
  unsigned int v88; // [rsp+38h] [rbp-98h]
  __int64 v89; // [rsp+40h] [rbp-90h] BYREF
  unsigned int v90; // [rsp+48h] [rbp-88h]
  __int64 v91; // [rsp+50h] [rbp-80h] BYREF
  unsigned int v92; // [rsp+58h] [rbp-78h]
  unsigned __int64 v93; // [rsp+60h] [rbp-70h] BYREF
  unsigned int v94; // [rsp+68h] [rbp-68h]
  unsigned __int64 v95; // [rsp+70h] [rbp-60h] BYREF
  unsigned int v96; // [rsp+78h] [rbp-58h]
  unsigned __int64 v97; // [rsp+80h] [rbp-50h] BYREF
  unsigned int v98; // [rsp+88h] [rbp-48h]
  __int64 v99; // [rsp+90h] [rbp-40h] BYREF
  unsigned int v100; // [rsp+98h] [rbp-38h]

  v82 = a3;
  v88 = *(_DWORD *)(a4 + 8);
  v83 = a6 + 1;
  sub_14B86A0(a2, a4, a6 + 1, a7);
  sub_14B86A0(a1, a5, v83, a7);
  if ( !a3 )
  {
    v10 = *(_DWORD *)(a5 + 8);
    v11 = *(_DWORD *)(a4 + 8);
    v81 = 0;
    v12 = *(_DWORD *)(a4 + 24);
    goto LABEL_3;
  }
  v11 = *(_DWORD *)(a4 + 8);
  v10 = *(_DWORD *)(a5 + 8);
  v81 = 0;
  v12 = *(_DWORD *)(a4 + 24);
  if ( a2 != a1 )
  {
    v85 = v11 - 1;
    if ( v11 <= 0x40 )
    {
      v74 = *(_QWORD *)a4;
      v50 = v10 - 1;
      v51 = *(_QWORD *)a5;
      if ( v10 <= 0x40 )
        goto LABEL_93;
    }
    else
    {
      v50 = v10 - 1;
      v51 = *(_QWORD *)a5;
      v74 = *(_QWORD *)(*(_QWORD *)a4 + 8LL * (v85 >> 6));
      if ( v10 <= 0x40 )
      {
LABEL_93:
        v75 = v51;
LABEL_94:
        v52 = *(_QWORD *)(a4 + 16);
        if ( v12 > 0x40 )
          v52 = *(_QWORD *)(v52 + 8LL * ((v12 - 1) >> 6));
        v53 = *(_DWORD *)(a5 + 24);
        v54 = (v52 & (1LL << ((unsigned __int8)v12 - 1))) != 0;
        v55 = *(_QWORD *)(a5 + 16);
        if ( v53 > 0x40 )
          v55 = *(_QWORD *)(v55 + 8LL * ((v53 - 1) >> 6));
        v81 = 0;
        v56 = (v55 & (1LL << ((unsigned __int8)v53 - 1))) != 0;
        v82 = v54 && v56;
        if ( !v54 || !v56 )
        {
          v57 = ((1LL << v85) & v74) != 0;
          v58 = ((1LL << v50) & v75) != 0;
          if ( v57 && v58 )
          {
            v82 = v57 && v58;
          }
          else
          {
            v82 = v58 && v54;
            if ( v58 && v54 )
            {
              v68 = sub_14BE170((__int64)a1, a6, a7);
              v82 = 0;
              v10 = *(_DWORD *)(a5 + 8);
              v81 = v68;
              v12 = *(_DWORD *)(a4 + 24);
              v11 = *(_DWORD *)(a4 + 8);
            }
            else
            {
              v81 = v57 && v56;
              if ( v57 && v56 )
              {
                v59 = sub_14BE170((__int64)a2, a6, a7);
                v10 = *(_DWORD *)(a5 + 8);
                v12 = *(_DWORD *)(a4 + 24);
                v81 = v59;
                v11 = *(_DWORD *)(a4 + 8);
              }
            }
          }
        }
        goto LABEL_3;
      }
    }
    v75 = *(_QWORD *)(v51 + 8LL * (v50 >> 6));
    goto LABEL_94;
  }
LABEL_3:
  if ( v11 <= 0x40 )
  {
    v13 = 64;
    v14 = ~(*(_QWORD *)a4 << (64 - (unsigned __int8)v11));
    if ( v14 )
    {
      _BitScanReverse64(&v14, v14);
      v13 = v14 ^ 0x3F;
    }
    if ( v10 <= 0x40 )
      goto LABEL_7;
LABEL_104:
    v87 = v12;
    v15 = sub_16A5810(a5);
    v12 = v87;
    goto LABEL_9;
  }
  v78 = v12;
  v86 = v10;
  v60 = sub_16A5810(a4);
  LOBYTE(v10) = v86;
  v12 = v78;
  v13 = v60;
  if ( v86 > 0x40 )
    goto LABEL_104;
LABEL_7:
  v15 = 64;
  if ( *(_QWORD *)a5 << (64 - (unsigned __int8)v10) != -1 )
  {
    _BitScanReverse64(&v16, ~(*(_QWORD *)a5 << (64 - (unsigned __int8)v10)));
    v15 = v16 ^ 0x3F;
  }
LABEL_9:
  v17 = v13 + v15;
  v84 = 0;
  if ( v17 >= v88 )
  {
    v18 = v17 - v88;
    if ( v18 > v88 )
      v18 = v88;
    v84 = v18;
  }
  v90 = v12;
  v77 = a4 + 16;
  if ( v12 <= 0x40 )
  {
    v19 = a5 + 16;
    v89 = *(_QWORD *)(a4 + 16);
    v92 = *(_DWORD *)(a5 + 24);
    if ( v92 <= 0x40 )
      goto LABEL_15;
LABEL_106:
    sub_16A4FD0(&v91, v19);
    v98 = *(_DWORD *)(a4 + 8);
    if ( v98 <= 0x40 )
      goto LABEL_16;
    goto LABEL_107;
  }
  v19 = a5 + 16;
  sub_16A4FD0(&v89, v77);
  v92 = *(_DWORD *)(a5 + 24);
  if ( v92 > 0x40 )
    goto LABEL_106;
LABEL_15:
  v91 = *(_QWORD *)(a5 + 16);
  v98 = *(_DWORD *)(a4 + 8);
  if ( v98 <= 0x40 )
  {
LABEL_16:
    v20 = *(_QWORD *)a4;
LABEL_17:
    v21 = *(_QWORD *)(a4 + 16) | v20;
    goto LABEL_18;
  }
LABEL_107:
  sub_16A4FD0(&v97, a4);
  if ( v98 <= 0x40 )
  {
    v20 = v97;
    goto LABEL_17;
  }
  sub_16A89F0(&v97, v77);
  v61 = v98;
  v21 = v97;
  v98 = 0;
  v100 = v61;
  v99 = v97;
  if ( v61 <= 0x40 )
  {
LABEL_18:
    if ( ~v21 )
    {
      __asm { tzcnt   rax, r9 }
      v73 = _RAX;
    }
    else
    {
      v73 = 64;
    }
    goto LABEL_20;
  }
  v71 = v97;
  v73 = sub_16A58F0(&v99);
  if ( v71 )
  {
    j_j___libc_free_0_0(v71);
    if ( v98 > 0x40 )
    {
      if ( v97 )
      {
        j_j___libc_free_0_0(v97);
        v98 = *(_DWORD *)(a5 + 8);
        if ( v98 <= 0x40 )
          goto LABEL_21;
        goto LABEL_113;
      }
    }
  }
LABEL_20:
  v98 = *(_DWORD *)(a5 + 8);
  if ( v98 <= 0x40 )
  {
LABEL_21:
    v24 = *(_QWORD *)a5;
LABEL_22:
    v25 = *(_QWORD *)(a5 + 16) | v24;
    goto LABEL_23;
  }
LABEL_113:
  sub_16A4FD0(&v97, a5);
  if ( v98 <= 0x40 )
  {
    v24 = v97;
    goto LABEL_22;
  }
  sub_16A89F0(&v97, v19);
  v62 = v98;
  v25 = v97;
  v98 = 0;
  v100 = v62;
  v99 = v97;
  if ( v62 <= 0x40 )
  {
LABEL_23:
    _R12 = ~v25;
    v27 = 64;
    __asm { tzcnt   rdx, r12 }
    if ( _R12 )
      v27 = _RDX;
    goto LABEL_25;
  }
  v63 = sub_16A58F0(&v99);
  v27 = v63;
  if ( v25 )
  {
    v72 = v63;
    j_j___libc_free_0_0(v25);
    v27 = v72;
    if ( v98 > 0x40 )
    {
      if ( v97 )
      {
        j_j___libc_free_0_0(v97);
        v27 = v72;
      }
    }
  }
LABEL_25:
  if ( *(_DWORD *)(a4 + 8) > 0x40u )
  {
    v70 = v27;
    v49 = sub_16A58F0(a4);
    v27 = v70;
    v31 = v49;
  }
  else
  {
    _RSI = ~*(_QWORD *)a4;
    __asm { tzcnt   rax, rsi }
    v31 = _RAX;
    if ( *(_QWORD *)a4 == -1 )
      v31 = 64;
  }
  if ( *(_DWORD *)(a5 + 8) > 0x40u )
  {
    v79 = v27;
    v69 = v31;
    v48 = sub_16A58F0(a5);
    v27 = v79;
    v31 = v69;
    v32 = v48;
  }
  else
  {
    v32 = 64;
    _RSI = ~*(_QWORD *)a5;
    __asm { tzcnt   r12, rsi }
    if ( *(_QWORD *)a5 != -1 )
      v32 = _R12;
  }
  v35 = v73 - v31;
  if ( v27 - v32 <= v73 - v31 )
    v35 = v27 - v32;
  v36 = v31 + v32 + v35;
  if ( v36 > v88 )
    v36 = v88;
  sub_16A88B0(&v99, &v91, v27);
  sub_16A88B0(&v97, &v89, v73);
  sub_16A7B50(&v93, &v97, &v99);
  if ( v98 > 0x40 && v97 )
    j_j___libc_free_0_0(v97);
  if ( v100 > 0x40 && v99 )
    j_j___libc_free_0_0(v99);
  v37 = *(_DWORD *)(a4 + 8);
  if ( v37 > 0x40 )
    memset(*(void **)a4, 0, 8 * (((unsigned __int64)v37 + 63) >> 6));
  else
    *(_QWORD *)a4 = 0;
  v38 = *(_DWORD *)(a4 + 24);
  if ( v38 > 0x40 )
    memset(*(void **)(a4 + 16), 0, 8 * (((unsigned __int64)v38 + 63) >> 6));
  else
    *(_QWORD *)(a4 + 16) = 0;
  v39 = *(unsigned int *)(a4 + 8);
  v40 = *(_DWORD *)(a4 + 8) - v84;
  if ( (_DWORD)v39 != (_DWORD)v40 )
  {
    if ( (unsigned int)v40 > 0x3F || (unsigned int)v39 > 0x40 )
      sub_16A5260(a4, v40, v39);
    else
      *(_QWORD *)a4 |= 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v84) << (*(_BYTE *)(a4 + 8) - (unsigned __int8)v84);
  }
  v41 = v94;
  v96 = v94;
  if ( v94 <= 0x40 )
  {
    v42 = v93;
LABEL_51:
    v43 = ~v42 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v41);
    v95 = v43;
    goto LABEL_52;
  }
  sub_16A4FD0(&v95, &v93);
  v41 = v96;
  if ( v96 <= 0x40 )
  {
    v42 = v95;
    goto LABEL_51;
  }
  sub_16A8F40(&v95);
  v41 = v96;
  v43 = v95;
LABEL_52:
  v97 = v43;
  v98 = v41;
  v96 = 0;
  sub_16A88B0(&v99, &v97, v36);
  if ( *(_DWORD *)(a4 + 8) > 0x40u )
    sub_16A89F0(a4, &v99);
  else
    *(_QWORD *)a4 |= v99;
  if ( v100 > 0x40 && v99 )
    j_j___libc_free_0_0(v99);
  if ( v98 > 0x40 && v97 )
    j_j___libc_free_0_0(v97);
  if ( v96 > 0x40 && v95 )
    j_j___libc_free_0_0(v95);
  sub_16A88B0(&v99, &v93, v36);
  if ( *(_DWORD *)(a4 + 24) > 0x40u )
  {
    result = sub_16A89F0(v77, &v99);
  }
  else
  {
    result = v99;
    *(_QWORD *)(a4 + 16) |= v99;
  }
  if ( v100 > 0x40 && v99 )
    result = j_j___libc_free_0_0(v99);
  if ( v82 )
  {
    v45 = *(_DWORD *)(a4 + 24);
    result = *(_QWORD *)(a4 + 16);
    if ( v45 > 0x40 )
      result = *(_QWORD *)(result + 8LL * ((v45 - 1) >> 6));
    if ( (result & (1LL << ((unsigned __int8)v45 - 1))) == 0 )
    {
      v64 = *(_DWORD *)(a4 + 8);
      v65 = *(_QWORD *)a4;
      v66 = v64 - 1;
      result = 1LL << ((unsigned __int8)v64 - 1);
      if ( v64 <= 0x40 )
      {
        result |= v65;
        *(_QWORD *)a4 = result;
        goto LABEL_76;
      }
LABEL_129:
      *(_QWORD *)(v65 + 8LL * (v66 >> 6)) |= result;
      goto LABEL_76;
    }
  }
  if ( v81 )
  {
    v46 = *(_DWORD *)(a4 + 8);
    v47 = *(_QWORD *)a4;
    result = 1LL << ((unsigned __int8)v46 - 1);
    if ( v46 > 0x40 )
      v47 = *(_QWORD *)(v47 + 8LL * ((v46 - 1) >> 6));
    if ( (v47 & result) == 0 )
    {
      v67 = *(_DWORD *)(a4 + 24);
      v65 = *(_QWORD *)(a4 + 16);
      v66 = v67 - 1;
      result = 1LL << ((unsigned __int8)v67 - 1);
      if ( v67 <= 0x40 )
      {
        result |= v65;
        *(_QWORD *)(a4 + 16) = result;
        goto LABEL_76;
      }
      goto LABEL_129;
    }
  }
LABEL_76:
  if ( v94 > 0x40 && v93 )
    result = j_j___libc_free_0_0(v93);
  if ( v92 > 0x40 && v91 )
    result = j_j___libc_free_0_0(v91);
  if ( v90 > 0x40 )
  {
    if ( v89 )
      return j_j___libc_free_0_0(v89);
  }
  return result;
}
