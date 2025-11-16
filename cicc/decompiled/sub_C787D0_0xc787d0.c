// Function: sub_C787D0
// Address: 0xc787d0
//
__int64 __fastcall sub_C787D0(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  unsigned int v7; // ebx
  __int64 v8; // rax
  unsigned int v9; // edx
  unsigned __int64 v10; // rax
  unsigned int v11; // eax
  __int64 v12; // rdx
  unsigned __int64 v13; // rdx
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // r8
  __int64 v16; // r8
  unsigned __int64 v19; // r9
  __int64 v20; // r9
  unsigned int v22; // r8d
  int v26; // ecx
  unsigned int v29; // r14d
  unsigned int v30; // r14d
  unsigned int v31; // edx
  unsigned int v32; // esi
  unsigned int v33; // eax
  unsigned __int64 v34; // rdx
  unsigned __int64 v35; // rdx
  __int64 v36; // rdi
  unsigned __int64 v38; // rax
  int v39; // eax
  int v40; // eax
  unsigned int v41; // eax
  unsigned int v42; // eax
  unsigned int v43; // eax
  __int64 v44; // rdi
  unsigned int v45; // [rsp+Ch] [rbp-E4h]
  unsigned int v46; // [rsp+10h] [rbp-E0h]
  int v47; // [rsp+10h] [rbp-E0h]
  __int64 v48; // [rsp+10h] [rbp-E0h]
  __int64 v49; // [rsp+10h] [rbp-E0h]
  unsigned int v50; // [rsp+10h] [rbp-E0h]
  unsigned int v52; // [rsp+1Ch] [rbp-D4h]
  __int64 *v53; // [rsp+20h] [rbp-D0h]
  __int64 *v54; // [rsp+28h] [rbp-C8h]
  int v55; // [rsp+30h] [rbp-C0h]
  bool v56; // [rsp+4Fh] [rbp-A1h] BYREF
  unsigned __int64 v57; // [rsp+50h] [rbp-A0h] BYREF
  unsigned int v58; // [rsp+58h] [rbp-98h]
  unsigned __int64 v59; // [rsp+60h] [rbp-90h] BYREF
  unsigned int v60; // [rsp+68h] [rbp-88h]
  unsigned __int64 v61; // [rsp+70h] [rbp-80h] BYREF
  unsigned int v62; // [rsp+78h] [rbp-78h]
  unsigned __int64 v63; // [rsp+80h] [rbp-70h] BYREF
  unsigned int v64; // [rsp+88h] [rbp-68h]
  unsigned __int64 v65; // [rsp+90h] [rbp-60h] BYREF
  unsigned int v66; // [rsp+98h] [rbp-58h]
  __int64 v67; // [rsp+A0h] [rbp-50h] BYREF
  unsigned int v68; // [rsp+A8h] [rbp-48h]
  __int64 v69; // [rsp+B0h] [rbp-40h] BYREF
  unsigned int v70; // [rsp+B8h] [rbp-38h]

  v7 = *(_DWORD *)(a2 + 8);
  v70 = v7;
  if ( v7 > 0x40 )
  {
    sub_C43780((__int64)&v69, (const void **)a2);
    v9 = v70;
    if ( v70 > 0x40 )
    {
      sub_C43D10((__int64)&v69);
      v9 = v70;
      v10 = v69;
      goto LABEL_5;
    }
    v8 = v69;
  }
  else
  {
    v8 = *(_QWORD *)a2;
    v9 = v7;
  }
  v10 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v9) & ~v8;
  if ( !v9 )
    v10 = 0;
LABEL_5:
  v57 = v10;
  v11 = *(_DWORD *)(a3 + 8);
  v58 = v9;
  v70 = v11;
  if ( v11 > 0x40 )
  {
    sub_C43780((__int64)&v69, (const void **)a3);
    v11 = v70;
    if ( v70 > 0x40 )
    {
      sub_C43D10((__int64)&v69);
      v11 = v70;
      v13 = v69;
      goto LABEL_9;
    }
    v12 = v69;
  }
  else
  {
    v12 = *(_QWORD *)a3;
  }
  v13 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v11) & ~v12;
  if ( !v11 )
    v13 = 0;
LABEL_9:
  v59 = v13;
  v60 = v11;
  sub_C49BE0((__int64)&v61, (__int64)&v57, (__int64)&v59, &v56);
  v55 = 0;
  if ( !v56 )
  {
    v55 = v62;
    if ( v62 > 0x40 )
    {
      v55 = sub_C444A0((__int64)&v61);
    }
    else if ( v61 )
    {
      _BitScanReverse64(&v14, v61);
      v55 = v62 - 64 + (v14 ^ 0x3F);
    }
  }
  v54 = (__int64 *)(a2 + 16);
  v53 = (__int64 *)(a3 + 16);
  v68 = *(_DWORD *)(a2 + 8);
  if ( v68 <= 0x40 )
  {
    v15 = *(_QWORD *)a2;
LABEL_15:
    v16 = *(_QWORD *)(a2 + 16) | v15;
    v67 = v16;
LABEL_16:
    _R8 = ~v16;
    __asm { tzcnt   rax, r8 }
    if ( !_R8 )
      LODWORD(_RAX) = 64;
    v52 = _RAX;
    goto LABEL_19;
  }
  sub_C43780((__int64)&v67, (const void **)a2);
  if ( v68 <= 0x40 )
  {
    v15 = v67;
    goto LABEL_15;
  }
  sub_C43BD0(&v67, v54);
  v41 = v68;
  v16 = v67;
  v68 = 0;
  v70 = v41;
  v69 = v67;
  if ( v41 <= 0x40 )
    goto LABEL_16;
  v48 = v67;
  v52 = sub_C445E0((__int64)&v69);
  if ( v48 )
  {
    j_j___libc_free_0_0(v48);
    if ( v68 > 0x40 )
    {
      if ( v67 )
      {
        j_j___libc_free_0_0(v67);
        v68 = *(_DWORD *)(a3 + 8);
        if ( v68 <= 0x40 )
          goto LABEL_20;
        goto LABEL_97;
      }
    }
  }
LABEL_19:
  v68 = *(_DWORD *)(a3 + 8);
  if ( v68 <= 0x40 )
  {
LABEL_20:
    v19 = *(_QWORD *)a3;
LABEL_21:
    v20 = *(_QWORD *)(a3 + 16) | v19;
    v68 = 0;
    v67 = v20;
    goto LABEL_22;
  }
LABEL_97:
  sub_C43780((__int64)&v67, (const void **)a3);
  if ( v68 <= 0x40 )
  {
    v19 = v67;
    goto LABEL_21;
  }
  sub_C43BD0(&v67, v53);
  v42 = v68;
  v20 = v67;
  v68 = 0;
  v70 = v42;
  v69 = v67;
  if ( v42 <= 0x40 )
  {
LABEL_22:
    _R9 = ~v20;
    v22 = 64;
    __asm { tzcnt   rdx, r9 }
    if ( _R9 )
      v22 = _RDX;
    goto LABEL_24;
  }
  v49 = v67;
  v43 = sub_C445E0((__int64)&v69);
  v22 = v43;
  if ( v49 )
  {
    v44 = v49;
    v50 = v43;
    j_j___libc_free_0_0(v44);
    v22 = v50;
    if ( v68 > 0x40 )
    {
      if ( v67 )
      {
        j_j___libc_free_0_0(v67);
        v22 = v50;
      }
    }
  }
LABEL_24:
  if ( *(_DWORD *)(a2 + 8) <= 0x40u )
  {
    _RSI = ~*(_QWORD *)a2;
    __asm { tzcnt   rax, rsi }
    v26 = _RAX;
    if ( *(_QWORD *)a2 == -1 )
      v26 = 64;
    if ( *(_DWORD *)(a3 + 8) <= 0x40u )
      goto LABEL_28;
LABEL_90:
    v45 = v22;
    v47 = v26;
    v40 = sub_C445E0(a3);
    v22 = v45;
    v26 = v47;
    LODWORD(_RDX) = v40;
    goto LABEL_30;
  }
  v46 = v22;
  v39 = sub_C445E0(a2);
  v22 = v46;
  v26 = v39;
  if ( *(_DWORD *)(a3 + 8) > 0x40u )
    goto LABEL_90;
LABEL_28:
  _RSI = ~*(_QWORD *)a3;
  __asm { tzcnt   rdx, rsi }
  if ( *(_QWORD *)a3 == -1 )
    LODWORD(_RDX) = 64;
LABEL_30:
  v29 = v52 - v26;
  if ( v22 - (unsigned int)_RDX <= v52 - v26 )
    v29 = v22 - _RDX;
  v30 = v26 + _RDX + v29;
  if ( v30 > v7 )
    v30 = v7;
  sub_C443A0((__int64)&v69, (__int64)v53, v22);
  sub_C443A0((__int64)&v67, (__int64)v54, v52);
  sub_C472A0((__int64)&v63, (__int64)&v67, &v69);
  if ( v68 > 0x40 && v67 )
    j_j___libc_free_0_0(v67);
  if ( v70 > 0x40 && v69 )
    j_j___libc_free_0_0(v69);
  *(_DWORD *)(a1 + 8) = v7;
  if ( v7 > 0x40 )
  {
    sub_C43690(a1, 0, 0);
    *(_DWORD *)(a1 + 24) = v7;
    sub_C43690(a1 + 16, 0, 0);
    v31 = *(_DWORD *)(a1 + 8);
  }
  else
  {
    *(_QWORD *)a1 = 0;
    v31 = v7;
    *(_DWORD *)(a1 + 24) = v7;
    *(_QWORD *)(a1 + 16) = 0;
  }
  v32 = v31 - v55;
  if ( v31 - v55 != v31 )
  {
    if ( v32 > 0x3F || v31 > 0x40 )
      sub_C43C90((_QWORD *)a1, v32, v31);
    else
      *(_QWORD *)a1 |= 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v55) << v32;
  }
  v33 = v64;
  v66 = v64;
  if ( v64 > 0x40 )
  {
    sub_C43780((__int64)&v65, (const void **)&v63);
    v33 = v66;
    if ( v66 > 0x40 )
    {
      sub_C43D10((__int64)&v65);
      v35 = v65;
      v33 = v66;
      goto LABEL_51;
    }
    v34 = v65;
  }
  else
  {
    v34 = v63;
  }
  v35 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v33) & ~v34;
  if ( !v33 )
    v35 = 0;
  v65 = v35;
LABEL_51:
  v67 = v35;
  v68 = v33;
  v66 = 0;
  sub_C443A0((__int64)&v69, (__int64)&v67, v30);
  if ( *(_DWORD *)(a1 + 8) > 0x40u )
    sub_C43BD0((_QWORD *)a1, &v69);
  else
    *(_QWORD *)a1 |= v69;
  if ( v70 > 0x40 && v69 )
    j_j___libc_free_0_0(v69);
  if ( v68 > 0x40 && v67 )
    j_j___libc_free_0_0(v67);
  if ( v66 > 0x40 && v65 )
    j_j___libc_free_0_0(v65);
  sub_C443A0((__int64)&v69, (__int64)&v63, v30);
  if ( *(_DWORD *)(a1 + 24) > 0x40u )
  {
    v36 = *(_QWORD *)(a1 + 16);
    if ( v36 )
      j_j___libc_free_0_0(v36);
  }
  *(_QWORD *)(a1 + 16) = v69;
  *(_DWORD *)(a1 + 24) = v70;
  if ( v7 > 1 && a4 )
  {
    v38 = *(_QWORD *)a1;
    if ( *(_DWORD *)(a1 + 8) > 0x40u )
      *(_QWORD *)v38 |= 2uLL;
    else
      *(_QWORD *)a1 = v38 | 2;
  }
  if ( v64 > 0x40 && v63 )
    j_j___libc_free_0_0(v63);
  if ( v62 > 0x40 && v61 )
    j_j___libc_free_0_0(v61);
  if ( v60 > 0x40 && v59 )
    j_j___libc_free_0_0(v59);
  if ( v58 > 0x40 && v57 )
    j_j___libc_free_0_0(v57);
  return a1;
}
