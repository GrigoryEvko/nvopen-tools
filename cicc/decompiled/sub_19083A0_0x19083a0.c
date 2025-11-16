// Function: sub_19083A0
// Address: 0x19083a0
//
__int64 __fastcall sub_19083A0(
        __int64 a1,
        __m128 a2,
        double a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        __m128 a9)
{
  unsigned __int64 v10; // rbx
  _QWORD *v11; // rdx
  _QWORD *v12; // rax
  _QWORD *v13; // r13
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 *v16; // r12
  __int64 v17; // rsi
  unsigned int v18; // r9d
  __int64 v19; // r8
  __int64 *v20; // rdx
  __int64 v21; // rdx
  unsigned int v22; // eax
  _QWORD *v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rdi
  unsigned int v26; // esi
  __int64 v27; // rcx
  _QWORD *v28; // r9
  __int64 v30; // rdx
  __int64 v31; // rsi
  int v32; // r8d
  int v33; // ecx
  __int64 v34; // r12
  unsigned int v35; // r13d
  __int64 v36; // rdx
  unsigned int v37; // r13d
  unsigned __int64 v38; // rax
  _QWORD *v39; // rdx
  int v40; // ecx
  int v41; // r10d
  __int64 v42; // rcx
  int v43; // edx
  unsigned int v44; // ebx
  __int64 v45; // rdx
  int v46; // ecx
  unsigned __int64 v47; // rax
  unsigned int v48; // edx
  unsigned int v49; // ebx
  unsigned int v50; // ebx
  unsigned __int64 v51; // rax
  unsigned __int64 v52; // rax
  void *v53; // rax
  unsigned int v54; // eax
  _QWORD *v55; // rdi
  __int64 v56; // rbx
  double v57; // xmm4_8
  double v58; // xmm5_8
  unsigned __int64 v59; // r13
  unsigned __int8 v60; // [rsp+17h] [rbp-99h]
  __int64 v61; // [rsp+18h] [rbp-98h]
  __int64 v62; // [rsp+20h] [rbp-90h]
  __int64 v63; // [rsp+28h] [rbp-88h]
  unsigned __int64 v64; // [rsp+40h] [rbp-70h] BYREF
  unsigned int v65; // [rsp+48h] [rbp-68h]
  unsigned __int64 v66; // [rsp+50h] [rbp-60h] BYREF
  unsigned int v67; // [rsp+58h] [rbp-58h]
  unsigned __int64 v68; // [rsp+60h] [rbp-50h] BYREF
  unsigned int v69; // [rsp+68h] [rbp-48h]
  unsigned __int64 v70; // [rsp+70h] [rbp-40h]
  unsigned int v71; // [rsp+78h] [rbp-38h]

  v61 = a1 + 168;
  v62 = *(_QWORD *)(a1 + 184);
  if ( v62 == a1 + 168 )
    return 0;
  v60 = 0;
  do
  {
    sub_15897D0((__int64)&v64, dword_4FAE5E0 + 1, 0);
    if ( (*(_BYTE *)(v62 + 40) & 1) == 0 )
      goto LABEL_41;
    v63 = 0;
    v10 = v62 + 32;
    do
    {
      v15 = *(unsigned int *)(a1 + 24);
      if ( !(_DWORD)v15 )
        goto LABEL_14;
      v16 = *(__int64 **)(v10 + 16);
      v17 = *(_QWORD *)(a1 + 8);
      v18 = (v15 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
      v19 = v17 + 16LL * v18;
      v20 = *(__int64 **)v19;
      if ( v16 != *(__int64 **)v19 )
      {
        v32 = 1;
        while ( v20 != (__int64 *)-8LL )
        {
          v33 = v32 + 1;
          v18 = (v15 - 1) & (v32 + v18);
          v19 = v17 + 16LL * v18;
          v20 = *(__int64 **)v19;
          if ( v16 == *(__int64 **)v19 )
            goto LABEL_17;
          v32 = v33;
        }
        goto LABEL_14;
      }
LABEL_17:
      if ( v19 == v17 + 16 * v15 )
        goto LABEL_14;
      v21 = *(_QWORD *)(a1 + 32) + 40LL * *(unsigned int *)(v19 + 8);
      if ( *(_QWORD *)(a1 + 40) == v21 )
        goto LABEL_14;
      sub_158C3A0((__int64)&v68, (__int64)&v64, v21 + 8);
      if ( v65 > 0x40 && v64 )
        j_j___libc_free_0_0(v64);
      v64 = v68;
      v22 = v69;
      v69 = 0;
      v65 = v22;
      if ( v67 > 0x40 && v66 )
      {
        j_j___libc_free_0_0(v66);
        v66 = v70;
        v67 = v71;
        if ( v69 > 0x40 && v68 )
          j_j___libc_free_0_0(v68);
        v11 = *(_QWORD **)(a1 + 72);
        v12 = *(_QWORD **)(a1 + 64);
        if ( v11 != v12 )
        {
LABEL_10:
          v13 = &v11[*(unsigned int *)(a1 + 80)];
          v12 = sub_16CC9F0(a1 + 56, (__int64)v16);
          if ( v16 == (__int64 *)*v12 )
          {
            v30 = *(_QWORD *)(a1 + 72);
            if ( v30 == *(_QWORD *)(a1 + 64) )
              v31 = *(unsigned int *)(a1 + 84);
            else
              v31 = *(unsigned int *)(a1 + 80);
            v39 = (_QWORD *)(v30 + 8 * v31);
            goto LABEL_31;
          }
          v14 = *(_QWORD *)(a1 + 72);
          if ( v14 == *(_QWORD *)(a1 + 64) )
          {
            v12 = (_QWORD *)(v14 + 8LL * *(unsigned int *)(a1 + 84));
            v39 = v12;
            goto LABEL_31;
          }
          v12 = (_QWORD *)(v14 + 8LL * *(unsigned int *)(a1 + 80));
LABEL_13:
          if ( v13 == v12 )
            goto LABEL_33;
          goto LABEL_14;
        }
      }
      else
      {
        v11 = *(_QWORD **)(a1 + 72);
        v66 = v70;
        v67 = v71;
        v12 = *(_QWORD **)(a1 + 64);
        if ( v11 != v12 )
          goto LABEL_10;
      }
      v13 = &v12[*(unsigned int *)(a1 + 84)];
      if ( v12 == v13 )
      {
        v39 = v12;
      }
      else
      {
        do
        {
          if ( v16 == (__int64 *)*v12 )
            break;
          ++v12;
        }
        while ( v13 != v12 );
        v39 = v13;
      }
LABEL_31:
      while ( v39 != v12 )
      {
        if ( *v12 < 0xFFFFFFFFFFFFFFFELL )
          goto LABEL_13;
        ++v12;
      }
      if ( v13 == v12 )
      {
LABEL_33:
        if ( !v63 )
          v63 = *v16;
        while ( 1 )
        {
          v16 = (__int64 *)v16[1];
          if ( !v16 )
            break;
          v23 = sub_1648700((__int64)v16);
          if ( *((_BYTE *)v23 + 16) <= 0x17u )
            goto LABEL_41;
          v24 = *(unsigned int *)(a1 + 24);
          if ( !(_DWORD)v24 )
            goto LABEL_41;
          v25 = *(_QWORD *)(a1 + 8);
          v26 = (v24 - 1) & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
          v27 = v25 + 16LL * v26;
          v28 = *(_QWORD **)v27;
          if ( v23 != *(_QWORD **)v27 )
          {
            v40 = 1;
            while ( v28 != (_QWORD *)-8LL )
            {
              v41 = v40 + 1;
              v42 = ((_DWORD)v24 - 1) & (v26 + v40);
              v26 = v42;
              v27 = v25 + 16 * v42;
              v28 = *(_QWORD **)v27;
              if ( v23 == *(_QWORD **)v27 )
                goto LABEL_39;
              v40 = v41;
            }
            goto LABEL_41;
          }
LABEL_39:
          if ( v27 == v25 + 16 * v24 || *(_QWORD *)(a1 + 40) == *(_QWORD *)(a1 + 32) + 40LL * *(unsigned int *)(v27 + 8) )
            goto LABEL_41;
        }
      }
LABEL_14:
      v10 = *(_QWORD *)(v10 + 8) & 0xFFFFFFFFFFFFFFFELL;
    }
    while ( v10 );
    v34 = *(_QWORD *)(v62 + 40) & 1LL;
    if ( (*(_QWORD *)(v62 + 40) & 1) == 0 || sub_158A0B0((__int64)&v64) || sub_158B9F0((__int64)&v64) )
      goto LABEL_41;
    v35 = v67 + 1;
    v36 = 1LL << ((unsigned __int8)v67 - 1);
    if ( v67 > 0x40 )
    {
      if ( (*(_QWORD *)(v66 + 8LL * ((v67 - 1) >> 6)) & v36) == 0 )
      {
        v37 = v35 - sub_16A57B0((__int64)&v66);
        goto LABEL_72;
      }
      v43 = sub_16A5810((__int64)&v66);
LABEL_70:
      v37 = v35 - v43;
      goto LABEL_72;
    }
    if ( (v36 & v66) != 0 )
    {
      v43 = 64;
      if ( v66 << (64 - (unsigned __int8)v67) != -1 )
      {
        _BitScanReverse64(&v51, ~(v66 << (64 - (unsigned __int8)v67)));
        v43 = v51 ^ 0x3F;
      }
      goto LABEL_70;
    }
    v37 = 1;
    if ( v66 )
    {
      _BitScanReverse64(&v38, v66);
      v37 = 65 - (v38 ^ 0x3F);
    }
LABEL_72:
    v44 = v65 + 1;
    v45 = 1LL << ((unsigned __int8)v65 - 1);
    if ( v65 > 0x40 )
    {
      if ( (*(_QWORD *)(v64 + 8LL * ((v65 - 1) >> 6)) & v45) != 0 )
      {
        v46 = sub_16A5810((__int64)&v64);
        goto LABEL_76;
      }
      v48 = v44 - sub_16A57B0((__int64)&v64);
    }
    else if ( (v45 & v64) != 0 )
    {
      v46 = 64;
      if ( v64 << (64 - (unsigned __int8)v65) != -1 )
      {
        _BitScanReverse64(&v47, ~(v64 << (64 - (unsigned __int8)v65)));
        v46 = v47 ^ 0x3F;
      }
LABEL_76:
      v48 = v44 - v46;
    }
    else
    {
      v48 = 1;
      if ( v64 )
      {
        _BitScanReverse64(&v52, v64);
        v48 = 65 - (v52 ^ 0x3F);
      }
    }
    v49 = v37;
    if ( v48 >= v37 )
      v49 = v48;
    v50 = v49 + 1;
    switch ( *(_BYTE *)(v63 + 8) )
    {
      case 0:
      case 6:
        v53 = sub_16982C0();
        break;
      case 1:
        v53 = sub_1698260();
        break;
      case 2:
        v53 = sub_1698270();
        break;
      case 3:
        v53 = sub_1698280();
        break;
      case 4:
        v53 = sub_16982A0();
        break;
      case 5:
        v53 = sub_1698290();
        break;
    }
    v54 = sub_16982D0((__int64)v53) - 1;
    if ( v54 > 0x40 )
      v54 = 64;
    if ( v54 >= v50 )
    {
      v55 = *(_QWORD **)(a1 + 264);
      if ( v50 <= 0x20 )
        v56 = sub_1643350(v55);
      else
        v56 = sub_1643360(v55);
      if ( (*(_BYTE *)(v62 + 40) & 1) != 0 )
      {
        v59 = v62 + 32;
        do
        {
          sub_1907AD0(a1, *(_QWORD *)(v59 + 16), v56, a2, a3, a4, a5, v57, v58, a8, a9);
          v59 = *(_QWORD *)(v59 + 8) & 0xFFFFFFFFFFFFFFFELL;
        }
        while ( v59 );
      }
      if ( v67 > 0x40 && v66 )
        j_j___libc_free_0_0(v66);
      if ( v65 > 0x40 && v64 )
        j_j___libc_free_0_0(v64);
      v60 = v34;
      goto LABEL_47;
    }
LABEL_41:
    if ( v67 > 0x40 && v66 )
      j_j___libc_free_0_0(v66);
    if ( v65 > 0x40 && v64 )
      j_j___libc_free_0_0(v64);
LABEL_47:
    v62 = sub_220EF30(v62);
  }
  while ( v61 != v62 );
  return v60;
}
