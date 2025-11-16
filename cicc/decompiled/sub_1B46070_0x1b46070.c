// Function: sub_1B46070
// Address: 0x1b46070
//
__int64 __fastcall sub_1B46070(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __m128 a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  __int64 v11; // r15
  __int64 **v13; // rax
  __int64 v14; // r14
  int v15; // ebx
  int v16; // r8d
  int v17; // r9d
  char v18; // bl
  __int64 v20; // r15
  __int64 v21; // rdx
  __int64 v22; // r13
  unsigned __int64 v23; // rax
  unsigned int v24; // edi
  unsigned int v25; // edx
  __int64 v26; // rsi
  int v27; // ecx
  unsigned __int64 v28; // rax
  unsigned __int64 v29; // rcx
  unsigned int v30; // edx
  __int64 v31; // rax
  __int64 v32; // rax
  char v33; // bl
  unsigned __int64 v34; // r12
  unsigned int *v35; // r12
  unsigned int v36; // r13d
  bool v37; // zf
  __int64 v38; // rax
  _BYTE *v39; // rdi
  __int64 v40; // rax
  __int64 v41; // rdx
  __int64 *v42; // rdi
  int v43; // eax
  unsigned int v44; // eax
  __int64 *v45; // rbx
  __int64 v46; // rdx
  int v47; // r8d
  int v48; // r9d
  __int64 v49; // r14
  __int64 v50; // rax
  __int64 v51; // rdx
  int v52; // r13d
  int v53; // ecx
  __int64 v54; // rdx
  unsigned int *v55; // rax
  unsigned int *v56; // rsi
  __int64 v57; // rdi
  __int64 v58; // rax
  int v59; // eax
  __int64 v60; // rbx
  unsigned int v62; // eax
  unsigned int *v63; // rcx
  __int64 v64; // rax
  __int64 v65; // rax
  double v66; // xmm4_8
  double v67; // xmm5_8
  _QWORD *v68; // r13
  __int64 v69; // rax
  __int64 v70; // rsi
  unsigned __int64 v71; // r13
  __int64 v72; // r14
  _QWORD *v73; // rdi
  int v74; // [rsp+Ch] [rbp-144h]
  __int64 v75; // [rsp+20h] [rbp-130h]
  __int64 *v76; // [rsp+20h] [rbp-130h]
  unsigned int v77; // [rsp+28h] [rbp-128h]
  char v78; // [rsp+2Fh] [rbp-121h]
  bool v79; // [rsp+2Fh] [rbp-121h]
  unsigned __int64 v80; // [rsp+30h] [rbp-120h] BYREF
  unsigned int v81; // [rsp+38h] [rbp-118h]
  __int64 v82; // [rsp+40h] [rbp-110h] BYREF
  unsigned int v83; // [rsp+48h] [rbp-108h]
  unsigned int *v84; // [rsp+50h] [rbp-100h] BYREF
  __int64 v85; // [rsp+58h] [rbp-F8h]
  _BYTE v86[32]; // [rsp+60h] [rbp-F0h] BYREF
  __int64 *v87; // [rsp+80h] [rbp-D0h] BYREF
  __int64 v88; // [rsp+88h] [rbp-C8h]
  _BYTE v89[64]; // [rsp+90h] [rbp-C0h] BYREF
  unsigned int *v90; // [rsp+D0h] [rbp-80h] BYREF
  __int64 v91; // [rsp+D8h] [rbp-78h]
  _BYTE v92[112]; // [rsp+E0h] [rbp-70h] BYREF

  v11 = a1;
  if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
    v13 = *(__int64 ***)(a1 - 8);
  else
    v13 = (__int64 **)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
  v14 = (__int64)*v13;
  v15 = *(_DWORD *)(**v13 + 8) >> 8;
  v74 = v15;
  sub_14C2530((__int64)&v80, *v13, a3, 0, a2, a1, 0, 0);
  v16 = sub_14C23D0(v14, a3, 0, a2, a1, 0);
  v77 = v15 - v16 + 1;
  v87 = (__int64 *)v89;
  v88 = 0x800000000LL;
  v75 = ((*(_DWORD *)(a1 + 20) & 0xFFFFFFFu) >> 1) - 1;
  if ( (*(_DWORD *)(a1 + 20) & 0xFFFFFFFu) >> 1 != 1 )
  {
    v18 = *(_BYTE *)(a1 + 23);
    v20 = 0;
    while ( 1 )
    {
      ++v20;
      v78 = v18 & 0x40;
      if ( (v18 & 0x40) != 0 )
        v21 = *(_QWORD *)(a1 - 8);
      else
        v21 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
      v22 = *(_QWORD *)(v21 + 24LL * (unsigned int)(2 * v20));
      if ( v81 <= 0x40 )
      {
        if ( (*(_QWORD *)(v22 + 24) & v80) != 0 )
        {
LABEL_18:
          v31 = (unsigned int)v88;
          if ( (unsigned int)v88 >= HIDWORD(v88) )
            goto LABEL_54;
          goto LABEL_19;
        }
        if ( v83 > 0x40 )
        {
LABEL_10:
          if ( !(unsigned __int8)sub_16A5A00(&v82, (__int64 *)(v22 + 24)) )
            goto LABEL_18;
          v23 = *(_QWORD *)(v22 + 24);
          goto LABEL_12;
        }
      }
      else
      {
        if ( (unsigned __int8)sub_16A59B0((__int64 *)&v80, (__int64 *)(v22 + 24)) )
          goto LABEL_18;
        if ( v83 > 0x40 )
          goto LABEL_10;
      }
      v23 = *(_QWORD *)(v22 + 24);
      if ( (v82 & ~v23) != 0 )
      {
        v31 = (unsigned int)v88;
        if ( (unsigned int)v88 >= HIDWORD(v88) )
        {
LABEL_54:
          sub_16CD150((__int64)&v87, v89, 0, 8, v16, v17);
          v31 = (unsigned int)v88;
        }
LABEL_19:
        v87[v31] = v22;
        v18 = *(_BYTE *)(a1 + 23);
        LODWORD(v88) = v88 + 1;
        v78 = v18 & 0x40;
        goto LABEL_20;
      }
LABEL_12:
      v24 = *(_DWORD *)(v22 + 32);
      v25 = v24 + 1;
      v26 = 1LL << ((unsigned __int8)v24 - 1);
      if ( v24 > 0x40 )
      {
        if ( (*(_QWORD *)(v23 + 8LL * ((v24 - 1) >> 6)) & v26) != 0 )
        {
          v59 = sub_16A5810(v22 + 24);
          v25 = v24 + 1;
          v27 = v59;
LABEL_16:
          v30 = v25 - v27;
          goto LABEL_17;
        }
        v30 = v25 - sub_16A57B0(v22 + 24);
      }
      else
      {
        if ( (v26 & v23) != 0 )
        {
          v27 = 64;
          v28 = ~(v23 << (64 - (unsigned __int8)v24));
          if ( v28 )
          {
            _BitScanReverse64(&v29, v28);
            v27 = v29 ^ 0x3F;
          }
          goto LABEL_16;
        }
        v30 = 1;
        if ( v23 )
        {
          _BitScanReverse64(&v23, v23);
          v30 = 65 - (v23 ^ 0x3F);
        }
      }
LABEL_17:
      if ( v77 < v30 )
        goto LABEL_18;
LABEL_20:
      if ( v75 == v20 )
      {
        v11 = a1;
        goto LABEL_22;
      }
    }
  }
  v78 = *(_BYTE *)(a1 + 23) & 0x40;
LABEL_22:
  if ( v78 )
    v32 = *(_QWORD *)(v11 - 8);
  else
    v32 = v11 - 24LL * (*(_DWORD *)(v11 + 20) & 0xFFFFFFF);
  v33 = *(_BYTE *)(sub_157ED60(*(_QWORD *)(v32 + 24)) + 16);
  LODWORD(v85) = v81;
  if ( v81 <= 0x40 )
  {
    v34 = v80;
LABEL_26:
    v35 = (unsigned int *)(v82 | v34);
    goto LABEL_27;
  }
  sub_16A4FD0((__int64)&v84, (const void **)&v80);
  if ( (unsigned int)v85 <= 0x40 )
  {
    v34 = (unsigned __int64)v84;
    goto LABEL_26;
  }
  sub_16A89F0((__int64 *)&v84, &v82);
  v62 = v85;
  v35 = v84;
  LODWORD(v85) = 0;
  LODWORD(v91) = v62;
  v90 = v84;
  if ( v62 > 0x40 )
  {
    v36 = v74 - sub_16A5940((__int64)&v90);
    if ( v35 )
    {
      j_j___libc_free_0_0(v35);
      if ( (unsigned int)v85 > 0x40 )
      {
        if ( v84 )
          j_j___libc_free_0_0(v84);
      }
    }
    goto LABEL_28;
  }
LABEL_27:
  v36 = v74 - sub_39FAC40(v35);
LABEL_28:
  if ( v33 != 31 )
  {
    LOBYTE(v35) = v36 <= 0x3F && (_DWORD)v88 == 0;
    if ( (_BYTE)v35 )
    {
      if ( ((*(_DWORD *)(v11 + 20) & 0xFFFFFFFu) >> 1) - 1 == 1LL << v36 )
      {
        v90 = *(unsigned int **)(v11 + 40);
        v65 = sub_13CF970(v11);
        v68 = (_QWORD *)sub_1AAB350(
                          *(_QWORD *)(v65 + 24),
                          (__int64 *)&v90,
                          1,
                          (char *)byte_3F871B3,
                          0,
                          0,
                          a4,
                          a5,
                          a6,
                          a7,
                          v66,
                          v67,
                          a10,
                          a11,
                          0);
        v69 = sub_13CF970(v11);
        sub_1593B40((_QWORD *)(v69 + 24), (__int64)v68);
        v70 = v68[6];
        if ( v70 )
          v70 -= 24;
        sub_1AA8CA0(v68, v70, 0, 0);
        v71 = sub_157EBA0((__int64)v68);
        v72 = sub_16498A0(v11);
        v73 = sub_1648A60(56, 0);
        if ( v73 )
          sub_15F82A0((__int64)v73, v72, v71);
        sub_1B44FE0(v71);
        v42 = v87;
        goto LABEL_73;
      }
    }
  }
  v37 = *(_QWORD *)(v11 + 48) == 0;
  v90 = (unsigned int *)v92;
  v91 = 0x800000000LL;
  if ( v37 && *(__int16 *)(v11 + 18) >= 0 )
  {
    v42 = v87;
    v76 = &v87[(unsigned int)v88];
    if ( v76 == v87 )
    {
      LOBYTE(v35) = (_DWORD)v88 != 0;
      goto LABEL_73;
    }
    goto LABEL_36;
  }
  v38 = sub_1625790(v11, 2);
  if ( !v38 )
    goto LABEL_82;
  v39 = *(_BYTE **)(v38 - 8LL * *(unsigned int *)(v38 + 8));
  if ( !v39 )
    goto LABEL_35;
  if ( *v39 )
  {
LABEL_82:
    v42 = v87;
    v43 = v88;
    v76 = &v87[(unsigned int)v88];
    if ( v76 != v87 )
      goto LABEL_36;
    goto LABEL_70;
  }
  v40 = sub_161E970((__int64)v39);
  if ( v41 == 14 )
  {
    if ( *(_QWORD *)v40 == 0x775F68636E617262LL && *(_DWORD *)(v40 + 8) == 1751607653 && *(_WORD *)(v40 + 12) == 29556 )
    {
      sub_1B43970(v11, (__int64)&v90);
      v42 = v87;
      v44 = *(_DWORD *)(v11 + 20) & 0xFFFFFFF;
      v79 = v44 >> 1 == (_DWORD)v91;
      v76 = &v87[(unsigned int)v88];
      if ( v87 == v76 )
      {
LABEL_67:
        if ( v79 )
        {
          v60 = (unsigned int)v91;
          if ( (unsigned int)v91 > 1uLL )
          {
            v35 = v90;
            v84 = (unsigned int *)v86;
            v63 = (unsigned int *)v86;
            v85 = 0x800000000LL;
            if ( (unsigned int)v91 > 8uLL )
            {
              sub_16CD150((__int64)&v84, v86, (unsigned int)v91, 4, v47, v48);
              v63 = &v84[(unsigned int)v85];
            }
            v64 = 0;
            do
            {
              v63[v64] = *(_QWORD *)&v35[2 * v64];
              ++v64;
            }
            while ( v60 - v64 > 0 );
            LODWORD(v85) = v85 + v60;
            sub_1B42940(v11, v84, (unsigned int)v85);
            if ( v84 != (unsigned int *)v86 )
              _libc_free((unsigned __int64)v84);
          }
        }
        v43 = v88;
        goto LABEL_70;
      }
LABEL_37:
      v45 = v42;
      while ( 1 )
      {
        v49 = (v44 >> 1) - 1;
        v50 = sub_1B44DF0(v11, 0, v11, v49, *v45);
        v35 = (unsigned int *)v50;
        v52 = v51;
        if ( v51 == v49 )
        {
          v54 = v11;
          v35 = (unsigned int *)v11;
          v53 = -2;
          v52 = -2;
        }
        else
        {
          v53 = v51;
          v54 = v50;
        }
        if ( v79 )
        {
          v55 = &v90[2 * (unsigned int)v91 - 2];
          v56 = &v90[2 * (v53 + 1)];
          v57 = *(_QWORD *)v56;
          *(_QWORD *)v56 = *(_QWORD *)v55;
          *(_QWORD *)v55 = v57;
          LODWORD(v91) = v91 - 1;
        }
        v58 = 24;
        if ( v53 != -2 )
          v58 = 24LL * (unsigned int)(2 * v53 + 3);
        v46 = (*((_BYTE *)v35 + 23) & 0x40) != 0 ? *((_QWORD *)v35 - 1) : v54 - 24LL * (v35[5] & 0xFFFFFFF);
        ++v45;
        sub_157F2D0(*(_QWORD *)(v46 + v58), *(_QWORD *)(v11 + 40), 0);
        sub_15FFDB0(v11, (__int64)v35, v52);
        if ( v76 == v45 )
          break;
        v44 = *(_DWORD *)(v11 + 20) & 0xFFFFFFF;
      }
      goto LABEL_67;
    }
    goto LABEL_82;
  }
LABEL_35:
  v42 = v87;
  v43 = v88;
  v76 = &v87[(unsigned int)v88];
  if ( v87 != v76 )
  {
LABEL_36:
    v79 = 0;
    v44 = *(_DWORD *)(v11 + 20) & 0xFFFFFFF;
    goto LABEL_37;
  }
LABEL_70:
  LOBYTE(v35) = v43 != 0;
  if ( v90 != (unsigned int *)v92 )
    _libc_free((unsigned __int64)v90);
  v42 = v87;
LABEL_73:
  if ( v42 != (__int64 *)v89 )
    _libc_free((unsigned __int64)v42);
  if ( v83 > 0x40 && v82 )
    j_j___libc_free_0_0(v82);
  if ( v81 > 0x40 && v80 )
    j_j___libc_free_0_0(v80);
  return (unsigned int)v35;
}
