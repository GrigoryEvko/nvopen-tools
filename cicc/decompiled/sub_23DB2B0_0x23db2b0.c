// Function: sub_23DB2B0
// Address: 0x23db2b0
//
__int64 __fastcall sub_23DB2B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int8 **v7; // rax
  _BYTE *v8; // r14
  int v9; // r15d
  __int64 v10; // rbx
  __int64 *v11; // r14
  int v12; // eax
  __int64 *v13; // rcx
  unsigned int v14; // eax
  unsigned __int64 v15; // rdx
  _DWORD *v16; // rdx
  unsigned int v17; // ebx
  unsigned int v18; // ebx
  char v19; // al
  int v20; // r12d
  unsigned __int8 *v21; // r15
  _BYTE *v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rdi
  unsigned int v25; // esi
  _QWORD *v26; // rcx
  _BYTE *v27; // r10
  unsigned __int8 *v28; // rdx
  int v29; // eax
  int v30; // ecx
  int v31; // r8d
  __int64 v33; // rax
  __int64 *v34; // rbx
  unsigned int v35; // r12d
  unsigned __int64 v36; // r8
  unsigned __int64 v37; // r8
  unsigned int v38; // eax
  unsigned int v39; // edx
  unsigned __int64 v40; // rdi
  unsigned __int64 v41; // rdi
  __int64 *v42; // rdi
  unsigned int v43; // eax
  __int64 *v44; // rsi
  unsigned int v45; // edx
  unsigned __int64 v46; // rax
  unsigned __int64 v47; // r12
  unsigned int v48; // r15d
  unsigned int v49; // r15d
  unsigned int v50; // eax
  unsigned int v51; // r14d
  _QWORD *v52; // rax
  unsigned int v53; // ebx
  int v54; // [rsp+4h] [rbp-CCh]
  unsigned __int64 v55; // [rsp+18h] [rbp-B8h]
  unsigned __int8 **v56; // [rsp+20h] [rbp-B0h]
  unsigned __int8 **v57; // [rsp+28h] [rbp-A8h]
  unsigned __int8 **v58; // [rsp+30h] [rbp-A0h]
  unsigned int v59; // [rsp+38h] [rbp-98h]
  unsigned int v60; // [rsp+3Ch] [rbp-94h]
  unsigned __int8 v61; // [rsp+3Ch] [rbp-94h]
  _DWORD *v62; // [rsp+40h] [rbp-90h] BYREF
  unsigned int v63; // [rsp+48h] [rbp-88h]
  unsigned __int64 v64; // [rsp+50h] [rbp-80h] BYREF
  unsigned int v65; // [rsp+58h] [rbp-78h]
  unsigned __int64 v66; // [rsp+60h] [rbp-70h] BYREF
  unsigned int v67; // [rsp+68h] [rbp-68h]
  unsigned __int64 v68; // [rsp+70h] [rbp-60h]
  unsigned int v69; // [rsp+78h] [rbp-58h]
  _DWORD *v70; // [rsp+80h] [rbp-50h] BYREF
  unsigned int v71; // [rsp+88h] [rbp-48h]
  unsigned __int64 v72; // [rsp+90h] [rbp-40h]
  unsigned int v73; // [rsp+98h] [rbp-38h]

  if ( !(unsigned __int8)sub_23DA1F0(a1, a2, a3, a4, a5, a6) )
    return 0;
  v7 = *(unsigned __int8 ***)(a1 + 120);
  v8 = *(_BYTE **)(a1 + 80);
  v57 = &v7[3 * *(unsigned int *)(a1 + 128)];
  if ( v7 == v57 )
  {
    v53 = sub_BCB060(*(_QWORD *)(*((_QWORD *)v8 - 4) + 8LL));
    v51 = sub_23DAC80(a1);
    if ( v53 <= v51 )
      return 0;
LABEL_134:
    v52 = (_QWORD *)sub_BD5C60(*(_QWORD *)(a1 + 80));
    return sub_BCCE00(v52, v51);
  }
  v56 = *(unsigned __int8 ***)(a1 + 120);
  v58 = v56;
  v9 = 0;
  do
  {
    v10 = *((_QWORD *)*v58 + 2);
    if ( v10 && *(_QWORD *)(v10 + 8) )
    {
      v61 = **v58 - 68;
      v20 = v9;
      v21 = *v58;
      while ( 1 )
      {
        v22 = *(_BYTE **)(v10 + 24);
        if ( *v22 <= 0x1Cu || v22 == v8 )
          goto LABEL_41;
        v23 = *(unsigned int *)(a1 + 112);
        v24 = *(_QWORD *)(a1 + 96);
        if ( !(_DWORD)v23 )
          goto LABEL_47;
        v25 = (v23 - 1) & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
        v26 = (_QWORD *)(v24 + 16LL * v25);
        v27 = (_BYTE *)*v26;
        if ( v22 != (_BYTE *)*v26 )
        {
          v30 = 1;
          while ( v27 != (_BYTE *)-4096LL )
          {
            v31 = v30 + 1;
            v25 = (v23 - 1) & (v30 + v25);
            v26 = (_QWORD *)(v24 + 16LL * v25);
            v27 = (_BYTE *)*v26;
            if ( v22 == (_BYTE *)*v26 )
              goto LABEL_46;
            v30 = v31;
          }
          goto LABEL_47;
        }
LABEL_46:
        if ( v26 != (_QWORD *)(v24 + 16 * v23) )
        {
LABEL_41:
          v10 = *(_QWORD *)(v10 + 8);
          if ( !v10 )
            goto LABEL_53;
        }
        else
        {
LABEL_47:
          if ( v61 > 1u )
            return 0;
          v28 = (v21[7] & 0x40) != 0
              ? (unsigned __int8 *)*((_QWORD *)v21 - 1)
              : &v21[-32 * (*((_DWORD *)v21 + 1) & 0x7FFFFFF)];
          v29 = sub_BCB060(*(_QWORD *)(*(_QWORD *)v28 + 8LL));
          if ( v20 )
          {
            if ( v29 != v20 )
              return 0;
          }
          v10 = *(_QWORD *)(v10 + 8);
          v20 = v29;
          if ( !v10 )
          {
LABEL_53:
            v9 = v20;
            break;
          }
        }
      }
    }
    v58 += 3;
  }
  while ( v57 != v58 );
  v54 = v9;
  v60 = sub_BCB060(*(_QWORD *)(*((_QWORD *)v8 - 4) + 8LL));
  while ( 1 )
  {
    v11 = (__int64 *)*v56;
    v12 = **v56;
    if ( (unsigned int)(v12 - 54) > 2 )
      goto LABEL_7;
    v13 = (*((_BYTE *)v11 + 7) & 0x40) != 0 ? (__int64 *)*(v11 - 1) : &v11[-4 * (*((_DWORD *)v11 + 1) & 0x7FFFFFF)];
    sub_9AC3E0(
      (__int64)&v66,
      v13[4],
      *(_QWORD *)(a1 + 16),
      0,
      *(_QWORD *)a1,
      *(_QWORD *)(a1 + 80),
      *(_QWORD *)(a1 + 24),
      1);
    v14 = v67;
    v71 = v67;
    if ( v67 > 0x40 )
    {
      sub_C43780((__int64)&v70, (const void **)&v66);
      v14 = v71;
      if ( v71 > 0x40 )
      {
        sub_C43D10((__int64)&v70);
        v14 = v71;
        v16 = v70;
        goto LABEL_17;
      }
      v15 = (unsigned __int64)v70;
    }
    else
    {
      v15 = v66;
    }
    v16 = (_DWORD *)((0xFFFFFFFFFFFFFFFFLL >> -(char)v14) & ~v15);
    if ( !v14 )
      v16 = 0;
LABEL_17:
    v63 = v14;
    v62 = v16;
    v65 = v60;
    if ( v60 > 0x40 )
      sub_C43690((__int64)&v64, 1, 0);
    else
      v64 = 1;
    sub_C49B30((__int64)&v70, (__int64)&v62, (__int64 *)&v64);
    v17 = v71;
    if ( v71 > 0x40 )
    {
      if ( v17 - (unsigned int)sub_C444A0((__int64)&v70) <= 0x40 )
      {
        v40 = (unsigned __int64)v70;
        if ( (unsigned __int64)v60 < *(_QWORD *)v70 )
LABEL_88:
          v18 = v60;
        else
          v18 = *v70;
        j_j___libc_free_0_0(v40);
        goto LABEL_22;
      }
      v40 = (unsigned __int64)v70;
      if ( v70 )
        goto LABEL_88;
    }
    else if ( v60 >= (unsigned __int64)v70 )
    {
      v18 = (unsigned int)v70;
      goto LABEL_22;
    }
    v18 = v60;
LABEL_22:
    if ( v65 > 0x40 && v64 )
      j_j___libc_free_0_0(v64);
    if ( v63 > 0x40 && v62 )
      j_j___libc_free_0_0((unsigned __int64)v62);
    if ( v60 == v18 )
      break;
    v19 = *(_BYTE *)v11;
    if ( *(_BYTE *)v11 == 55 )
    {
      if ( (*((_BYTE *)v11 + 7) & 0x40) != 0 )
        v44 = (__int64 *)*(v11 - 1);
      else
        v44 = &v11[-4 * (*((_DWORD *)v11 + 1) & 0x7FFFFFF)];
      sub_9AC3E0(
        (__int64)&v70,
        *v44,
        *(_QWORD *)(a1 + 16),
        0,
        *(_QWORD *)a1,
        *(_QWORD *)(a1 + 80),
        *(_QWORD *)(a1 + 24),
        1);
      v45 = v71;
      v65 = v71;
      if ( v71 > 0x40 )
      {
        sub_C43780((__int64)&v64, (const void **)&v70);
        v45 = v65;
        if ( v65 > 0x40 )
        {
          sub_C43D10((__int64)&v64);
          v48 = v65;
          v47 = v64;
          v63 = v65;
          v62 = (_DWORD *)v64;
          if ( v65 > 0x40 )
          {
            v49 = v48 - sub_C444A0((__int64)&v62);
            if ( v18 < v49 )
              v18 = v49;
            if ( v47 )
              j_j___libc_free_0_0(v47);
            goto LABEL_115;
          }
          goto LABEL_112;
        }
        v46 = v64;
      }
      else
      {
        v46 = (unsigned __int64)v70;
      }
      v47 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v45) & ~v46;
      if ( !v45 )
      {
LABEL_115:
        if ( v73 > 0x40 && v72 )
          j_j___libc_free_0_0(v72);
        if ( v71 > 0x40 && v70 )
          j_j___libc_free_0_0((unsigned __int64)v70);
        v19 = *(_BYTE *)v11;
        goto LABEL_30;
      }
LABEL_112:
      if ( v47 )
      {
        _BitScanReverse64(&v47, v47);
        if ( v18 < 64 - ((unsigned int)v47 ^ 0x3F) )
          v18 = 64 - (v47 ^ 0x3F);
      }
      goto LABEL_115;
    }
LABEL_30:
    if ( v19 == 56 )
    {
      v42 = (*((_BYTE *)v11 + 7) & 0x40) != 0 ? (__int64 *)*(v11 - 1) : &v11[-4 * (*((_DWORD *)v11 + 1) & 0x7FFFFFF)];
      v43 = v60
          + 1
          - sub_9AF8B0(*v42, *(_QWORD *)(a1 + 16), 0, *(_QWORD *)a1, *(_QWORD *)(a1 + 80), *(_QWORD *)(a1 + 24), 1);
      if ( v18 < v43 )
        v18 = v43;
    }
    if ( v18 >= v60 )
      break;
    *((_DWORD *)v56 + 3) = v18;
    if ( v69 > 0x40 && v68 )
      j_j___libc_free_0_0(v68);
    if ( v67 > 0x40 && v66 )
      j_j___libc_free_0_0(v66);
    LOBYTE(v12) = *(_BYTE *)v11;
LABEL_7:
    if ( (_BYTE)v12 != 48 && (_BYTE)v12 != 51 )
      goto LABEL_9;
    v33 = 32LL * (*((_DWORD *)v11 + 1) & 0x7FFFFFF);
    if ( (*((_BYTE *)v11 + 7) & 0x40) != 0 )
    {
      v34 = (__int64 *)*(v11 - 1);
      v11 = &v34[(unsigned __int64)v33 / 8];
    }
    else
    {
      v34 = &v11[v33 / 0xFFFFFFFFFFFFFFF8LL];
    }
    v35 = 0;
    if ( v11 != v34 )
    {
      while ( 1 )
      {
        sub_9AC3E0(
          (__int64)&v70,
          *v34,
          *(_QWORD *)(a1 + 16),
          0,
          *(_QWORD *)a1,
          *(_QWORD *)(a1 + 80),
          *(_QWORD *)(a1 + 24),
          1);
        v38 = v71;
        v67 = v71;
        if ( v71 <= 0x40 )
          break;
        sub_C43780((__int64)&v66, (const void **)&v70);
        v38 = v67;
        if ( v67 <= 0x40 )
        {
          v36 = v66;
LABEL_67:
          v37 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v38) & ~v36;
          if ( !v38 )
            goto LABEL_71;
          goto LABEL_68;
        }
        sub_C43D10((__int64)&v66);
        v37 = v66;
        v65 = v67;
        v64 = v66;
        if ( v67 > 0x40 )
        {
          v59 = v67;
          v55 = v66;
          v39 = v59 - sub_C444A0((__int64)&v64);
          if ( v35 < v39 )
            v35 = v39;
          if ( v55 )
            j_j___libc_free_0_0(v55);
          goto LABEL_71;
        }
LABEL_68:
        if ( v37 )
        {
          _BitScanReverse64(&v37, v37);
          if ( v35 < 64 - ((unsigned int)v37 ^ 0x3F) )
            v35 = 64 - (v37 ^ 0x3F);
        }
LABEL_71:
        if ( v60 <= v35 )
        {
          if ( v73 > 0x40 && v72 )
            j_j___libc_free_0_0(v72);
          if ( v71 > 0x40 )
          {
            v41 = (unsigned __int64)v70;
            if ( v70 )
              goto LABEL_99;
          }
          return 0;
        }
        if ( v73 > 0x40 && v72 )
          j_j___libc_free_0_0(v72);
        if ( v71 > 0x40 && v70 )
          j_j___libc_free_0_0((unsigned __int64)v70);
        v34 += 4;
        if ( v11 == v34 )
          goto LABEL_92;
      }
      v36 = (unsigned __int64)v70;
      goto LABEL_67;
    }
LABEL_92:
    *((_DWORD *)v56 + 3) = v35;
LABEL_9:
    v56 += 3;
    if ( v57 == v56 )
    {
      v50 = sub_23DAC80(a1);
      v51 = v50;
      if ( v50 >= v60 || v54 && v54 != v50 )
        return 0;
      goto LABEL_134;
    }
  }
  if ( v69 > 0x40 && v68 )
    j_j___libc_free_0_0(v68);
  if ( v67 > 0x40 )
  {
    v41 = v66;
    if ( v66 )
LABEL_99:
      j_j___libc_free_0_0(v41);
  }
  return 0;
}
