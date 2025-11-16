// Function: sub_2AEBCD0
// Address: 0x2aebcd0
//
__int64 __fastcall sub_2AEBCD0(_QWORD *a1, int a2, unsigned int a3, unsigned int a4, __int64 a5, char a6)
{
  unsigned __int64 v7; // r13
  __int64 v9; // rdi
  unsigned __int64 v10; // rax
  __int64 v11; // r9
  unsigned __int64 v12; // r15
  __int64 v13; // rdx
  __int64 v14; // r8
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // rax
  __int64 v17; // rdx
  unsigned __int64 v18; // rcx
  unsigned __int64 v19; // rax
  unsigned __int64 *v20; // rdx
  unsigned __int64 v21; // rax
  int v22; // r13d
  unsigned __int64 v23; // rdx
  int v25; // ecx
  unsigned __int64 v26; // rax
  unsigned __int64 v27; // rcx
  __int64 *v28; // rdx
  unsigned int v29; // r13d
  unsigned __int64 v30; // rax
  unsigned __int64 v31; // rcx
  __int64 v32; // rbx
  __int64 v33; // r13
  unsigned __int64 v34; // rax
  char *v35; // r15
  __int64 v36; // rbx
  __int64 v37; // rax
  __int64 v38; // rbx
  char *v39; // rbx
  unsigned int v40; // r12d
  unsigned int v41; // r12d
  unsigned int v42; // r12d
  unsigned int v43; // r12d
  __int64 v44; // rbx
  unsigned __int64 v45; // r12
  unsigned __int64 v46; // rdi
  unsigned __int64 v47; // rdi
  unsigned int v48; // ebx
  unsigned int v49; // ebx
  unsigned int v50; // ebx
  unsigned int *v51; // rax
  char v52; // al
  int v53; // eax
  char v54; // al
  unsigned int v55; // ebx
  unsigned int v56; // ecx
  char v57; // dl
  unsigned int v58; // eax
  unsigned int v59; // ecx
  char v60; // [rsp+8h] [rbp-6F8h]
  char v61; // [rsp+8h] [rbp-6F8h]
  unsigned int v63; // [rsp+14h] [rbp-6ECh]
  char v64; // [rsp+1Bh] [rbp-6E5h]
  char v65; // [rsp+1Ch] [rbp-6E4h]
  unsigned int v66; // [rsp+1Ch] [rbp-6E4h]
  unsigned int v67; // [rsp+1Ch] [rbp-6E4h]
  char *v68; // [rsp+20h] [rbp-6E0h]
  int v69; // [rsp+20h] [rbp-6E0h]
  unsigned int v70; // [rsp+20h] [rbp-6E0h]
  __int64 v71; // [rsp+28h] [rbp-6D8h] BYREF
  int v72; // [rsp+30h] [rbp-6D0h] BYREF
  char v73; // [rsp+34h] [rbp-6CCh]
  unsigned int v74; // [rsp+38h] [rbp-6C8h] BYREF
  char v75; // [rsp+3Ch] [rbp-6C4h]
  __int64 v76; // [rsp+40h] [rbp-6C0h]
  __int64 v77; // [rsp+48h] [rbp-6B8h]
  unsigned __int64 v78; // [rsp+50h] [rbp-6B0h]
  __int64 v79; // [rsp+58h] [rbp-6A8h]
  unsigned __int64 v80; // [rsp+60h] [rbp-6A0h]
  __int64 v81; // [rsp+68h] [rbp-698h]
  unsigned int *v82; // [rsp+70h] [rbp-690h] BYREF
  __int64 v83; // [rsp+78h] [rbp-688h]
  _BYTE v84[64]; // [rsp+80h] [rbp-680h] BYREF
  unsigned __int64 v85; // [rsp+C0h] [rbp-640h] BYREF
  unsigned int v86; // [rsp+C8h] [rbp-638h]
  char v87; // [rsp+D0h] [rbp-630h] BYREF

  v7 = a4;
  v71 = a5;
  v9 = a1[56];
  v64 = BYTE4(a5);
  if ( BYTE4(a5) )
  {
    v16 = sub_DFB1B0(v9);
    v80 = v16;
    v12 = v16;
    v81 = v17;
    if ( v16 < v7 )
    {
      v73 = 1;
      v14 = 2;
      LODWORD(v19) = 0;
      v72 = 0;
      goto LABEL_8;
    }
    v14 = 2;
    v15 = v16 / v7;
  }
  else
  {
    v10 = sub_DFB1B0(v9);
    v80 = v10;
    v12 = v10;
    v81 = v13;
    if ( v10 < v7 )
    {
      LODWORD(v19) = 0;
      v14 = 1;
      goto LABEL_7;
    }
    v14 = 1;
    v15 = v10 / v7;
  }
  _BitScanReverse64(&v18, v15);
  v19 = 0x8000000000000000LL >> ((unsigned __int8)v18 ^ 0x3Fu);
LABEL_7:
  v72 = v19;
  v73 = v64;
LABEL_8:
  v20 = (unsigned __int64 *)&v72;
  if ( (unsigned int)v71 <= (unsigned int)v19 )
    v20 = (unsigned __int64 *)&v71;
  v21 = *v20;
  v78 = v21;
  v22 = v21;
  v23 = HIDWORD(v21);
  v85 = v21;
  v72 = v21;
  v73 = BYTE4(v21);
  v63 = v21;
  if ( !(_DWORD)v21 )
  {
    LODWORD(v85) = 1;
    BYTE4(v85) = 0;
    return v85;
  }
  v25 = v21;
  if ( BYTE4(v21) )
  {
    v60 = BYTE4(v21);
    v66 = v14;
    v69 = v21;
    v52 = sub_B2D610(a1[61], 96);
    v25 = v69;
    v14 = v66;
    LOBYTE(v23) = v60;
    if ( v52 )
    {
      v85 = sub_B2D7D0(a1[61], 96);
      v53 = sub_A71EB0((__int64 *)&v85);
      LOBYTE(v23) = v60;
      v14 = v66;
      v25 = v22 * v53;
    }
  }
  if ( a2 )
  {
    v67 = v14;
    v61 = v23;
    v70 = v25;
    v54 = sub_2AB31C0((__int64)a1, 1);
    v14 = v67;
    v55 = (v54 == 0) + a2 - 1;
    if ( (v54 == 0) + a2 != 1 && v55 <= v70 )
    {
      if ( !a6 )
      {
        _BitScanReverse(&v56, v55);
        v57 = 0;
        v58 = 0x80000000 >> (v56 ^ 0x1F);
LABEL_82:
        LODWORD(v85) = v58;
        BYTE4(v85) = v57;
        return v85;
      }
      v57 = v61;
      if ( (((v54 == 0) + a2 - 1) & ((v54 == 0) + a2 - 2)) == 0 )
      {
        _BitScanReverse(&v59, v55);
        v58 = 0x80000000 >> (v59 ^ 0x1F);
        goto LABEL_82;
      }
    }
  }
  v65 = v73;
  if ( !byte_500E468
    && ((unsigned int)sub_23DF0D0(dword_500E3E8)
     || !(unsigned __int8)sub_DFB2D0(a1[56]) && (!(_BYTE)qword_500CE88 || !*(_BYTE *)(a1[55] + 592LL))) )
  {
    goto LABEL_55;
  }
  LODWORD(v26) = 0;
  if ( a3 <= v12 )
  {
    _BitScanReverse64(&v27, v12 / a3);
    v26 = 0x8000000000000000LL >> ((unsigned __int8)v27 ^ 0x3Fu);
  }
  v74 = v26;
  v28 = &v71;
  v75 = v64;
  if ( (unsigned int)v71 > (unsigned int)v26 )
    v28 = (__int64 *)&v74;
  v29 = 2 * v22;
  v79 = *v28;
  v85 = v79;
  v74 = v79;
  v75 = BYTE4(v79);
  v82 = (unsigned int *)v84;
  v83 = 0x800000000LL;
  v30 = 0;
  while ( !v65 || v75 )
  {
    v31 = v30;
    if ( v29 > v74 )
      goto LABEL_28;
    LODWORD(v76) = v29;
    BYTE4(v76) = v65;
    v32 = v76;
    if ( v30 + 1 > HIDWORD(v83) )
    {
      sub_C8D5F0((__int64)&v82, v84, v30 + 1, 8u, v14, v11);
      v30 = (unsigned int)v83;
    }
    v29 *= 2;
    *(_QWORD *)&v82[2 * v30] = v32;
    v30 = (unsigned int)(v83 + 1);
    LODWORD(v83) = v83 + 1;
  }
  v31 = (unsigned int)v30;
LABEL_28:
  sub_2AE96E0((__int64)&v85, (__int64)a1, v82, v31);
  LODWORD(v33) = v86 - 1;
  if ( (int)(v86 - 1) < 0 )
    goto LABEL_39;
  v33 = (int)v33;
  while ( 1 )
  {
    v34 = v85 + 192 * v33;
    v35 = *(char **)(v34 + 144);
    v36 = 8LL * *(unsigned int *)(v34 + 152);
    v68 = &v35[v36];
    v37 = v36 >> 3;
    v38 = v36 >> 5;
    if ( v38 )
    {
      v39 = &v35[32 * v38];
      while ( 1 )
      {
        v43 = *((_DWORD *)v35 + 1);
        if ( v43 > (unsigned int)sub_DFB120(a1[56]) )
          goto LABEL_37;
        v40 = *((_DWORD *)v35 + 3);
        if ( v40 > (unsigned int)sub_DFB120(a1[56]) )
        {
          v35 += 8;
          goto LABEL_37;
        }
        v41 = *((_DWORD *)v35 + 5);
        if ( v41 > (unsigned int)sub_DFB120(a1[56]) )
        {
          v35 += 16;
          goto LABEL_37;
        }
        v42 = *((_DWORD *)v35 + 7);
        if ( v42 > (unsigned int)sub_DFB120(a1[56]) )
        {
          v35 += 24;
          goto LABEL_37;
        }
        v35 += 32;
        if ( v39 == v35 )
        {
          v37 = (v68 - v35) >> 3;
          break;
        }
      }
    }
    if ( v37 == 2 )
      goto LABEL_61;
    if ( v37 == 3 )
    {
      v48 = *((_DWORD *)v35 + 1);
      if ( v48 > (unsigned int)sub_DFB120(a1[56]) )
        goto LABEL_37;
      v35 += 8;
LABEL_61:
      v49 = *((_DWORD *)v35 + 1);
      if ( v49 > (unsigned int)sub_DFB120(a1[56]) )
        goto LABEL_37;
      v35 += 8;
      goto LABEL_63;
    }
    if ( v37 != 1 )
      break;
LABEL_63:
    v50 = *((_DWORD *)v35 + 1);
    if ( v50 <= (unsigned int)sub_DFB120(a1[56]) )
      break;
LABEL_37:
    if ( v68 == v35 )
      break;
    if ( (int)--v33 < 0 )
      goto LABEL_39;
  }
  v51 = &v82[2 * v33];
  v63 = *v51;
  v65 = *((_BYTE *)v51 + 4);
LABEL_39:
  v77 = sub_DFB300((__int64 *)a1[56], a3, v64);
  if ( (_DWORD)v77 && (!v65 || BYTE4(v77)) && (unsigned int)v77 > v63 )
  {
    v65 = BYTE4(v77);
    v63 = v77;
  }
  sub_2AC31B0((__int64)a1);
  v44 = v85;
  v45 = v85 + 192LL * v86;
  if ( v85 != v45 )
  {
    do
    {
      v45 -= 192LL;
      v46 = *(_QWORD *)(v45 + 144);
      if ( v46 != v45 + 160 )
        _libc_free(v46);
      if ( (*(_BYTE *)(v45 + 104) & 1) == 0 )
        sub_C7D6A0(*(_QWORD *)(v45 + 112), 8LL * *(unsigned int *)(v45 + 120), 4);
      v47 = *(_QWORD *)(v45 + 48);
      if ( v47 != v45 + 64 )
        _libc_free(v47);
      if ( (*(_BYTE *)(v45 + 8) & 1) == 0 )
        sub_C7D6A0(*(_QWORD *)(v45 + 16), 8LL * *(unsigned int *)(v45 + 24), 4);
    }
    while ( v44 != v45 );
    v45 = v85;
  }
  if ( (char *)v45 != &v87 )
    _libc_free(v45);
  if ( v82 != (unsigned int *)v84 )
    _libc_free((unsigned __int64)v82);
LABEL_55:
  LODWORD(v85) = v63;
  BYTE4(v85) = v65;
  return v85;
}
