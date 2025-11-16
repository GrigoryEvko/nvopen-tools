// Function: sub_1EEAB30
// Address: 0x1eeab30
//
__int64 __fastcall sub_1EEAB30(__int64 a1, __int64 **a2, _QWORD *a3, char a4, unsigned int a5)
{
  __int64 *v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rbx
  __int64 (*v9)(); // rax
  unsigned __int64 v10; // r12
  char v11; // r13
  int v12; // ecx
  int v13; // r8d
  int v14; // r9d
  unsigned __int16 *v15; // r15
  unsigned int v16; // eax
  __int16 v17; // cx
  unsigned __int64 v18; // rdi
  _WORD *v19; // rax
  __int16 *v20; // rsi
  unsigned __int16 v21; // cx
  __int16 *v22; // rax
  __int16 v23; // dx
  bool v24; // zf
  unsigned __int16 v25; // dx
  __int64 v26; // rax
  __int64 v27; // rcx
  _QWORD *v28; // rax
  _QWORD *v29; // rdx
  __int64 v30; // rax
  __int64 i; // r12
  unsigned __int16 *v32; // rax
  unsigned int v33; // edi
  __int16 v34; // cx
  _WORD *v35; // rsi
  __int64 v36; // r8
  unsigned __int16 v37; // cx
  __int16 v38; // r10
  __int16 *v39; // rsi
  int v40; // r13d
  unsigned int v41; // r12d
  __int64 v42; // rax
  __int64 v43; // rcx
  unsigned __int64 v44; // rax
  __int64 j; // rax
  __int64 v46; // rcx
  int v47; // eax
  _WORD *v48; // rdx
  _WORD *v49; // rsi
  unsigned __int64 v50; // rcx
  _WORD *v51; // rdx
  _QWORD *v52; // rax
  int v53; // eax
  _WORD *v55; // rdx
  __int64 v56; // r8
  unsigned int v57; // esi
  __int16 v58; // cx
  _WORD *v59; // rax
  __int64 v60; // rsi
  unsigned __int16 v61; // cx
  __int16 *v62; // rax
  __int16 v63; // r11
  __int64 v64; // rcx
  unsigned int v65; // esi
  __int16 v66; // ax
  _WORD *v67; // rsi
  __int16 *v68; // r8
  unsigned __int16 v69; // cx
  __int16 *v70; // rax
  __int16 v71; // si
  _BYTE *v72; // rax
  _BYTE *v76; // [rsp+18h] [rbp-98h]
  __int64 v77; // [rsp+20h] [rbp-90h]
  _QWORD *v78; // [rsp+28h] [rbp-88h]
  _QWORD *v79; // [rsp+30h] [rbp-80h]
  __int64 v80; // [rsp+38h] [rbp-78h]
  __int64 v81; // [rsp+48h] [rbp-68h]
  int v83; // [rsp+58h] [rbp-58h]
  unsigned __int16 v85; // [rsp+5Eh] [rbp-52h]
  __int64 v86; // [rsp+60h] [rbp-50h] BYREF
  unsigned __int64 v87; // [rsp+68h] [rbp-48h] BYREF
  __int64 v88; // [rsp+70h] [rbp-40h]
  __int64 v89; // [rsp+78h] [rbp-38h]

  v6 = a2[5];
  v77 = a3[3];
  if ( v6 )
  {
    v81 = ((__int64 (__fastcall *)(_QWORD))v6)(*(_QWORD *)(a3[3] + 56LL));
    v8 = v7;
  }
  else
  {
    v8 = *((unsigned __int16 *)*a2 + 10);
    v81 = **a2;
  }
  v79 = *(_QWORD **)(a1 + 16);
  v9 = *(__int64 (**)())(**(_QWORD **)(*v79 + 16LL) + 112LL);
  if ( v9 == sub_1D00B10 )
  {
    v86 = 0;
    v87 = 0;
    v88 = 0;
    v89 = 0;
    BUG();
  }
  v10 = *(_QWORD *)(a1 + 32);
  v11 = 0;
  v76 = (_BYTE *)v10;
  v80 = *(_QWORD *)(v10 + 24);
  v87 = 0;
  v86 = v9();
  v88 = 0;
  LODWORD(v89) = 0;
  sub_13A49F0((__int64)&v87, *(_DWORD *)(v86 + 44), 0, v12, v13, v14);
  v78 = 0;
  v85 = 0;
  v15 = (unsigned __int16 *)(v81 + 2 * v8);
  v83 = 25;
  while ( 1 )
  {
    sub_2104340(&v86, v10);
    if ( (_QWORD *)v10 == a3 )
      break;
    if ( !v11 )
      goto LABEL_21;
LABEL_7:
    if ( !v85 )
      goto LABEL_28;
    if ( !v86 )
      BUG();
    v16 = *(_DWORD *)(*(_QWORD *)(v86 + 8) + 24LL * v85 + 16);
    v17 = v85 * (v16 & 0xF);
    v18 = v87;
    v19 = (_WORD *)(*(_QWORD *)(v86 + 56) + 2LL * (v16 >> 4));
    v20 = v19 + 1;
    v21 = *v19 + v17;
LABEL_10:
    v22 = v20;
    if ( v20 )
    {
      while ( (*(_QWORD *)(v87 + 8 * ((unsigned __int64)v21 >> 6)) & (1LL << v21)) == 0 )
      {
        v23 = *v22;
        v20 = 0;
        ++v22;
        if ( !v23 )
          goto LABEL_10;
        v21 += v23;
        if ( !v22 )
          goto LABEL_14;
      }
LABEL_28:
      if ( v15 == (unsigned __int16 *)v81 )
      {
LABEL_41:
        v18 = v87;
        goto LABEL_42;
      }
      v32 = (unsigned __int16 *)v81;
      while ( 1 )
      {
        v25 = *v32;
        if ( (*(_QWORD *)(v79[38] + 8 * ((unsigned __int64)*v32 >> 6)) & (1LL << *v32)) == 0 )
          break;
LABEL_30:
        if ( v15 == ++v32 )
          goto LABEL_41;
      }
      if ( !v86 )
        BUG();
      v33 = *(_DWORD *)(*(_QWORD *)(v86 + 8) + 24LL * v25 + 16);
      v34 = v25 * (v33 & 0xF);
      v35 = (_WORD *)(*(_QWORD *)(v86 + 56) + 2LL * (v33 >> 4));
      v18 = v87;
      v36 = (__int64)(v35 + 1);
      v37 = *v35 + v34;
LABEL_36:
      v39 = (__int16 *)v36;
      while ( v39 )
      {
        v36 = *(_QWORD *)(v87 + 8 * ((unsigned __int64)v37 >> 6)) & (1LL << v37);
        if ( v36 )
          goto LABEL_30;
        v38 = *v39++;
        v37 += v38;
        if ( !v38 )
          goto LABEL_36;
      }
      if ( !v25 )
        goto LABEL_42;
      if ( !--v83 )
      {
LABEL_40:
        v85 = v25;
LABEL_42:
        v40 = v85;
        goto LABEL_43;
      }
      goto LABEL_15;
    }
LABEL_14:
    v24 = v83-- == 1;
    v25 = v85;
    if ( v24 )
      goto LABEL_40;
LABEL_15:
    v26 = *(_QWORD *)(v10 + 32);
    v27 = v26 + 40LL * *(unsigned int *)(v10 + 40);
    if ( v26 != v27 )
    {
      while ( *(_BYTE *)v26 || *(int *)(v26 + 8) >= 0 )
      {
        v26 += 40;
        if ( v27 == v26 )
          goto LABEL_19;
      }
      v78 = (_QWORD *)v10;
      v83 = 25;
    }
LABEL_19:
    v85 = v25;
    if ( v10 == *(_QWORD *)(v80 + 32) )
      goto LABEL_42;
    v11 = 1;
LABEL_21:
    v28 = (_QWORD *)(*(_QWORD *)v10 & 0xFFFFFFFFFFFFFFF8LL);
    v29 = v28;
    if ( !v28 )
      BUG();
    v10 = *(_QWORD *)v10 & 0xFFFFFFFFFFFFFFF8LL;
    v30 = *v28;
    if ( (v30 & 4) == 0 && (*((_BYTE *)v29 + 46) & 4) != 0 )
    {
      for ( i = v30; ; i = *(_QWORD *)v10 )
      {
        v10 = i & 0xFFFFFFFFFFFFFFF8LL;
        if ( (*(_BYTE *)(v10 + 46) & 4) == 0 )
          break;
      }
    }
  }
  if ( v15 == (unsigned __int16 *)v81 )
  {
LABEL_77:
    if ( a4 )
    {
      v72 = v76;
      if ( (*v76 & 4) == 0 )
      {
        v72 = v76;
        if ( (v76[46] & 8) != 0 )
        {
          do
            v72 = (_BYTE *)*((_QWORD *)v72 + 1);
          while ( (v72[46] & 8) != 0 );
        }
        else
        {
          v72 = v76;
        }
      }
      sub_2104340(&v86, *((_QWORD *)v72 + 1));
    }
    v78 = a3;
    goto LABEL_7;
  }
  v55 = (_WORD *)v81;
  while ( 1 )
  {
    v40 = (unsigned __int16)*v55;
    if ( (*(_QWORD *)(v79[38] + 8 * ((unsigned __int64)(unsigned __int16)v40 >> 6)) & (1LL << *v55)) == 0 )
      break;
LABEL_62:
    if ( v15 == ++v55 )
      goto LABEL_77;
  }
  if ( !v86 )
    BUG();
  v18 = v87;
  v56 = 24LL * (unsigned __int16)v40;
  v57 = *(_DWORD *)(*(_QWORD *)(v86 + 8) + v56 + 16);
  v58 = v40 * (v57 & 0xF);
  v59 = (_WORD *)(*(_QWORD *)(v86 + 56) + 2LL * (v57 >> 4));
  v60 = (__int64)(v59 + 1);
  v61 = *v59 + v58;
LABEL_66:
  v62 = (__int16 *)v60;
  if ( v60 )
  {
    do
    {
      v60 = *(_QWORD *)(v87 + 8 * ((unsigned __int64)v61 >> 6)) & (1LL << v61);
      if ( v60 )
        goto LABEL_62;
      v63 = *v62++;
      if ( !v63 )
        goto LABEL_66;
      v61 += v63;
    }
    while ( v62 );
  }
  v64 = *(_QWORD *)(a1 + 96);
  if ( !v64 )
    BUG();
  v65 = *(_DWORD *)(*(_QWORD *)(v64 + 8) + v56 + 16);
  v66 = v65 & 0xF;
  v67 = (_WORD *)(*(_QWORD *)(v64 + 56) + 2LL * (v65 >> 4));
  v68 = v67 + 1;
  v69 = *v67 + v40 * v66;
LABEL_72:
  v70 = v68;
  if ( v68 )
  {
    while ( (*(_QWORD *)(*(_QWORD *)(a1 + 104) + 8 * ((unsigned __int64)v69 >> 6)) & (1LL << v69)) == 0 )
    {
      v71 = *v70;
      v68 = 0;
      ++v70;
      if ( !v71 )
        goto LABEL_72;
      v69 += v71;
      if ( !v70 )
        goto LABEL_76;
    }
    goto LABEL_62;
  }
LABEL_76:
  v78 = (_QWORD *)(v80 + 24);
LABEL_43:
  _libc_free(v18);
  v41 = (unsigned __int16)v40;
  if ( v78 != (_QWORD *)(v77 + 24) )
  {
    v42 = *(_QWORD *)(a1 + 32);
    if ( a4 )
    {
      if ( !v42 )
        BUG();
      if ( (*(_BYTE *)v42 & 4) == 0 && (*(_BYTE *)(v42 + 46) & 8) != 0 )
      {
        do
          v42 = *(_QWORD *)(v42 + 8);
        while ( (*(_BYTE *)(v42 + 46) & 8) != 0 );
      }
      v42 = *(_QWORD *)(v42 + 8);
    }
    if ( !v42 )
      BUG();
    if ( (*(_BYTE *)v42 & 4) == 0 && (*(_BYTE *)(v42 + 46) & 8) != 0 )
    {
      do
        v42 = *(_QWORD *)(v42 + 8);
      while ( (*(_BYTE *)(v42 + 46) & 8) != 0 );
    }
    v41 = (unsigned __int16)v40;
    v86 = *(_QWORD *)(v42 + 8);
    v43 = sub_1EEA6B0((unsigned int *)a1, (unsigned __int16)v40, (__int64)a2, a5, v78, &v86);
    v44 = *v78 & 0xFFFFFFFFFFFFFFF8LL;
    if ( !v44 )
      BUG();
    if ( (*(_QWORD *)v44 & 4) == 0 && (*(_BYTE *)(v44 + 46) & 4) != 0 )
    {
      for ( j = *(_QWORD *)v44; ; j = *(_QWORD *)v44 )
      {
        v44 = j & 0xFFFFFFFFFFFFFFF8LL;
        if ( (*(_BYTE *)(v44 + 46) & 4) == 0 )
          break;
      }
    }
    *(_QWORD *)(v43 + 8) = v44;
    v46 = *(_QWORD *)(a1 + 96);
    if ( !v46 )
      BUG();
    v47 = *(_DWORD *)(*(_QWORD *)(v46 + 8) + 24LL * (unsigned __int16)v40 + 16) & 0xF;
    v48 = (_WORD *)(*(_QWORD *)(v46 + 56)
                  + 2LL * (*(_DWORD *)(*(_QWORD *)(v46 + 8) + 24LL * (unsigned __int16)v40 + 16) >> 4));
    v50 = (unsigned int)(v40 * v47);
    v49 = v48 + 1;
    LOWORD(v50) = *v48 + v40 * v47;
    while ( 1 )
    {
      v51 = v49;
      if ( !v49 )
        break;
      while ( 1 )
      {
        ++v51;
        v52 = (_QWORD *)(*(_QWORD *)(a1 + 104) + ((v50 >> 3) & 0x1FF8));
        *v52 &= ~(1LL << v50);
        v53 = (unsigned __int16)*(v51 - 1);
        v49 = 0;
        if ( !(_WORD)v53 )
          break;
        v50 = (unsigned int)(v53 + v50);
        if ( !v51 )
          return v41;
      }
    }
  }
  return v41;
}
