// Function: sub_1F2C610
// Address: 0x1f2c610
//
__int64 __fastcall sub_1F2C610(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // r14
  __int64 v4; // rbx
  __int64 i; // r15
  __int64 v6; // r13
  unsigned int v7; // r15d
  __int64 v9; // rax
  __int64 v10; // rbx
  _QWORD *v11; // rbx
  _QWORD *v12; // r14
  _QWORD *v13; // rdi
  _QWORD *v14; // rbx
  _QWORD *v15; // r13
  _QWORD *v16; // rdi
  __int64 v17; // rax
  __int64 v18; // r15
  __int64 v19; // r14
  __int64 v20; // r12
  __int64 v21; // rdx
  unsigned __int64 v22; // rsi
  unsigned __int64 v23; // rax
  unsigned __int8 v24; // bl
  __int64 v25; // rsi
  __int64 v26; // rax
  __int64 v27; // rbx
  _QWORD *v28; // rbx
  _QWORD *v29; // r13
  _QWORD *v30; // rdi
  _QWORD *v31; // rbx
  _QWORD *v32; // r13
  _QWORD *v33; // rdi
  __int64 *v34; // r12
  __int64 v35; // rax
  __int64 v36; // r13
  _QWORD *v37; // r13
  _QWORD *v38; // rbx
  _QWORD *v39; // rdi
  _QWORD *v40; // r13
  _QWORD *v41; // rbx
  _QWORD *v42; // rdi
  __int64 v43; // rax
  __int64 v44; // rax
  __int64 v45; // rax
  __int64 v46; // rax
  __int64 v47; // rax
  __int64 v48; // rax
  __int64 v49; // [rsp+0h] [rbp-4C0h]
  __int64 v50; // [rsp+10h] [rbp-4B0h]
  unsigned __int8 v51; // [rsp+20h] [rbp-4A0h]
  int v52; // [rsp+2Ch] [rbp-494h]
  unsigned __int8 v53; // [rsp+2Ch] [rbp-494h]
  __int64 v54; // [rsp+30h] [rbp-490h]
  __int64 v55; // [rsp+38h] [rbp-488h]
  unsigned __int8 v56; // [rsp+38h] [rbp-488h]
  char v57; // [rsp+4Fh] [rbp-471h] BYREF
  __int64 *v58[2]; // [rsp+50h] [rbp-470h] BYREF
  __int64 *v59; // [rsp+60h] [rbp-460h]
  _QWORD v60[2]; // [rsp+70h] [rbp-450h] BYREF
  _QWORD v61[2]; // [rsp+80h] [rbp-440h] BYREF
  _QWORD *v62; // [rsp+90h] [rbp-430h]
  _QWORD v63[6]; // [rsp+A0h] [rbp-420h] BYREF
  void *v64; // [rsp+D0h] [rbp-3F0h] BYREF
  int v65; // [rsp+D8h] [rbp-3E8h] BYREF
  char v66; // [rsp+DCh] [rbp-3E4h]
  __int64 v67; // [rsp+E0h] [rbp-3E0h]
  __m128i v68; // [rsp+E8h] [rbp-3D8h]
  __int64 v69; // [rsp+F8h] [rbp-3C8h]
  __int64 v70; // [rsp+100h] [rbp-3C0h]
  __m128i v71; // [rsp+108h] [rbp-3B8h]
  __int64 v72; // [rsp+118h] [rbp-3A8h]
  char v73; // [rsp+120h] [rbp-3A0h]
  _BYTE *v74; // [rsp+128h] [rbp-398h] BYREF
  __int64 v75; // [rsp+130h] [rbp-390h]
  _BYTE v76[352]; // [rsp+138h] [rbp-388h] BYREF
  char v77; // [rsp+298h] [rbp-228h]
  int v78; // [rsp+29Ch] [rbp-224h]
  __int64 v79; // [rsp+2A0h] [rbp-220h]
  _QWORD v80[11]; // [rsp+2B0h] [rbp-210h] BYREF
  _QWORD *v81; // [rsp+308h] [rbp-1B8h]
  unsigned int v82; // [rsp+310h] [rbp-1B0h]
  _BYTE v83[424]; // [rsp+318h] [rbp-1A8h] BYREF

  v2 = *(_QWORD *)(a1 + 232);
  v3 = *(_QWORD *)(v2 + 80);
  v55 = v2 + 72;
  if ( v3 != v2 + 72 )
  {
    do
    {
      if ( !v3 )
        BUG();
      v4 = *(_QWORD *)(v3 + 24);
      for ( i = v3 + 16; i != v4; v4 = *(_QWORD *)(v4 + 8) )
      {
        while ( 1 )
        {
          if ( !v4 )
            BUG();
          if ( *(_BYTE *)(v4 - 8) == 78 )
          {
            v6 = *(_QWORD *)(v4 - 48);
            if ( *(_BYTE *)(v6 + 16) )
              v6 = 0;
            if ( v6 == sub_15E26F0(*(__int64 **)(*(_QWORD *)(a1 + 232) + 40LL), 200, 0, 0) )
              break;
          }
          v4 = *(_QWORD *)(v4 + 8);
          if ( i == v4 )
            goto LABEL_12;
        }
        *(_BYTE *)(a1 + 464) = 1;
      }
LABEL_12:
      v3 = *(_QWORD *)(v3 + 8);
    }
    while ( v55 != v3 );
    v2 = *(_QWORD *)(a1 + 232);
  }
  v7 = 0;
  if ( !(unsigned __int8)sub_1560180(v2 + 112, 41) )
  {
    sub_143A950(v58, *(__int64 **)(a1 + 232));
    v7 = sub_1560180(*(_QWORD *)(a1 + 232) + 112LL, 50);
    if ( (_BYTE)v7 )
    {
      v9 = sub_15E0530((__int64)v58[0]);
      if ( sub_1602790(v9)
        || (v45 = sub_15E0530((__int64)v58[0]),
            v46 = sub_16033E0(v45),
            (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v46 + 48LL))(v46)) )
      {
        sub_15CA470(
          (__int64)v80,
          (__int64)"stack-protector",
          (__int64)"StackProtectorRequested",
          23,
          *(_QWORD *)(a1 + 232));
        sub_15CAB20((__int64)v80, "Stack protection applied to function ", 0x25u);
        sub_15C9340((__int64)v60, "Function", 8u, *(_QWORD *)(a1 + 232));
        v10 = sub_17C2270((__int64)v80, (__int64)v60);
        sub_15CAB20(v10, " due to a function attribute or command-line switch", 0x33u);
        v65 = *(_DWORD *)(v10 + 8);
        v66 = *(_BYTE *)(v10 + 12);
        v67 = *(_QWORD *)(v10 + 16);
        v68 = _mm_loadu_si128((const __m128i *)(v10 + 24));
        v69 = *(_QWORD *)(v10 + 40);
        v64 = &unk_49ECF68;
        v70 = *(_QWORD *)(v10 + 48);
        v71 = _mm_loadu_si128((const __m128i *)(v10 + 56));
        v73 = *(_BYTE *)(v10 + 80);
        if ( v73 )
          v72 = *(_QWORD *)(v10 + 72);
        v74 = v76;
        v75 = 0x400000000LL;
        if ( *(_DWORD *)(v10 + 96) )
          sub_1F2BB90((__int64)&v74, v10 + 88);
        v77 = *(_BYTE *)(v10 + 456);
        v78 = *(_DWORD *)(v10 + 460);
        v79 = *(_QWORD *)(v10 + 464);
        v64 = &unk_49ECF98;
        if ( v62 != v63 )
          j_j___libc_free_0(v62, v63[0] + 1LL);
        if ( (_QWORD *)v60[0] != v61 )
          j_j___libc_free_0(v60[0], v61[0] + 1LL);
        v11 = v81;
        v80[0] = &unk_49ECF68;
        v12 = &v81[11 * v82];
        if ( v81 != v12 )
        {
          do
          {
            v12 -= 11;
            v13 = (_QWORD *)v12[4];
            if ( v13 != v12 + 6 )
              j_j___libc_free_0(v13, v12[6] + 1LL);
            if ( (_QWORD *)*v12 != v12 + 2 )
              j_j___libc_free_0(*v12, v12[2] + 1LL);
          }
          while ( v11 != v12 );
          v12 = v81;
        }
        if ( v12 != (_QWORD *)v83 )
          _libc_free((unsigned __int64)v12);
        sub_143AA50(v58, (__int64)&v64);
        v14 = v74;
        v64 = &unk_49ECF68;
        v15 = &v74[88 * (unsigned int)v75];
        if ( v74 != (_BYTE *)v15 )
        {
          do
          {
            v15 -= 11;
            v16 = (_QWORD *)v15[4];
            if ( v16 != v15 + 6 )
              j_j___libc_free_0(v16, v15[6] + 1LL);
            if ( (_QWORD *)*v15 != v15 + 2 )
              j_j___libc_free_0(*v15, v15[2] + 1LL);
          }
          while ( v14 != v15 );
          v15 = v74;
        }
        if ( v15 != (_QWORD *)v76 )
          _libc_free((unsigned __int64)v15);
      }
      v56 = v7;
    }
    else
    {
      v56 = sub_1560180(*(_QWORD *)(a1 + 232) + 112LL, 51);
      if ( !v56 )
      {
        v7 = *(unsigned __int8 *)(a1 + 464);
        if ( !(_BYTE)v7 && !(unsigned __int8)sub_1560180(*(_QWORD *)(a1 + 232) + 112LL, 49) )
          goto LABEL_91;
      }
    }
    v17 = *(_QWORD *)(a1 + 232);
    v49 = v17 + 72;
    v54 = *(_QWORD *)(v17 + 80);
    if ( v17 + 72 == v54 )
    {
LABEL_91:
      v34 = v59;
      if ( v59 )
      {
        sub_1368A00(v59);
        j_j___libc_free_0(v34, 8);
      }
      return v7;
    }
    v51 = v7;
    v18 = a1;
    while ( 1 )
    {
      if ( !v54 )
        BUG();
      v19 = *(_QWORD *)(v54 + 24);
      if ( v54 + 16 != v19 )
        break;
LABEL_89:
      v54 = *(_QWORD *)(v54 + 8);
      if ( v49 == v54 )
      {
        v7 = v51;
        goto LABEL_91;
      }
    }
    v20 = v54 + 16;
    while ( 1 )
    {
      if ( !v19 )
        BUG();
      if ( *(_BYTE *)(v19 - 8) != 53 )
        goto LABEL_54;
      v24 = sub_15F8BF0(v19 - 24);
      if ( v24 )
        break;
      v25 = *(_QWORD *)(v19 + 32);
      v57 = 0;
      v53 = sub_1F2B030(v18, v25, &v57, v56, 0);
      if ( v53 )
      {
        v64 = (void *)(v19 - 24);
        v65 = (v57 == 0) + 1;
        sub_1F2C360((__int64)v80, v18 + 256, (__int64 *)&v64, &v65);
        v35 = sub_15E0530((__int64)v58[0]);
        if ( sub_1602790(v35)
          || (v43 = sub_15E0530((__int64)v58[0]),
              v44 = sub_16033E0(v43),
              (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v44 + 48LL))(v44)) )
        {
          sub_15CA3B0((__int64)v80, (__int64)"stack-protector", (__int64)"StackProtectorBuffer", 20, v19 - 24);
          sub_15CAB20((__int64)v80, "Stack protection applied to function ", 0x25u);
          sub_15C9340((__int64)v60, "Function", 8u, *(_QWORD *)(v18 + 232));
          v36 = sub_17C2270((__int64)v80, (__int64)v60);
          sub_15CAB20(v36, " due to a stack allocated buffer or struct containing a buffer", 0x3Eu);
          v65 = *(_DWORD *)(v36 + 8);
          v66 = *(_BYTE *)(v36 + 12);
          v67 = *(_QWORD *)(v36 + 16);
          v68 = _mm_loadu_si128((const __m128i *)(v36 + 24));
          v69 = *(_QWORD *)(v36 + 40);
          v64 = &unk_49ECF68;
          v70 = *(_QWORD *)(v36 + 48);
          v71 = _mm_loadu_si128((const __m128i *)(v36 + 56));
          v73 = *(_BYTE *)(v36 + 80);
          if ( v73 )
            v72 = *(_QWORD *)(v36 + 72);
          v74 = v76;
          v75 = 0x400000000LL;
          if ( *(_DWORD *)(v36 + 96) )
            sub_1F2BB90((__int64)&v74, v36 + 88);
          v77 = *(_BYTE *)(v36 + 456);
          v78 = *(_DWORD *)(v36 + 460);
          v79 = *(_QWORD *)(v36 + 464);
          v64 = &unk_49ECF98;
          if ( v62 != v63 )
            j_j___libc_free_0(v62, v63[0] + 1LL);
          if ( (_QWORD *)v60[0] != v61 )
            j_j___libc_free_0(v60[0], v61[0] + 1LL);
          v37 = v81;
          v80[0] = &unk_49ECF68;
          v38 = &v81[11 * v82];
          if ( v81 != v38 )
          {
            do
            {
              v38 -= 11;
              v39 = (_QWORD *)v38[4];
              if ( v39 != v38 + 6 )
                j_j___libc_free_0(v39, v38[6] + 1LL);
              if ( (_QWORD *)*v38 != v38 + 2 )
                j_j___libc_free_0(*v38, v38[2] + 1LL);
            }
            while ( v37 != v38 );
            v38 = v81;
          }
          if ( v38 != (_QWORD *)v83 )
            _libc_free((unsigned __int64)v38);
          sub_143AA50(v58, (__int64)&v64);
          v40 = v74;
          v64 = &unk_49ECF68;
          v41 = &v74[88 * (unsigned int)v75];
          if ( v74 != (_BYTE *)v41 )
          {
            do
            {
              v41 -= 11;
              v42 = (_QWORD *)v41[4];
              if ( v42 != v41 + 6 )
                j_j___libc_free_0(v42, v41[6] + 1LL);
              if ( (_QWORD *)*v41 != v41 + 2 )
                j_j___libc_free_0(*v41, v41[2] + 1LL);
            }
            while ( v40 != v41 );
            v41 = v74;
          }
          if ( v41 != (_QWORD *)v76 )
            _libc_free((unsigned __int64)v41);
        }
        goto LABEL_88;
      }
      if ( v56 && (v53 = sub_1F2B3D0(v18, v19 - 24)) != 0 )
      {
        v64 = (void *)(v19 - 24);
        v65 = 3;
        sub_1F2C360((__int64)v80, v18 + 256, (__int64 *)&v64, &v65);
        v26 = sub_15E0530((__int64)v58[0]);
        if ( sub_1602790(v26)
          || (v47 = sub_15E0530((__int64)v58[0]),
              v48 = sub_16033E0(v47),
              (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v48 + 48LL))(v48)) )
        {
          sub_15CA3B0((__int64)v80, (__int64)"stack-protector", (__int64)"StackProtectorAddressTaken", 26, v19 - 24);
          sub_15CAB20((__int64)v80, "Stack protection applied to function ", 0x25u);
          sub_15C9340((__int64)v60, "Function", 8u, *(_QWORD *)(v18 + 232));
          v27 = sub_17C2270((__int64)v80, (__int64)v60);
          sub_15CAB20(v27, " due to the address of a local variable being taken", 0x33u);
          v65 = *(_DWORD *)(v27 + 8);
          v66 = *(_BYTE *)(v27 + 12);
          v67 = *(_QWORD *)(v27 + 16);
          v68 = _mm_loadu_si128((const __m128i *)(v27 + 24));
          v69 = *(_QWORD *)(v27 + 40);
          v64 = &unk_49ECF68;
          v70 = *(_QWORD *)(v27 + 48);
          v71 = _mm_loadu_si128((const __m128i *)(v27 + 56));
          v73 = *(_BYTE *)(v27 + 80);
          if ( v73 )
            v72 = *(_QWORD *)(v27 + 72);
          v74 = v76;
          v75 = 0x400000000LL;
          if ( *(_DWORD *)(v27 + 96) )
            sub_1F2BB90((__int64)&v74, v27 + 88);
          v77 = *(_BYTE *)(v27 + 456);
          v78 = *(_DWORD *)(v27 + 460);
          v79 = *(_QWORD *)(v27 + 464);
          v64 = &unk_49ECF98;
          if ( v62 != v63 )
            j_j___libc_free_0(v62, v63[0] + 1LL);
          if ( (_QWORD *)v60[0] != v61 )
            j_j___libc_free_0(v60[0], v61[0] + 1LL);
          v28 = v81;
          v80[0] = &unk_49ECF68;
          v29 = &v81[11 * v82];
          if ( v81 != v29 )
          {
            do
            {
              v29 -= 11;
              v30 = (_QWORD *)v29[4];
              if ( v30 != v29 + 6 )
                j_j___libc_free_0(v30, v29[6] + 1LL);
              if ( (_QWORD *)*v29 != v29 + 2 )
                j_j___libc_free_0(*v29, v29[2] + 1LL);
            }
            while ( v28 != v29 );
            v29 = v81;
          }
          if ( v29 != (_QWORD *)v83 )
            _libc_free((unsigned __int64)v29);
          sub_143AA50(v58, (__int64)&v64);
          v31 = v74;
          v64 = &unk_49ECF68;
          v32 = &v74[88 * (unsigned int)v75];
          if ( v74 != (_BYTE *)v32 )
          {
            do
            {
              v32 -= 11;
              v33 = (_QWORD *)v32[4];
              if ( v33 != v32 + 6 )
                j_j___libc_free_0(v33, v32[6] + 1LL);
              if ( (_QWORD *)*v32 != v32 + 2 )
                j_j___libc_free_0(*v32, v32[2] + 1LL);
            }
            while ( v31 != v32 );
            v32 = v74;
          }
          if ( v32 != (_QWORD *)v76 )
            _libc_free((unsigned __int64)v32);
        }
LABEL_88:
        v19 = *(_QWORD *)(v19 + 8);
        v51 = v53;
        if ( v20 == v19 )
          goto LABEL_89;
      }
      else
      {
LABEL_54:
        v19 = *(_QWORD *)(v19 + 8);
        if ( v20 == v19 )
          goto LABEL_89;
      }
    }
    v21 = *(_QWORD *)(v19 - 48);
    if ( *(_BYTE *)(v21 + 16) == 13 )
    {
      v22 = *(unsigned int *)(v18 + 288);
      if ( *(_DWORD *)(v21 + 32) > 0x40u )
      {
        v50 = *(_QWORD *)(v19 - 48);
        v52 = *(_DWORD *)(v21 + 32);
        if ( v52 - (unsigned int)sub_16A57B0(v21 + 24) > 0x40 )
          goto LABEL_53;
        v23 = **(_QWORD **)(v50 + 24);
        if ( v22 < v23 )
          goto LABEL_53;
      }
      else
      {
        v23 = *(_QWORD *)(v21 + 24);
        if ( v22 < v23 )
          goto LABEL_53;
      }
      if ( v22 > v23 )
      {
        if ( v56 )
        {
          v64 = (void *)(v19 - 24);
          v65 = 2;
          sub_1F2C360((__int64)v80, v18 + 256, (__int64 *)&v64, &v65);
          sub_1F2BE20((__int64 *)v58, v19 - 24, v18);
          v51 = v56;
        }
        goto LABEL_54;
      }
    }
LABEL_53:
    v64 = (void *)(v19 - 24);
    v65 = 1;
    sub_1F2C360((__int64)v80, v18 + 256, (__int64 *)&v64, &v65);
    sub_1F2BE20((__int64 *)v58, v19 - 24, v18);
    v51 = v24;
    goto LABEL_54;
  }
  return v7;
}
