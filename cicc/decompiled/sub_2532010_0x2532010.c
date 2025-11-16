// Function: sub_2532010
// Address: 0x2532010
//
__int64 __fastcall sub_2532010(__int64 a1, __int64 a2, __int64 a3, char a4, char a5)
{
  _QWORD *v7; // r14
  _QWORD *v8; // r13
  unsigned __int64 v9; // rsi
  _QWORD *v10; // rax
  _QWORD *v11; // rdi
  __int64 v12; // rcx
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // rax
  _QWORD *v16; // rdi
  __int64 v17; // rdx
  __m128i v18; // xmm1
  __int64 *v19; // r12
  __int64 v20; // r14
  unsigned int v21; // eax
  _BYTE *v22; // r15
  _QWORD *v23; // rax
  _QWORD *v24; // rdx
  __int64 *v25; // r12
  __int64 v26; // r15
  __int64 v27; // rdx
  unsigned __int8 *v28; // rdi
  int v29; // eax
  unsigned __int64 v30; // rax
  __int64 v31; // rax
  int v32; // ecx
  __int64 v33; // r8
  int v34; // edi
  unsigned int v35; // ecx
  __int64 v36; // r9
  unsigned int v37; // eax
  _QWORD *v38; // rbx
  _QWORD *v39; // r12
  unsigned __int64 v40; // r13
  __m128i v42; // xmm2
  __m128i v43; // xmm0
  __int64 v44; // r12
  _BYTE *v45; // r15
  int v46; // r10d
  __int64 i; // rdx
  unsigned __int8 *v48; // rdi
  int v49; // eax
  unsigned __int64 v50; // rax
  __int64 v51; // rsi
  __int64 v52; // rax
  __int64 v53; // [rsp+8h] [rbp-1308h]
  __int64 v54; // [rsp+10h] [rbp-1300h]
  __int64 *v56; // [rsp+18h] [rbp-12F8h]
  __int64 v57; // [rsp+20h] [rbp-12F0h] BYREF
  _QWORD *v58; // [rsp+28h] [rbp-12E8h]
  __int64 v59; // [rsp+30h] [rbp-12E0h]
  unsigned int v60; // [rsp+38h] [rbp-12D8h]
  __int64 v61; // [rsp+40h] [rbp-12D0h]
  _BYTE v62[16]; // [rsp+48h] [rbp-12C8h] BYREF
  void (__fastcall *v63)(unsigned __int64 *, unsigned __int64 *, __int64); // [rsp+58h] [rbp-12B8h]
  __int64 v64; // [rsp+60h] [rbp-12B0h]
  __m128i v65; // [rsp+68h] [rbp-12A8h] BYREF
  __int64 (__fastcall *v66)(_QWORD *, _QWORD *, int); // [rsp+78h] [rbp-1298h]
  __int64 (__fastcall *v67)(__int64 *, __int64, __int64, __int64, __int64); // [rsp+80h] [rbp-1290h]
  __int64 v68; // [rsp+88h] [rbp-1288h]
  __int64 v69; // [rsp+90h] [rbp-1280h]
  __int64 v70; // [rsp+98h] [rbp-1278h]
  __m128i v71; // [rsp+A0h] [rbp-1270h] BYREF
  __int64 v72; // [rsp+B0h] [rbp-1260h]
  _BYTE v73[16]; // [rsp+B8h] [rbp-1258h] BYREF
  void (__fastcall *v74)(_BYTE *, _BYTE *, __int64, __int64); // [rsp+C8h] [rbp-1248h]
  __int64 v75; // [rsp+D0h] [rbp-1240h]
  __m128i v76; // [rsp+E0h] [rbp-1230h] BYREF
  void (__fastcall *v77)(unsigned __int64 *, unsigned __int64 *, __int64); // [rsp+F8h] [rbp-1218h]
  __int64 v78; // [rsp+100h] [rbp-1210h]
  _BYTE v79[16]; // [rsp+108h] [rbp-1208h] BYREF
  void (__fastcall *v80)(_BYTE *, _BYTE *, __int64); // [rsp+118h] [rbp-11F8h]
  __int64 (__fastcall *v81)(__int64 *, __int64, __int64, __int64, __int64); // [rsp+120h] [rbp-11F0h]
  __int64 v82; // [rsp+128h] [rbp-11E8h]
  __int64 v83; // [rsp+130h] [rbp-11E0h]
  __int64 v84; // [rsp+138h] [rbp-11D8h]
  __m128i v85; // [rsp+140h] [rbp-11D0h]
  __int64 v86; // [rsp+150h] [rbp-11C0h]
  _BYTE v87[16]; // [rsp+158h] [rbp-11B8h] BYREF
  void (__fastcall *v88)(_BYTE *, _BYTE *, __int64); // [rsp+168h] [rbp-11A8h]
  __int64 v89; // [rsp+170h] [rbp-11A0h]
  __m128i v90; // [rsp+180h] [rbp-1190h] BYREF
  __int64 (__fastcall *v91)(_QWORD *, _QWORD *, int); // [rsp+190h] [rbp-1180h]
  __int64 (__fastcall *v92)(__int64 *, __int64, __int64, __int64, __int64); // [rsp+198h] [rbp-1178h]
  __int64 v93; // [rsp+250h] [rbp-10C0h]
  _BYTE v94[16]; // [rsp+12C0h] [rbp-50h] BYREF
  __int64 v95; // [rsp+12D0h] [rbp-40h]
  unsigned __int8 (__fastcall *v96)(_BYTE *, _BYTE *); // [rsp+12D8h] [rbp-38h]

  v61 = 0x101010000LL;
  v68 = a3;
  LOBYTE(v61) = a5;
  BYTE1(v61) = a4;
  v63 = 0;
  v66 = 0;
  v69 = 0;
  BYTE4(v70) = 0;
  v71.m128i_i64[0] = 0;
  v72 = 0;
  v74 = 0;
  v57 = 0;
  v58 = 0;
  v59 = 0;
  v60 = 0;
  v7 = sub_C52410();
  v8 = v7 + 1;
  v9 = sub_C959E0();
  v10 = (_QWORD *)v7[2];
  if ( v10 )
  {
    v11 = v7 + 1;
    do
    {
      while ( 1 )
      {
        v12 = v10[2];
        v13 = v10[3];
        if ( v9 <= v10[4] )
          break;
        v10 = (_QWORD *)v10[3];
        if ( !v13 )
          goto LABEL_6;
      }
      v11 = v10;
      v10 = (_QWORD *)v10[2];
    }
    while ( v12 );
LABEL_6:
    if ( v8 != v11 && v9 >= v11[4] )
      v8 = v11;
  }
  if ( v8 != (_QWORD *)((char *)sub_C52410() + 8) )
  {
    v15 = v8[7];
    v16 = v8 + 6;
    if ( v15 )
    {
      do
      {
        while ( 1 )
        {
          v14 = *(_QWORD *)(v15 + 16);
          v17 = *(_QWORD *)(v15 + 24);
          if ( *(_DWORD *)(v15 + 32) >= dword_4FEF068 )
            break;
          v15 = *(_QWORD *)(v15 + 24);
          if ( !v17 )
            goto LABEL_15;
        }
        v16 = (_QWORD *)v15;
        v15 = *(_QWORD *)(v15 + 16);
      }
      while ( v14 );
LABEL_15:
      if ( v8 + 6 != v16 && dword_4FEF068 >= *((_DWORD *)v16 + 8) && *((_DWORD *)v16 + 9) )
      {
        v42 = _mm_loadu_si128(&v65);
        v90.m128i_i64[0] = (__int64)&v57;
        v43 = _mm_loadu_si128(&v90);
        v90 = v42;
        v91 = v66;
        v66 = sub_25060E0;
        v65 = v43;
        v92 = v67;
        v67 = sub_2514460;
        sub_A17130((__int64)&v90);
      }
    }
  }
  v77 = 0;
  v76.m128i_i32[0] = v61;
  v76.m128i_i16[2] = WORD2(v61);
  if ( v63 )
  {
    ((void (__fastcall *)(unsigned __int64 *, _BYTE *, __int64, __int64))v63)(&v76.m128i_u64[1], v62, 2, v14);
    v78 = v64;
    v77 = v63;
  }
  v80 = 0;
  if ( v66 )
  {
    ((void (__fastcall *)(_BYTE *, __m128i *, __int64, __int64))v66)(v79, &v65, 2, v14);
    v81 = v67;
    v80 = (void (__fastcall *)(_BYTE *, _BYTE *, __int64))v66;
  }
  v18 = _mm_loadu_si128(&v71);
  v88 = 0;
  v82 = v68;
  v85 = v18;
  v83 = v69;
  v84 = v70;
  v86 = v72;
  if ( v74 )
  {
    v74(v87, v73, 2, v14);
    v89 = v75;
    v88 = (void (__fastcall *)(_BYTE *, _BYTE *, __int64))v74;
  }
  sub_250EFA0((__int64)&v90, a2, a1, &v76);
  if ( v88 )
    v88(v87, v87, 3);
  if ( v80 )
    v80(v79, v79, 3);
  if ( v77 )
    v77(&v76.m128i_u64[1], &v76.m128i_u64[1], 3);
  if ( !(_BYTE)qword_4FEED68 )
  {
LABEL_38:
    if ( !(_BYTE)qword_4FEEC88 )
    {
LABEL_39:
      v25 = *(__int64 **)(a2 + 32);
      v20 = 0x8000000000041LL;
      v56 = &v25[*(unsigned int *)(a2 + 40)];
      if ( v25 == v56 )
        goto LABEL_55;
      while ( 1 )
      {
LABEL_43:
        v26 = *v25;
        if ( !sub_B2FC80(*v25) )
          sub_B2FC00((_BYTE *)v26);
        if ( (*(_BYTE *)(v26 + 32) & 0xFu) - 7 <= 1 )
        {
          v27 = *(_QWORD *)(v26 + 16);
          if ( !v27 )
            goto LABEL_42;
          while ( 1 )
          {
            v28 = *(unsigned __int8 **)(v27 + 24);
            v29 = *v28;
            if ( (unsigned __int8)v29 <= 0x1Cu )
              break;
            v30 = (unsigned int)(v29 - 34);
            if ( (unsigned __int8)v30 > 0x33u )
              break;
            if ( !_bittest64(&v20, v30) )
              break;
            if ( (unsigned __int8 *)v27 != v28 - 32 )
              break;
            v54 = v27;
            v31 = sub_B491C0((__int64)v28);
            v32 = *(_DWORD *)(a2 + 24);
            v33 = *(_QWORD *)(a2 + 8);
            if ( !v32 )
              break;
            v34 = v32 - 1;
            v35 = (v32 - 1) & (((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4));
            v36 = *(_QWORD *)(v33 + 8LL * v35);
            if ( v31 != v36 )
            {
              v46 = 1;
              while ( v36 != -4096 )
              {
                v35 = v34 & (v46 + v35);
                v36 = *(_QWORD *)(v33 + 8LL * v35);
                if ( v31 == v36 )
                  goto LABEL_53;
                ++v46;
              }
              break;
            }
LABEL_53:
            v27 = *(_QWORD *)(v54 + 8);
            if ( !v27 )
            {
              if ( v56 != ++v25 )
                goto LABEL_43;
              goto LABEL_55;
            }
          }
        }
        sub_252C210((__int64)&v90, v26);
LABEL_42:
        if ( v56 == ++v25 )
          goto LABEL_55;
      }
    }
    v21 = *(_DWORD *)(a2 + 40);
LABEL_75:
    v20 = 0;
    v44 = 8LL * v21;
    if ( !v21 )
      goto LABEL_55;
    do
    {
      v45 = *(_BYTE **)(*(_QWORD *)(a2 + 32) + v20);
      if ( !sub_B2FC80((__int64)v45) && (unsigned __int8)sub_B2FC00(v45) && (unsigned int)sub_BD3960((__int64)v45) )
      {
        switch ( v45[32] & 0xF )
        {
          case 0:
          case 1:
          case 3:
          case 5:
          case 6:
          case 7:
          case 8:
            v76.m128i_i64[0] = sub_25177C0((__int64)v45, 0);
            sub_2519280(a2, v76.m128i_i64);
            sub_29A24B0(a3, v45, v76.m128i_i64[0]);
            for ( i = *(_QWORD *)(v76.m128i_i64[0] + 16); i; i = *(_QWORD *)(i + 8) )
            {
              v48 = *(unsigned __int8 **)(i + 24);
              v49 = *v48;
              if ( (unsigned __int8)v49 > 0x1Cu )
              {
                v50 = (unsigned int)(v49 - 34);
                if ( (unsigned __int8)v50 <= 0x33u )
                {
                  v51 = 0x8000000000041LL;
                  if ( _bittest64(&v51, v50) )
                  {
                    v53 = i;
                    v52 = sub_B491C0((__int64)v48);
                    sub_29A27C0(a3, v52);
                    i = v53;
                  }
                }
              }
            }
            break;
          case 2:
          case 4:
          case 9:
          case 0xA:
            break;
          default:
            BUG();
        }
      }
      v20 += 8;
    }
    while ( v20 != v44 );
    goto LABEL_39;
  }
  v19 = *(__int64 **)(a2 + 32);
  v20 = (__int64)&v19[*(unsigned int *)(a2 + 40)];
  v21 = *(_DWORD *)(a2 + 40);
  if ( v19 != (__int64 *)v20 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v22 = (_BYTE *)*v19;
        if ( sub_B2FC80(*v19) || (unsigned __int8)sub_B2FC00(v22) )
          break;
        if ( (__int64 *)v20 == ++v19 )
          goto LABEL_38;
      }
      if ( *(_BYTE *)(v93 + 276) )
      {
        v23 = *(_QWORD **)(v93 + 256);
        v24 = &v23[*(unsigned int *)(v93 + 268)];
        if ( v23 != v24 )
        {
          while ( v22 != (_BYTE *)*v23 )
          {
            if ( v24 == ++v23 )
              goto LABEL_86;
          }
          goto LABEL_37;
        }
LABEL_86:
        if ( v95 && v96(v94, v22) )
          goto LABEL_37;
        ++v19;
        sub_250E310((__int64)v22);
        if ( (__int64 *)v20 == v19 )
          goto LABEL_38;
      }
      else
      {
        if ( !sub_C8CA60(v93 + 248, (__int64)v22) )
          goto LABEL_86;
LABEL_37:
        if ( (__int64 *)v20 == ++v19 )
          goto LABEL_38;
      }
    }
  }
  if ( (_BYTE)qword_4FEEC88 )
    goto LABEL_75;
LABEL_55:
  LOBYTE(v20) = (unsigned int)sub_2531E70((__int64)&v90) == 0;
  sub_250D880((__int64)&v90);
  v37 = v60;
  if ( v60 )
  {
    v38 = v58;
    v39 = &v58[2 * v60];
    do
    {
      if ( *v38 != -4096 && *v38 != -8192 )
      {
        v40 = v38[1];
        if ( v40 )
        {
          if ( !*(_BYTE *)(v40 + 28) )
            _libc_free(*(_QWORD *)(v40 + 8));
          j_j___libc_free_0(v40);
        }
      }
      v38 += 2;
    }
    while ( v39 != v38 );
    v37 = v60;
  }
  sub_C7D6A0((__int64)v58, 16LL * v37, 8);
  if ( v74 )
    ((void (__fastcall *)(_BYTE *, _BYTE *, __int64))v74)(v73, v73, 3);
  if ( v66 )
    v66(&v65, &v65, 3);
  if ( v63 )
    v63((unsigned __int64 *)v62, (unsigned __int64 *)v62, 3);
  return (unsigned int)v20;
}
