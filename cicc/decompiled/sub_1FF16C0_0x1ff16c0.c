// Function: sub_1FF16C0
// Address: 0x1ff16c0
//
__int64 *__fastcall sub_1FF16C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r13
  __int64 v6; // r12
  __int64 v7; // rax
  __int64 v8; // rsi
  __m128i v9; // xmm1
  __m128i v10; // xmm2
  __int64 v11; // r9
  __int64 v12; // r15
  __int64 v13; // rbx
  int v14; // eax
  __int64 v15; // rbx
  __int64 v17; // r12
  __int64 v18; // r14
  __int64 v19; // rdi
  int v20; // r9d
  __int64 v21; // rdx
  __int64 v22; // rcx
  int v23; // r9d
  __int64 v24; // rax
  __int64 v25; // r9
  unsigned int v26; // ebx
  __int64 v27; // rdx
  __int64 v28; // rax
  __int64 v29; // r11
  __int64 v30; // rax
  __int64 v31; // rax
  unsigned int v32; // edx
  __int64 v33; // r15
  _QWORD *v34; // rdi
  __int64 v35; // rdx
  _QWORD *v36; // rax
  _QWORD *v37; // rdi
  unsigned int v38; // edx
  unsigned int v39; // edx
  unsigned __int64 v40; // rax
  unsigned __int8 *v41; // rdx
  char v42; // al
  bool v43; // al
  _QWORD *v44; // r11
  __int64 v45; // r10
  int v46; // r13d
  __int64 v47; // rsi
  __int64 v48; // r9
  __int64 *v49; // r9
  __int64 v50; // rax
  const __m128i *v51; // r13
  __int64 v52; // rcx
  const __m128i *v53; // rax
  unsigned __int64 v54; // rdx
  __m128i *v55; // rcx
  int v56; // esi
  __int64 *v57; // rdi
  __int64 *v58; // r12
  __int64 v60; // rax
  char v61; // cl
  __int64 v62; // rdx
  __int64 v63; // rax
  __int64 v64; // rsi
  __int64 v65; // r8
  __int128 v66; // [rsp-60h] [rbp-390h]
  __int64 v67; // [rsp+8h] [rbp-328h]
  __int64 v68; // [rsp+20h] [rbp-310h]
  __int64 v69; // [rsp+28h] [rbp-308h]
  const __m128i *v70; // [rsp+28h] [rbp-308h]
  _QWORD *v71; // [rsp+30h] [rbp-300h]
  unsigned __int8 *v72; // [rsp+30h] [rbp-300h]
  int v73; // [rsp+50h] [rbp-2E0h]
  __int64 v74; // [rsp+58h] [rbp-2D8h]
  __int64 v75; // [rsp+60h] [rbp-2D0h]
  __int64 v76; // [rsp+60h] [rbp-2D0h]
  int v77; // [rsp+60h] [rbp-2D0h]
  __int64 v78; // [rsp+60h] [rbp-2D0h]
  __int64 v79; // [rsp+60h] [rbp-2D0h]
  __int64 v80; // [rsp+68h] [rbp-2C8h]
  __int64 v81; // [rsp+68h] [rbp-2C8h]
  __int64 v82; // [rsp+90h] [rbp-2A0h] BYREF
  int v83; // [rsp+98h] [rbp-298h]
  unsigned int v84; // [rsp+A0h] [rbp-290h] BYREF
  __int64 v85; // [rsp+A8h] [rbp-288h]
  __int64 v86; // [rsp+B0h] [rbp-280h]
  __int64 v87; // [rsp+B8h] [rbp-278h]
  __int64 v88; // [rsp+C0h] [rbp-270h]
  __int64 *v89; // [rsp+D0h] [rbp-260h] BYREF
  __int64 v90; // [rsp+D8h] [rbp-258h]
  _QWORD v91[12]; // [rsp+E0h] [rbp-250h] BYREF
  unsigned __int64 v92[2]; // [rsp+140h] [rbp-1F0h] BYREF
  _QWORD v93[16]; // [rsp+150h] [rbp-1E0h] BYREF
  __int64 v94; // [rsp+1D0h] [rbp-160h] BYREF
  _BYTE *v95; // [rsp+1D8h] [rbp-158h]
  _BYTE *v96; // [rsp+1E0h] [rbp-150h]
  __int64 v97; // [rsp+1E8h] [rbp-148h]
  int v98; // [rsp+1F0h] [rbp-140h]
  _BYTE v99[312]; // [rsp+1F8h] [rbp-138h] BYREF

  v5 = a2;
  v6 = a1;
  v7 = *(_QWORD *)(a2 + 32);
  v8 = *(_QWORD *)(a2 + 72);
  v9 = _mm_loadu_si128((const __m128i *)v7);
  v10 = _mm_loadu_si128((const __m128i *)(v7 + 40));
  v11 = *(_QWORD *)v7;
  v12 = *(unsigned int *)(v7 + 8);
  v82 = v8;
  v13 = *(_QWORD *)(v7 + 40);
  if ( v8 )
  {
    v75 = v11;
    sub_1623A60((__int64)&v82, v8, 2);
    v11 = v75;
  }
  v14 = *(_DWORD *)(v5 + 64);
  v93[0] = v13;
  v94 = 0;
  v15 = *(_QWORD *)(v11 + 48);
  v83 = v14;
  v95 = v99;
  v96 = v99;
  v92[0] = (unsigned __int64)v93;
  v92[1] = 0x1000000001LL;
  v97 = 32;
  v98 = 0;
  if ( v15 )
  {
    v69 = v5;
    v17 = v11;
    do
    {
      v18 = *(_QWORD *)(v15 + 16);
      if ( *(_WORD *)(v18 + 24) == 186 && (*(_WORD *)(v18 + 26) & 0x380) == 0 && (*(_BYTE *)(v18 + 27) & 4) == 0 )
      {
        v19 = *(_QWORD *)(v18 + 32);
        if ( v17 == *(_QWORD *)(v19 + 40) && (_DWORD)v12 == *(_DWORD *)(v19 + 48) )
        {
          v68 &= 0xFFFFFFFF00000000LL;
          if ( sub_1D18E30((_QWORD *)v19, *(_QWORD *)(a1 + 16) + 88LL, v68, 2) )
          {
            if ( !(unsigned __int8)sub_1D15B50(v18, (__int64)&v94, (__int64)v92, 0, 0, v20)
              && !(unsigned __int8)sub_1D19270(v18, v69, v21, v22, a5, v23) )
            {
              v24 = *(_QWORD *)(v18 + 32);
              v25 = v17;
              v26 = 0;
              v6 = a1;
              v5 = v69;
              v27 = *(unsigned int *)(v24 + 88);
              v76 = *(_QWORD *)(v24 + 80);
              v28 = *(_QWORD *)(v25 + 40) + 16LL * (unsigned int)v12;
              v29 = v27;
              LOBYTE(v27) = *(_BYTE *)v28;
              v30 = *(_QWORD *)(v28 + 8);
              v80 = v29;
              LOBYTE(v84) = v27;
              v85 = v30;
              v31 = sub_20BD400(
                      *(_QWORD *)(v6 + 8),
                      *(_QWORD *)(v6 + 16),
                      v76,
                      v29,
                      v84,
                      v30,
                      v10.m128i_i64[0],
                      v10.m128i_i64[1]);
              goto LABEL_17;
            }
          }
        }
      }
      v15 = *(_QWORD *)(v15 + 32);
    }
    while ( v15 );
    v11 = v17;
    v6 = a1;
    v5 = v69;
  }
  v33 = *(_QWORD *)(v11 + 40) + 16 * v12;
  v34 = *(_QWORD **)(v6 + 16);
  v35 = *(_QWORD *)(v33 + 8);
  LOBYTE(v84) = *(_BYTE *)v33;
  v85 = v35;
  v36 = sub_1D29C20(v34, v84, v35, 1, a5, v11);
  v87 = 0;
  v77 = (int)v36;
  v37 = *(_QWORD **)(v6 + 16);
  v88 = 0;
  v80 = v38;
  v89 = 0;
  v90 = 0;
  v91[0] = 0;
  v86 = 0;
  v18 = sub_1D2BF40(
          v37,
          (__int64)(v37 + 11),
          0,
          (__int64)&v82,
          v9.m128i_i64[0],
          v9.m128i_i64[1],
          (__int64)v36,
          v38,
          0,
          0,
          0,
          0,
          (__int64)&v89);
  v26 = v39;
  v74 = v39;
  v31 = sub_20BD400(*(_QWORD *)(v6 + 8), *(_QWORD *)(v6 + 16), v77, v80, v84, v85, v10.m128i_i64[0], v10.m128i_i64[1]);
LABEL_17:
  v78 = v31;
  v40 = v32 | v80 & 0xFFFFFFFF00000000LL;
  v41 = *(unsigned __int8 **)(v5 + 40);
  v81 = v40;
  v42 = *v41;
  v90 = *((_QWORD *)v41 + 1);
  LOBYTE(v89) = v42;
  if ( v42 )
  {
    v43 = (unsigned __int8)(v42 - 14) <= 0x5Fu;
  }
  else
  {
    v72 = v41;
    v43 = sub_1F58D20((__int64)&v89);
    v41 = v72;
  }
  v89 = 0;
  v44 = *(_QWORD **)(v6 + 16);
  v90 = 0;
  v91[0] = 0;
  if ( v43 )
  {
    v45 = *((_QWORD *)v41 + 1);
    v46 = v26;
    v47 = *v41;
    v87 = 0;
    v88 = 0;
    v86 = 0;
    v48 = sub_1D2B730(
            v44,
            v47,
            v45,
            (__int64)&v82,
            v18,
            v26 | v74 & 0xFFFFFFFF00000000LL,
            v78,
            v81,
            0,
            0,
            0,
            0,
            (__int64)&v89,
            0);
  }
  else
  {
    if ( (_BYTE)v84 )
    {
      switch ( (char)v84 )
      {
        case 14:
        case 15:
        case 16:
        case 17:
        case 18:
        case 19:
        case 20:
        case 21:
        case 22:
        case 23:
        case 56:
        case 57:
        case 58:
        case 59:
        case 60:
        case 61:
          v61 = 2;
          v63 = 0;
          break;
        case 24:
        case 25:
        case 26:
        case 27:
        case 28:
        case 29:
        case 30:
        case 31:
        case 32:
        case 62:
        case 63:
        case 64:
        case 65:
        case 66:
        case 67:
          v61 = 3;
          v63 = 0;
          break;
        case 33:
        case 34:
        case 35:
        case 36:
        case 37:
        case 38:
        case 39:
        case 40:
        case 68:
        case 69:
        case 70:
        case 71:
        case 72:
        case 73:
          v61 = 4;
          v63 = 0;
          break;
        case 41:
        case 42:
        case 43:
        case 44:
        case 45:
        case 46:
        case 47:
        case 48:
        case 74:
        case 75:
        case 76:
        case 77:
        case 78:
        case 79:
          v61 = 5;
          v63 = 0;
          break;
        case 49:
        case 50:
        case 51:
        case 52:
        case 53:
        case 54:
        case 80:
        case 81:
        case 82:
        case 83:
        case 84:
        case 85:
          v61 = 6;
          v63 = 0;
          break;
        case 55:
          v61 = 7;
          v63 = 0;
          break;
        case 86:
        case 87:
        case 88:
        case 98:
        case 99:
        case 100:
          v61 = 8;
          v63 = 0;
          break;
        case 89:
        case 90:
        case 91:
        case 92:
        case 93:
        case 101:
        case 102:
        case 103:
        case 104:
        case 105:
          v61 = 9;
          v63 = 0;
          break;
        case 94:
        case 95:
        case 96:
        case 97:
        case 106:
        case 107:
        case 108:
        case 109:
          v61 = 10;
          v63 = 0;
          break;
      }
    }
    else
    {
      v71 = v44;
      LOBYTE(v60) = sub_1F596B0((__int64)&v84);
      v44 = v71;
      v67 = v60;
      v61 = v60;
      v63 = v62;
      v41 = *(unsigned __int8 **)(v5 + 40);
    }
    v64 = v67;
    v46 = v26;
    v65 = *((_QWORD *)v41 + 1);
    v87 = 0;
    LOBYTE(v64) = v61;
    v88 = 0;
    v86 = 0;
    *((_QWORD *)&v66 + 1) = v26 | v74 & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v66 = v18;
    v48 = sub_1D2B810(v44, 1u, (__int64)&v82, *v41, v65, 0, v66, v78, v81, 0, 0, v64, v63, 0, (__int64)&v89);
  }
  v79 = v48;
  sub_1D44C70(*(_QWORD *)(v6 + 16), v18, v46, v48, 1u);
  v49 = (__int64 *)v79;
  v50 = *(unsigned int *)(v79 + 56);
  v51 = *(const __m128i **)(v79 + 32);
  v90 = 0x600000000LL;
  v89 = v91;
  v52 = 40 * v50;
  v53 = (const __m128i *)((char *)v51 + 40 * v50);
  v54 = 0xCCCCCCCCCCCCCCCDLL * (v52 >> 3);
  if ( (unsigned __int64)v52 > 0xF0 )
  {
    v70 = v53;
    v73 = -858993459 * (v52 >> 3);
    sub_16CD150((__int64)&v89, v91, v54, 16, (int)v91, v79);
    v56 = v90;
    v57 = v89;
    LODWORD(v54) = v73;
    v49 = (__int64 *)v79;
    v53 = v70;
    v55 = (__m128i *)&v89[2 * (unsigned int)v90];
  }
  else
  {
    v55 = (__m128i *)v91;
    v56 = 0;
    v57 = v91;
  }
  if ( v51 != v53 )
  {
    do
    {
      if ( v55 )
        *v55 = _mm_loadu_si128(v51);
      v51 = (const __m128i *)((char *)v51 + 40);
      ++v55;
    }
    while ( v53 != v51 );
    v57 = v89;
    v56 = v90;
  }
  LODWORD(v90) = v56 + v54;
  *v57 = v18;
  *((_DWORD *)v57 + 2) = v26;
  v58 = sub_1D2E160(*(_QWORD **)(v6 + 16), v49, (__int64)v89, (unsigned int)v90);
  if ( v89 != v91 )
    _libc_free((unsigned __int64)v89);
  if ( (_QWORD *)v92[0] != v93 )
    _libc_free(v92[0]);
  if ( v96 != v95 )
    _libc_free((unsigned __int64)v96);
  if ( v82 )
    sub_161E7C0((__int64)&v82, v82);
  return v58;
}
