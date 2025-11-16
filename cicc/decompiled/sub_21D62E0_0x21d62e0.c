// Function: sub_21D62E0
// Address: 0x21d62e0
//
__int64 __fastcall sub_21D62E0(double a1, double a2, __m128i a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 v9; // rax
  __int64 v10; // rsi
  __m128i v11; // xmm0
  __int64 v12; // r15
  __int64 v13; // r14
  __int64 v14; // rax
  char v15; // dl
  __int64 v16; // rax
  __int64 v17; // r13
  __int64 v19; // rax
  __m128i v20; // xmm3
  __int64 v21; // rbx
  __int64 v22; // rax
  __int64 *v23; // rax
  __int64 v24; // rax
  __int128 v25; // rax
  __int128 v26; // rax
  int v27; // r8d
  __int64 v28; // r14
  unsigned __int64 v29; // r15
  unsigned int v30; // edx
  char v31; // cl
  const void **v32; // rdx
  unsigned int v33; // eax
  __m128i v34; // xmm2
  __int64 v35; // rbx
  __int128 v36; // rax
  __int64 *v37; // rax
  unsigned __int64 v38; // rdx
  unsigned __int64 v39; // r15
  __int64 v40; // r14
  __int128 v41; // rax
  __int128 v42; // rax
  int v43; // r8d
  __int64 *v44; // r14
  __int64 *v45; // rdx
  __int64 *v46; // r15
  __int64 v47; // rax
  __int64 **v48; // rax
  __int64 v49; // rbx
  const __m128i *v50; // r14
  __int64 v51; // rdx
  __int64 v52; // r8
  const __m128i *v53; // r14
  unsigned __int64 v54; // r15
  unsigned __int64 v55; // rdx
  __m128i *v56; // rcx
  unsigned int v57; // eax
  __int64 v58; // r15
  unsigned __int8 v59; // r14
  __int64 v60; // rax
  int v61; // edx
  int v62; // r9d
  __m128i v63; // xmm4
  __int64 v64; // [rsp+8h] [rbp-148h]
  unsigned __int16 v65; // [rsp+14h] [rbp-13Ch]
  __int64 v66; // [rsp+20h] [rbp-130h]
  __int64 v67; // [rsp+28h] [rbp-128h]
  __int64 v68; // [rsp+28h] [rbp-128h]
  __int64 v69; // [rsp+30h] [rbp-120h]
  unsigned int v70; // [rsp+30h] [rbp-120h]
  char v71; // [rsp+30h] [rbp-120h]
  __int64 v72; // [rsp+30h] [rbp-120h]
  unsigned int v73; // [rsp+38h] [rbp-118h]
  unsigned int v74; // [rsp+38h] [rbp-118h]
  __int64 v75; // [rsp+38h] [rbp-118h]
  __int64 v76; // [rsp+38h] [rbp-118h]
  __int64 *v77; // [rsp+40h] [rbp-110h]
  __int64 v78; // [rsp+40h] [rbp-110h]
  unsigned __int64 v79; // [rsp+48h] [rbp-108h]
  __int64 v80; // [rsp+48h] [rbp-108h]
  __int64 v81; // [rsp+60h] [rbp-F0h] BYREF
  int v82; // [rsp+68h] [rbp-E8h]
  char v83[8]; // [rsp+70h] [rbp-E0h] BYREF
  __int64 v84; // [rsp+78h] [rbp-D8h]
  unsigned int v85; // [rsp+80h] [rbp-D0h] BYREF
  const void **v86; // [rsp+88h] [rbp-C8h]
  _OWORD *v87; // [rsp+90h] [rbp-C0h] BYREF
  __int64 v88; // [rsp+98h] [rbp-B8h]
  _OWORD v89[11]; // [rsp+A0h] [rbp-B0h] BYREF

  v9 = *(_QWORD *)(a5 + 32);
  v10 = *(_QWORD *)(a5 + 72);
  v11 = _mm_loadu_si128((const __m128i *)(v9 + 40));
  v81 = v10;
  v12 = *(_QWORD *)(v9 + 40);
  v13 = *(unsigned int *)(v9 + 48);
  v79 = v11.m128i_u64[1];
  if ( v10 )
    sub_1623A60((__int64)&v81, v10, 2);
  v82 = *(_DWORD *)(a5 + 64);
  v14 = *(_QWORD *)(v12 + 40) + 16 * v13;
  v15 = *(_BYTE *)v14;
  v16 = *(_QWORD *)(v14 + 8);
  v83[0] = v15;
  v84 = v16;
  if ( !v15 || (unsigned __int8)(v15 - 14) > 0x5Fu )
    goto LABEL_5;
  switch ( v15 )
  {
    case 25:
    case 26:
    case 34:
    case 35:
    case 42:
    case 43:
    case 50:
    case 86:
    case 87:
    case 88:
    case 90:
    case 91:
    case 95:
      v69 = sub_1E0A0C0(*(_QWORD *)(a7 + 32));
      v73 = sub_1E34390(*(_QWORD *)(a5 + 104));
      v19 = sub_1F58E60((__int64)v83, *(_QWORD **)(a7 + 48));
      if ( (unsigned int)sub_15AAE50(v69, v19) > v73 )
        goto LABEL_5;
      if ( v83[0] )
      {
        switch ( v83[0] )
        {
          case 0xE:
          case 0xF:
          case 0x10:
          case 0x11:
          case 0x12:
          case 0x13:
          case 0x14:
          case 0x15:
          case 0x16:
          case 0x17:
          case 0x38:
          case 0x39:
          case 0x3A:
          case 0x3B:
          case 0x3C:
          case 0x3D:
            LOBYTE(v85) = 2;
            goto LABEL_14;
          case 0x18:
          case 0x19:
          case 0x1A:
          case 0x1B:
          case 0x1C:
          case 0x1D:
          case 0x1E:
          case 0x1F:
          case 0x20:
          case 0x3E:
          case 0x3F:
          case 0x40:
          case 0x41:
          case 0x42:
          case 0x43:
            LOBYTE(v85) = 3;
            goto LABEL_14;
          case 0x21:
          case 0x22:
          case 0x23:
          case 0x24:
          case 0x25:
          case 0x26:
          case 0x27:
          case 0x28:
          case 0x44:
          case 0x45:
          case 0x46:
          case 0x47:
          case 0x48:
          case 0x49:
            LOBYTE(v85) = 4;
            goto LABEL_14;
          case 0x29:
          case 0x2A:
          case 0x2B:
          case 0x2C:
          case 0x2D:
          case 0x2E:
          case 0x2F:
          case 0x30:
          case 0x4A:
          case 0x4B:
          case 0x4C:
          case 0x4D:
          case 0x4E:
          case 0x4F:
            LOBYTE(v85) = 5;
            goto LABEL_14;
          case 0x31:
          case 0x32:
          case 0x33:
          case 0x34:
          case 0x35:
          case 0x36:
          case 0x50:
          case 0x51:
          case 0x52:
          case 0x53:
          case 0x54:
          case 0x55:
            LOBYTE(v85) = 6;
            goto LABEL_14;
          case 0x37:
            LOBYTE(v85) = 7;
            v86 = 0;
            v74 = 1;
            goto LABEL_15;
          case 0x56:
          case 0x57:
          case 0x58:
          case 0x62:
          case 0x63:
          case 0x64:
            LOBYTE(v85) = 8;
            goto LABEL_14;
          case 0x59:
          case 0x5A:
          case 0x5B:
          case 0x5C:
          case 0x5D:
          case 0x65:
          case 0x66:
          case 0x67:
          case 0x68:
          case 0x69:
            LOBYTE(v85) = 9;
            goto LABEL_14;
          case 0x5E:
          case 0x5F:
          case 0x60:
          case 0x61:
          case 0x6A:
          case 0x6B:
          case 0x6C:
          case 0x6D:
            LOBYTE(v85) = 10;
LABEL_14:
            v86 = 0;
            v74 = (unsigned __int16)word_435D740[(unsigned __int8)(v83[0] - 14)];
            break;
        }
        goto LABEL_15;
      }
      v31 = sub_1F596B0((__int64)v83);
      v86 = v32;
      LOBYTE(v85) = v31;
      if ( v83[0] )
      {
        v74 = (unsigned __int16)word_435D740[(unsigned __int8)(v83[0] - 14)];
      }
      else
      {
        v71 = v31;
        v33 = sub_1F58D30((__int64)v83);
        v31 = v71;
        v74 = v33;
      }
      if ( v31 )
      {
LABEL_15:
        v70 = sub_1F3E310(&v85);
        goto LABEL_16;
      }
      v70 = sub_1F58D40((__int64)&v85);
LABEL_16:
      if ( v74 == 4 )
      {
        v65 = 666;
        v87 = v89;
        v88 = 0x800000000LL;
        v63 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a5 + 32));
        LODWORD(v88) = 1;
        v89[0] = v63;
LABEL_20:
        v64 = a5;
        v67 = v12;
        v21 = 0;
        v66 = v13;
        do
        {
          *(_QWORD *)&v25 = sub_1D38E70(a7, v21, (__int64)&v81, 0, v11, a2, a3);
          v79 = v66 | v79 & 0xFFFFFFFF00000000LL;
          *(_QWORD *)&v26 = sub_1D332F0(
                              (__int64 *)a7,
                              106,
                              (__int64)&v81,
                              v85,
                              v86,
                              0,
                              *(double *)v11.m128i_i64,
                              a2,
                              a3,
                              v67,
                              v79,
                              v25);
          v29 = *((_QWORD *)&v26 + 1);
          v28 = v26;
          if ( v70 <= 0xF )
          {
            v28 = sub_1D309E0(
                    (__int64 *)a7,
                    144,
                    (__int64)&v81,
                    4,
                    0,
                    0,
                    *(double *)v11.m128i_i64,
                    a2,
                    *(double *)a3.m128i_i64,
                    v26);
            v29 = v30 | v29 & 0xFFFFFFFF00000000LL;
          }
          v22 = (unsigned int)v88;
          if ( (unsigned int)v88 >= HIDWORD(v88) )
          {
            sub_16CD150((__int64)&v87, v89, 0, 16, v27, v62);
            v22 = (unsigned int)v88;
          }
          v23 = (__int64 *)&v87[v22];
          ++v21;
          *v23 = v28;
          v23[1] = v29;
          v24 = (unsigned int)(v88 + 1);
          LODWORD(v88) = v88 + 1;
        }
        while ( v74 > (unsigned int)v21 );
        v49 = v64;
        goto LABEL_43;
      }
      if ( v74 != 8 )
      {
        if ( v74 != 2 )
        {
LABEL_5:
          v17 = 0;
          goto LABEL_6;
        }
        v65 = 665;
        v87 = v89;
        v88 = 0x800000000LL;
        v20 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a5 + 32));
        LODWORD(v88) = 1;
        v89[0] = v20;
        goto LABEL_20;
      }
      v75 = v12;
      v87 = v89;
      v88 = 0x800000000LL;
      v72 = v13;
      v34 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a5 + 32));
      v68 = a5;
      v35 = 0;
      LODWORD(v88) = 1;
      v89[0] = v34;
      do
      {
        *(_QWORD *)&v36 = sub_1D38E70(a7, v35, (__int64)&v81, 0, v11, a2, v34);
        v79 = v72 | v79 & 0xFFFFFFFF00000000LL;
        v37 = sub_1D332F0((__int64 *)a7, 106, (__int64)&v81, 8, 0, 0, *(double *)v11.m128i_i64, a2, v34, v75, v79, v36);
        v39 = v38;
        v40 = (__int64)v37;
        *(_QWORD *)&v41 = sub_1D38E70(a7, v35 + 1, (__int64)&v81, 0, v11, a2, v34);
        *(_QWORD *)&v42 = sub_1D332F0(
                            (__int64 *)a7,
                            106,
                            (__int64)&v81,
                            8,
                            0,
                            0,
                            *(double *)v11.m128i_i64,
                            a2,
                            v34,
                            v75,
                            v79,
                            v41);
        v44 = sub_1D332F0((__int64 *)a7, 104, (__int64)&v81, 86, 0, 0, *(double *)v11.m128i_i64, a2, v34, v40, v39, v42);
        v46 = v45;
        v47 = (unsigned int)v88;
        if ( (unsigned int)v88 >= HIDWORD(v88) )
        {
          sub_16CD150((__int64)&v87, v89, 0, 16, v43, v62);
          v47 = (unsigned int)v88;
        }
        v48 = (__int64 **)&v87[v47];
        v35 += 2;
        *v48 = v44;
        v48[1] = v46;
        v24 = (unsigned int)(v88 + 1);
        LODWORD(v88) = v88 + 1;
      }
      while ( v35 != 8 );
      v65 = 666;
      v49 = v68;
LABEL_43:
      v50 = *(const __m128i **)(v49 + 32);
      v51 = 5LL * *(unsigned int *)(v49 + 56);
      v52 = (__int64)&v50->m128i_i64[v51];
      v53 = v50 + 5;
      v54 = 0xCCCCCCCCCCCCCCCDLL * ((v51 * 8 - 80) >> 3);
      if ( v54 > (unsigned __int64)HIDWORD(v88) - v24 )
      {
        v78 = v52;
        sub_16CD150((__int64)&v87, v89, v54 + v24, 16, v52, v62);
        v24 = (unsigned int)v88;
        v52 = v78;
      }
      v55 = (unsigned __int64)v87;
      v56 = (__m128i *)&v87[v24];
      if ( v53 != (const __m128i *)v52 )
      {
        do
        {
          if ( v56 )
            *v56 = _mm_loadu_si128(v53);
          v53 = (const __m128i *)((char *)v53 + 40);
          ++v56;
        }
        while ( (const __m128i *)v52 != v53 );
        v55 = (unsigned __int64)v87;
        LODWORD(v24) = v88;
      }
      v57 = v54 + v24;
      v58 = *(_QWORD *)(v49 + 96);
      v77 = (__int64 *)v55;
      v59 = *(_BYTE *)(v49 + 88);
      v76 = *(_QWORD *)(v49 + 104);
      v80 = v57;
      LODWORD(v88) = v57;
      v60 = sub_1D29190(a7, 1u, 0, (__int64)v56, v52, v76);
      v17 = sub_1D24DC0((_QWORD *)a7, v65, (__int64)&v81, v60, v61, v76, v77, v80, v59, v58);
      if ( v87 != v89 )
        _libc_free((unsigned __int64)v87);
LABEL_6:
      if ( v81 )
        sub_161E7C0((__int64)&v81, v81);
      return v17;
    default:
      goto LABEL_5;
  }
}
