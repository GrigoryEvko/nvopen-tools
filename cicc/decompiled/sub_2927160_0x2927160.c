// Function: sub_2927160
// Address: 0x2927160
//
void __fastcall sub_2927160(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  _QWORD *v6; // rax
  const __m128i *v7; // rdx
  __int64 *v8; // rax
  __int64 *v9; // rax
  __int64 v10; // rsi
  __int64 v11; // rax
  __int64 *v12; // r8
  __int64 v13; // r9
  __int64 v14; // rax
  _QWORD **v15; // rcx
  __int64 v16; // rax
  unsigned int v17; // r15d
  __int64 v18; // r12
  _QWORD *v19; // r14
  _BYTE *v20; // rax
  unsigned __int64 v21; // rdi
  unsigned __int64 v22; // rdx
  __int64 v23; // r12
  __int64 v24; // rax
  bool v25; // zf
  _BYTE *v26; // rbx
  unsigned __int64 v27; // r12
  unsigned __int64 v28; // rdi
  __int64 v29; // rax
  __int64 v30; // rdx
  __m128i *v31; // r12
  __int64 v32; // rax
  const __m128i *v33; // rsi
  signed __int64 v34; // rax
  __m128i *v35; // rdi
  const __m128i *v36; // rax
  size_t v37; // rdx
  __m128i *v38; // r13
  unsigned int v39; // eax
  __int64 v40; // rax
  const __m128i *v41; // r13
  _BYTE *v42; // rbx
  unsigned __int64 v43; // rdi
  __int64 v44; // rsi
  _QWORD *v45; // rax
  __m128i *v46; // rdx
  unsigned __int64 v47; // rsi
  _QWORD *v48; // rax
  char v49; // al
  _QWORD *v50; // rax
  const __m128i *v51; // rdx
  __int64 v52; // rsi
  unsigned __int64 v53; // rcx
  __int64 v54; // r8
  unsigned int v55; // eax
  _QWORD *v56; // rdx
  int v57; // eax
  bool v58; // al
  __int64 v59; // rsi
  _BOOL8 v60; // r8
  char v61; // al
  bool v62; // al
  unsigned __int64 v63; // [rsp+8h] [rbp-2E8h]
  _QWORD **v64; // [rsp+10h] [rbp-2E0h]
  _QWORD **v65; // [rsp+10h] [rbp-2E0h]
  unsigned int v66; // [rsp+18h] [rbp-2D8h]
  __int64 v67; // [rsp+18h] [rbp-2D8h]
  _QWORD **v68; // [rsp+20h] [rbp-2D0h]
  unsigned __int64 v69; // [rsp+20h] [rbp-2D0h]
  _QWORD *v70; // [rsp+20h] [rbp-2D0h]
  _QWORD *v71; // [rsp+40h] [rbp-2B0h] BYREF
  const __m128i *v72; // [rsp+48h] [rbp-2A8h]
  __m128i *v73; // [rsp+50h] [rbp-2A0h]
  __int64 v74; // [rsp+60h] [rbp-290h] BYREF
  __int64 v75; // [rsp+68h] [rbp-288h]
  __int64 v76; // [rsp+70h] [rbp-280h]
  __int64 v77; // [rsp+78h] [rbp-278h]
  _BYTE *v78; // [rsp+80h] [rbp-270h]
  __int64 v79; // [rsp+88h] [rbp-268h]
  _BYTE v80[192]; // [rsp+90h] [rbp-260h] BYREF
  __int64 v81; // [rsp+150h] [rbp-1A0h]
  char *v82; // [rsp+158h] [rbp-198h]
  __int64 v83; // [rsp+160h] [rbp-190h]
  int v84; // [rsp+168h] [rbp-188h]
  char v85; // [rsp+16Ch] [rbp-184h]
  char v86; // [rsp+170h] [rbp-180h] BYREF
  unsigned __int64 v87; // [rsp+1B0h] [rbp-140h]
  char v88; // [rsp+1B8h] [rbp-138h]
  _QWORD *v89; // [rsp+1C0h] [rbp-130h] BYREF
  unsigned int v90; // [rsp+1C8h] [rbp-128h]
  unsigned __int64 v91; // [rsp+1D0h] [rbp-120h]
  _BYTE *v92; // [rsp+1D8h] [rbp-118h]
  __int64 v93; // [rsp+1E0h] [rbp-110h]
  __int64 v94; // [rsp+1E8h] [rbp-108h]
  __int64 v95; // [rsp+1F0h] [rbp-100h] BYREF
  unsigned int v96; // [rsp+1F8h] [rbp-F8h]
  __int64 v97; // [rsp+230h] [rbp-C0h] BYREF
  __int64 v98; // [rsp+238h] [rbp-B8h]
  __int64 v99; // [rsp+240h] [rbp-B0h] BYREF
  unsigned int v100; // [rsp+248h] [rbp-A8h]
  __int64 v101; // [rsp+280h] [rbp-70h] BYREF
  char *v102; // [rsp+288h] [rbp-68h]
  __int64 v103; // [rsp+290h] [rbp-60h]
  int v104; // [rsp+298h] [rbp-58h]
  char v105; // [rsp+29Ch] [rbp-54h]
  char v106; // [rsp+2A0h] [rbp-50h] BYREF

  *(_QWORD *)(a1 + 24) = a1 + 40;
  *(_QWORD *)(a1 + 232) = a1 + 248;
  *(_BYTE *)a1 = a4;
  *(_QWORD *)(a1 + 32) = 0x800000000LL;
  *(_QWORD *)(a1 + 240) = 0x800000000LL;
  *(_QWORD *)(a1 + 312) = a1 + 328;
  *(_QWORD *)(a1 + 320) = 0x800000000LL;
  *(_QWORD *)(a1 + 400) = 0x800000000LL;
  v79 = 0x800000000LL;
  *(_QWORD *)(a1 + 392) = a1 + 408;
  v82 = &v86;
  v78 = v80;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  v74 = a2;
  v75 = 0;
  v76 = 0;
  v77 = 0;
  v81 = 0;
  v83 = 8;
  v84 = 0;
  v85 = 1;
  v90 = 1;
  v89 = 0;
  v6 = (_QWORD *)sub_BDB740(a2, *(_QWORD *)(a3 + 72));
  v92 = (_BYTE *)a1;
  v93 = 0;
  v94 = 1;
  v71 = v6;
  v72 = v7;
  v91 = (unsigned __int64)v6;
  v8 = &v95;
  do
  {
    *v8 = -4096;
    v8 += 2;
  }
  while ( v8 != &v97 );
  v97 = 0;
  v9 = &v99;
  v98 = 1;
  do
  {
    *v9 = -4096;
    v9 += 2;
  }
  while ( v9 != &v101 );
  v10 = *(_QWORD *)(a3 + 8);
  v101 = 0;
  v102 = &v106;
  v103 = 4;
  v104 = 0;
  v105 = 1;
  v11 = sub_AE4570(v74, v10);
  v88 = 1;
  LODWORD(v72) = *(_DWORD *)(v11 + 8) >> 8;
  if ( (unsigned int)v72 > 0x40 )
    sub_C43690((__int64)&v71, 0, 0);
  else
    v71 = 0;
  if ( v90 > 0x40 && v89 )
    j_j___libc_free_0_0((unsigned __int64)v89);
  v75 = 0;
  v76 = 0;
  v89 = v71;
  v90 = (unsigned int)v72;
  sub_3109010(&v74, a3);
  v14 = (unsigned int)v79;
  v15 = &v71;
  if ( (_DWORD)v79 )
  {
    while ( 2 )
    {
      v16 = (__int64)&v78[24 * v14 - 24];
      v17 = *(_DWORD *)(v16 + 16);
      v18 = *(_QWORD *)v16;
      *(_DWORD *)(v16 + 16) = 0;
      v19 = *(_QWORD **)(v16 + 8);
      LODWORD(v79) = v79 - 1;
      v20 = &v78[24 * (unsigned int)v79];
      if ( *((_DWORD *)v20 + 4) > 0x40u )
      {
        v21 = *((_QWORD *)v20 + 1);
        if ( v21 )
          j_j___libc_free_0_0(v21);
      }
      v22 = v18 & 0xFFFFFFFFFFFFFFF8LL;
      v87 = v18 & 0xFFFFFFFFFFFFFFF8LL;
      v88 = (v18 >> 2) & 1;
      if ( v88 )
      {
        if ( v90 > 0x40 && v89 )
        {
          j_j___libc_free_0_0((unsigned __int64)v89);
          v22 = v87;
        }
        v90 = v17;
        v17 = 0;
        v89 = v19;
      }
      v23 = *(_QWORD *)(v22 + 24);
      switch ( *(_BYTE *)v23 )
      {
        case 0x1E:
        case 0x1F:
        case 0x20:
        case 0x21:
        case 0x23:
        case 0x24:
        case 0x25:
        case 0x26:
        case 0x27:
        case 0x29:
        case 0x2A:
        case 0x2B:
        case 0x2C:
        case 0x2D:
        case 0x2E:
        case 0x2F:
        case 0x30:
        case 0x31:
        case 0x32:
        case 0x33:
        case 0x34:
        case 0x35:
        case 0x36:
        case 0x37:
        case 0x38:
        case 0x39:
        case 0x3A:
        case 0x3B:
        case 0x3C:
        case 0x40:
        case 0x41:
        case 0x42:
        case 0x43:
        case 0x44:
        case 0x45:
        case 0x46:
        case 0x47:
        case 0x48:
        case 0x49:
        case 0x4A:
        case 0x4B:
        case 0x4D:
        case 0x50:
        case 0x51:
        case 0x52:
        case 0x53:
        case 0x57:
        case 0x58:
        case 0x59:
        case 0x5A:
        case 0x5B:
        case 0x5C:
        case 0x5D:
        case 0x5E:
        case 0x5F:
        case 0x60:
          v75 = *(_QWORD *)(v22 + 24);
          goto LABEL_17;
        case 0x22:
        case 0x28:
          goto LABEL_48;
        case 0x3D:
          if ( !*v92 && !sub_B46500(*(unsigned __int8 **)(v22 + 24)) && (*(_BYTE *)(v23 + 2) & 1) == 0 )
          {
            v49 = sub_2918CB0(*(_QWORD *)(v23 + 8));
            v30 = v23;
            if ( !v49 )
              goto LABEL_120;
          }
          if ( !v88 )
            goto LABEL_119;
          v50 = (_QWORD *)sub_9C6480(v74, *(_QWORD *)(v23 + 8));
          v72 = v51;
          v71 = v50;
          if ( (_BYTE)v51 )
            goto LABEL_119;
          if ( *v92 )
            goto LABEL_128;
          v52 = *(_QWORD *)(v23 + 8);
          v53 = (unsigned __int64)v71;
          if ( *(_BYTE *)(v52 + 8) == 12 && (*(_BYTE *)(v23 + 2) & 1) == 0 )
          {
            v70 = v71;
            v62 = sub_2919770(v74, v52);
            v53 = (unsigned __int64)v70;
            v54 = v62;
          }
          else
          {
            v54 = 0;
          }
          sub_2916EE0((__int64)&v74, v23, (__int64 *)&v89, v53, v54);
          v30 = v75;
          goto LABEL_49;
        case 0x3E:
          v15 = *(_QWORD ***)(v23 - 64);
          if ( v15 == *(_QWORD ***)v22 )
          {
            v76 = *(_QWORD *)(v22 + 24);
            v30 = v23;
            v75 = v23;
            goto LABEL_49;
          }
          if ( !v88
            || (v68 = *(_QWORD ***)(v23 - 64),
                v45 = (_QWORD *)sub_9C6480(v74, (__int64)v15[1]),
                v15 = v68,
                v72 = v46,
                v71 = v45,
                (_BYTE)v46) )
          {
LABEL_119:
            v30 = v23;
LABEL_120:
            v75 = v30;
            goto LABEL_49;
          }
          if ( *v92 )
            goto LABEL_128;
          v47 = v91;
          v69 = (unsigned __int64)v71;
          if ( (unsigned __int64)v71 > v91 )
          {
LABEL_103:
            sub_2916B30((__int64)&v74, v23, v46->m128i_i64, (__int64)v15, (__int64)v12, v13);
            v30 = v75;
            goto LABEL_49;
          }
          v46 = (__m128i *)v90;
          v66 = v90;
          if ( v90 > 0x40 )
          {
            v63 = v91;
            v65 = v15;
            v46 = (__m128i *)(v66 - (unsigned int)sub_C444A0((__int64)&v89));
            if ( (unsigned int)v46 > 0x40 )
              goto LABEL_103;
            v15 = v65;
            v47 = v63;
            v48 = (_QWORD *)*v89;
          }
          else
          {
            v48 = v89;
          }
          v64 = v15;
          if ( v47 - v69 < (unsigned __int64)v48 )
            goto LABEL_103;
          v58 = sub_B46500((unsigned __int8 *)v23);
          LOBYTE(v12) = *(_BYTE *)(v23 + 2) & 1;
          if ( v58 )
          {
            v59 = (__int64)v64[1];
          }
          else
          {
            if ( (_BYTE)v12 )
              goto LABEL_159;
            v67 = (__int64)v64[1];
            v61 = sub_2918CB0(v67);
            v59 = v67;
            v12 = 0;
            v30 = v23;
            if ( !v61 )
              goto LABEL_120;
          }
          if ( *(_BYTE *)(v59 + 8) != 12 || (_BYTE)v12 )
          {
LABEL_159:
            v60 = 0;
            goto LABEL_160;
          }
          v60 = sub_2919770(v74, v59);
LABEL_160:
          sub_2916EE0((__int64)&v74, v23, (__int64 *)&v89, v69, v60);
          v30 = v75;
LABEL_49:
          if ( v30 )
          {
LABEL_17:
            if ( v17 > 0x40 && v19 )
              j_j___libc_free_0_0((unsigned __int64)v19);
            break;
          }
          if ( v17 > 0x40 && v19 )
            j_j___libc_free_0_0((unsigned __int64)v19);
          v14 = (unsigned int)v79;
          if ( (_DWORD)v79 )
            continue;
          v24 = v76;
          if ( !v76 )
            goto LABEL_55;
          goto LABEL_19;
        case 0x3F:
          v44 = *(_QWORD *)(v22 + 24);
          if ( !*(_QWORD *)(v23 + 16) )
            goto LABEL_121;
          if ( !(unsigned __int8)sub_3108E30(&v74, v44) )
          {
            v88 = 0;
            LODWORD(v72) = 1;
            v71 = 0;
            if ( v90 > 0x40 && v89 )
            {
              j_j___libc_free_0_0((unsigned __int64)v89);
              v56 = v71;
              v57 = (int)v72;
            }
            else
            {
              v57 = 1;
              v56 = 0;
            }
            v90 = v57;
            v89 = v56;
            LODWORD(v72) = 0;
            sub_969240((__int64 *)&v71);
          }
          sub_3109010(&v74, v23);
          v30 = v75;
          goto LABEL_49;
        case 0x4C:
          v76 = *(_QWORD *)(v22 + 24);
          v30 = v75;
          goto LABEL_49;
        case 0x4E:
        case 0x4F:
          v44 = *(_QWORD *)(v22 + 24);
          if ( *(_QWORD *)(v23 + 16) )
          {
            sub_3109010(&v74, v44);
            v30 = v75;
          }
          else
          {
LABEL_121:
            sub_2916B30((__int64)&v74, v44, (__int64 *)v22, (__int64)v15, (__int64)v12, v13);
            v30 = v75;
          }
          goto LABEL_49;
        case 0x54:
        case 0x56:
          sub_2925AD0((__int64)&v74, *(_QWORD *)(v22 + 24), (__int64 *)v22, (__int64)v15, v12, v13);
          v30 = v75;
          goto LABEL_49;
        case 0x55:
          v29 = *(_QWORD *)(v23 - 32);
          if ( !v29 )
            goto LABEL_48;
          if ( *(_BYTE *)v29 )
            goto LABEL_48;
          v15 = *(_QWORD ***)(v23 + 80);
          if ( *(_QWORD ***)(v29 + 24) != v15 )
            goto LABEL_48;
          v55 = *(_DWORD *)(v29 + 36);
          if ( v55 > 0xF5 )
            goto LABEL_127;
          if ( v55 > 0xED )
          {
            switch ( v55 )
            {
              case 0xEEu:
              case 0xF0u:
              case 0xF1u:
                sub_2926B40((__int64)&v74, *(_QWORD *)(v22 + 24), v22, (__int64)v15, (unsigned __int64)v12, v13);
                v30 = v75;
                break;
              case 0xF3u:
              case 0xF5u:
                sub_2917140((__int64)&v74, *(_QWORD *)(v22 + 24), v22, (__int64)v15, (__int64)v12, v13);
                v30 = v75;
                break;
              default:
                goto LABEL_127;
            }
          }
          else
          {
            if ( v55 <= 0x47 )
            {
              v30 = v75;
              if ( v55 > 0x44 )
                goto LABEL_49;
              if ( !v55 )
              {
LABEL_48:
                sub_2918CF0(&v74, (unsigned __int8 *)v23);
                v30 = v75;
                goto LABEL_49;
              }
            }
LABEL_127:
            sub_2918EF0((__int64)&v74, v23);
LABEL_128:
            v30 = v75;
          }
          goto LABEL_49;
        default:
          BUG();
      }
      break;
    }
  }
  v24 = v76;
  if ( v76 )
  {
LABEL_19:
    v25 = v105 == 0;
    *(_QWORD *)(a1 + 8) = v24;
    if ( v25 )
      _libc_free((unsigned __int64)v102);
    if ( (v98 & 1) != 0 )
    {
      if ( (v94 & 1) != 0 )
        goto LABEL_23;
    }
    else
    {
      sub_C7D6A0(v99, 16LL * v100, 8);
      if ( (v94 & 1) != 0 )
      {
LABEL_23:
        if ( v90 <= 0x40 )
          goto LABEL_24;
LABEL_38:
        if ( v89 )
        {
          j_j___libc_free_0_0((unsigned __int64)v89);
          if ( v85 )
            goto LABEL_25;
          goto LABEL_40;
        }
LABEL_24:
        if ( v85 )
        {
LABEL_25:
          v26 = v78;
          v27 = (unsigned __int64)&v78[24 * (unsigned int)v79];
          if ( v78 == (_BYTE *)v27 )
            goto LABEL_31;
          do
          {
            v27 -= 24LL;
            if ( *(_DWORD *)(v27 + 16) > 0x40u )
            {
              v28 = *(_QWORD *)(v27 + 8);
              if ( v28 )
                j_j___libc_free_0_0(v28);
            }
          }
          while ( v26 != (_BYTE *)v27 );
          goto LABEL_30;
        }
LABEL_40:
        _libc_free((unsigned __int64)v82);
        goto LABEL_25;
      }
    }
    sub_C7D6A0(v95, 16LL * v96, 8);
    if ( v90 <= 0x40 )
      goto LABEL_24;
    goto LABEL_38;
  }
LABEL_55:
  v24 = v75;
  if ( v75 )
    goto LABEL_19;
  v31 = *(__m128i **)(a1 + 24);
  *(_QWORD *)(a1 + 16) = v77;
  v32 = 24LL * *(unsigned int *)(a1 + 32);
  v33 = (__m128i *)((char *)v31 + v32);
  v34 = 0xAAAAAAAAAAAAAAABLL * (v32 >> 3);
  if ( !(v34 >> 2) )
  {
    v35 = v31;
LABEL_135:
    if ( v34 != 2 )
    {
      if ( v34 != 3 )
      {
        if ( v34 != 1 )
          goto LABEL_138;
        goto LABEL_153;
      }
      if ( (v35[1].m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL) == 0 )
        goto LABEL_63;
      v35 = (__m128i *)((char *)v35 + 24);
    }
    if ( (v35[1].m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL) == 0 )
      goto LABEL_63;
    v35 = (__m128i *)((char *)v35 + 24);
LABEL_153:
    if ( (v35[1].m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL) == 0 )
      goto LABEL_63;
LABEL_138:
    v35 = (__m128i *)v33;
LABEL_139:
    v38 = v35;
    goto LABEL_70;
  }
  v35 = v31;
  while ( (v35[1].m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL) != 0 )
  {
    if ( (v35[2].m128i_i64[1] & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    {
      v35 = (__m128i *)((char *)v35 + 24);
      break;
    }
    if ( (v35[4].m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    {
      v35 += 3;
      break;
    }
    if ( (v35[5].m128i_i64[1] & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    {
      v35 = (__m128i *)((char *)v35 + 72);
      break;
    }
    v35 += 6;
    if ( &v31[6 * (v34 >> 2)] == v35 )
    {
      v34 = 0xAAAAAAAAAAAAAAABLL * (((char *)v33 - (char *)v35) >> 3);
      goto LABEL_135;
    }
  }
LABEL_63:
  if ( v33 == v35 )
    goto LABEL_139;
  v36 = (__m128i *)((char *)v35 + 24);
  if ( v33 == (const __m128i *)&v35[1].m128i_u64[1] )
    goto LABEL_139;
  do
  {
    if ( (v36[1].m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL) != 0 )
    {
      v35 = (__m128i *)((char *)v35 + 24);
      *(__m128i *)((char *)v35 - 24) = _mm_loadu_si128(v36);
      v35[-1].m128i_i64[1] = v36[1].m128i_i64[0];
    }
    v36 = (const __m128i *)((char *)v36 + 24);
  }
  while ( v33 != v36 );
  v31 = *(__m128i **)(a1 + 24);
  v37 = (char *)v31 + 24 * *(unsigned int *)(a1 + 32) - (char *)v33;
  v38 = (__m128i *)((char *)v35 + v37);
  if ( (__m128i *)((char *)v31 + 24 * *(unsigned int *)(a1 + 32)) != v33 )
  {
    memmove(v35, v33, v37);
    v31 = *(__m128i **)(a1 + 24);
  }
LABEL_70:
  v39 = -1431655765 * (((char *)v38 - (char *)v31) >> 3);
  *(_DWORD *)(a1 + 32) = v39;
  v40 = 3LL * v39;
  v41 = (__m128i *)((char *)v31 + 8 * v40);
  sub_2912200((__int64 *)&v71, v31, 0xAAAAAAAAAAAAAAABLL * ((8 * v40) >> 3));
  if ( v73 )
    sub_2915A90(v31, v41, v73, v72);
  else
    sub_2914CE0(v31, v41);
  j_j___libc_free_0((unsigned __int64)v73);
  if ( !v105 )
    _libc_free((unsigned __int64)v102);
  if ( (v98 & 1) == 0 )
    sub_C7D6A0(v99, 16LL * v100, 8);
  if ( (v94 & 1) == 0 )
    sub_C7D6A0(v95, 16LL * v96, 8);
  if ( v90 > 0x40 && v89 )
    j_j___libc_free_0_0((unsigned __int64)v89);
  if ( !v85 )
    _libc_free((unsigned __int64)v82);
  v42 = v78;
  v27 = (unsigned __int64)&v78[24 * (unsigned int)v79];
  if ( v78 != (_BYTE *)v27 )
  {
    do
    {
      v27 -= 24LL;
      if ( *(_DWORD *)(v27 + 16) > 0x40u )
      {
        v43 = *(_QWORD *)(v27 + 8);
        if ( v43 )
          j_j___libc_free_0_0(v43);
      }
    }
    while ( v42 != (_BYTE *)v27 );
LABEL_30:
    v27 = (unsigned __int64)v78;
  }
LABEL_31:
  if ( (_BYTE *)v27 != v80 )
    _libc_free(v27);
}
