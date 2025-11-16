// Function: sub_2F2ACC0
// Address: 0x2f2acc0
//
__int64 __fastcall sub_2F2ACC0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rdi
  __int64 (*v6)(); // rax
  unsigned int v7; // r12d
  __int64 v11; // r9
  __int64 v12; // rdi
  __int64 (__fastcall *v13)(__int64, __int64); // rax
  __int64 v14; // rdi
  __int64 (__fastcall *v15)(__int64, __int64); // rcx
  __int64 v16; // rax
  __int64 v17; // r15
  __int64 v18; // r15
  __int64 v19; // r8
  __int64 v20; // rax
  unsigned int v21; // ebx
  _QWORD *v22; // r13
  __int64 v23; // r12
  unsigned int v24; // r14d
  __int64 v25; // rsi
  int v26; // eax
  __int64 v27; // rdx
  __int64 (__fastcall *v28)(__int64, __int64); // rax
  __int64 (__fastcall *v29)(__int64, __int64); // rcx
  __int64 v30; // rax
  unsigned __int64 v31; // rdx
  __int64 v32; // rax
  _QWORD *v33; // r14
  __int64 v34; // rbx
  __int64 *v35; // rcx
  __int64 v36; // rbx
  __int64 v37; // rdx
  __int64 v38; // rax
  __int64 v39; // rsi
  __int64 *v40; // rax
  unsigned __int64 v41; // r15
  __int64 *v42; // rbx
  __int64 v43; // r13
  __int64 *v44; // rax
  __int64 *v45; // rcx
  __int64 *v46; // rax
  __int64 v47; // rax
  unsigned __int64 v48; // rdx
  _QWORD *v49; // rax
  _QWORD *v50; // rdx
  __int64 v51; // rdx
  __int64 v52; // r8
  __int64 v53; // rsi
  __int64 (__fastcall *v54)(__int64, __int64); // rax
  __int64 v55; // rax
  void *v56; // r13
  __int64 v57; // rdi
  __int32 v58; // eax
  unsigned __int8 *v59; // rsi
  __int32 v60; // r12d
  __int64 v61; // rax
  _QWORD *v62; // rax
  __int64 v63; // rdx
  __int64 v64; // [rsp+10h] [rbp-210h]
  _BYTE *v65; // [rsp+20h] [rbp-200h]
  unsigned __int64 v66; // [rsp+28h] [rbp-1F8h]
  __int64 v67; // [rsp+28h] [rbp-1F8h]
  __int64 v68; // [rsp+30h] [rbp-1F0h]
  unsigned __int64 v69; // [rsp+40h] [rbp-1E0h]
  unsigned __int8 v70; // [rsp+4Fh] [rbp-1D1h]
  unsigned __int64 v71; // [rsp+50h] [rbp-1D0h]
  _DWORD *v73; // [rsp+58h] [rbp-1C8h]
  int v74; // [rsp+6Ch] [rbp-1B4h] BYREF
  int v75; // [rsp+70h] [rbp-1B0h] BYREF
  unsigned int v76; // [rsp+74h] [rbp-1ACh] BYREF
  unsigned __int8 *v77; // [rsp+78h] [rbp-1A8h] BYREF
  __int64 v78[4]; // [rsp+80h] [rbp-1A0h] BYREF
  __m128i v79; // [rsp+A0h] [rbp-180h] BYREF
  __int64 v80; // [rsp+B0h] [rbp-170h]
  __int64 v81; // [rsp+B8h] [rbp-168h]
  __int64 v82; // [rsp+C0h] [rbp-160h]
  __int64 v83; // [rsp+D0h] [rbp-150h] BYREF
  __int64 (__fastcall *v84)(__int64, __int64); // [rsp+D8h] [rbp-148h]
  __int64 v85; // [rsp+E0h] [rbp-140h]
  int v86; // [rsp+E8h] [rbp-138h]
  unsigned __int8 v87; // [rsp+ECh] [rbp-134h]
  char v88; // [rsp+F0h] [rbp-130h] BYREF
  __int64 v89; // [rsp+110h] [rbp-110h] BYREF
  __int64 *v90; // [rsp+118h] [rbp-108h]
  __int64 v91; // [rsp+120h] [rbp-100h]
  int v92; // [rsp+128h] [rbp-F8h]
  unsigned __int8 v93; // [rsp+12Ch] [rbp-F4h]
  char v94; // [rsp+130h] [rbp-F0h] BYREF
  _BYTE *v95; // [rsp+150h] [rbp-D0h] BYREF
  __int64 v96; // [rsp+158h] [rbp-C8h]
  _BYTE v97[64]; // [rsp+160h] [rbp-C0h] BYREF
  void *src; // [rsp+1A0h] [rbp-80h] BYREF
  __int64 v99; // [rsp+1A8h] [rbp-78h]
  _BYTE v100[112]; // [rsp+1B0h] [rbp-70h] BYREF

  v5 = a1[1];
  v74 = 0;
  v75 = 0;
  v6 = *(__int64 (**)())(*(_QWORD *)v5 + 80LL);
  if ( v6 == sub_2F28CC0 )
    return 0;
  v70 = ((__int64 (__fastcall *)(__int64, __int64, int *, int *, unsigned int *))v6)(v5, a2, &v74, &v75, &v76);
  if ( !v70 )
    return 0;
  if ( (unsigned int)(v75 - 1) <= 0x3FFFFFFE )
    return 0;
  if ( (unsigned int)(v74 - 1) <= 0x3FFFFFFE )
    return 0;
  v7 = sub_2EBEF70(a1[3], v74);
  if ( (_BYTE)v7 )
    return 0;
  v12 = a1[2];
  v69 = *(_QWORD *)(*(_QWORD *)(a1[3] + 56LL) + 16LL * (v75 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL;
  v13 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v12 + 272LL);
  if ( v13 != sub_2E85430 )
    v69 = ((__int64 (__fastcall *)(__int64, unsigned __int64, _QWORD))v13)(v12, v69, v76);
  if ( !v69 )
    return 0;
  v14 = a1[2];
  v15 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v14 + 272LL);
  v16 = a1[3];
  v71 = *(_QWORD *)(*(_QWORD *)(v16 + 56) + 16LL * (v74 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v15 != sub_2E85430 )
  {
    v71 = ((__int64 (__fastcall *)(__int64, unsigned __int64, _QWORD))v15)(v14, v71, v76);
    v16 = a1[3];
  }
  v83 = 0;
  v84 = (__int64 (__fastcall *)(__int64, __int64))&v88;
  v85 = 4;
  v86 = 0;
  v87 = 1;
  if ( v75 < 0 )
  {
    v17 = *(_QWORD *)(*(_QWORD *)(v16 + 56) + 16LL * (v75 & 0x7FFFFFFF) + 8);
  }
  else
  {
    v15 = *(__int64 (__fastcall **)(__int64, __int64))(v16 + 304);
    v17 = *((_QWORD *)v15 + (unsigned int)v75);
  }
  while ( v17 )
  {
    if ( (*(_BYTE *)(v17 + 3) & 0x10) == 0 && (*(_BYTE *)(v17 + 4) & 8) == 0 )
    {
      v51 = *(_QWORD *)(v17 + 16);
      v52 = v70;
LABEL_104:
      v53 = *(_QWORD *)(v51 + 24);
      if ( !(_BYTE)v52 )
        goto LABEL_116;
      v54 = v84;
      v15 = (__int64 (__fastcall *)(__int64, __int64))((char *)v84 + 8 * HIDWORD(v85));
      if ( v84 != v15 )
      {
        while ( v53 != *(_QWORD *)v54 )
        {
          v54 = (__int64 (__fastcall *)(__int64, __int64))((char *)v54 + 8);
          if ( v15 == v54 )
            goto LABEL_117;
        }
        v55 = v51;
        goto LABEL_110;
      }
LABEL_117:
      if ( HIDWORD(v85) < (unsigned int)v85 )
      {
        ++HIDWORD(v85);
        *(_QWORD *)v15 = v53;
        v55 = *(_QWORD *)(v17 + 16);
        ++v83;
        v52 = v87;
      }
      else
      {
LABEL_116:
        sub_C8CC70((__int64)&v83, v53, v51, (__int64)v15, v52, v11);
        v55 = *(_QWORD *)(v17 + 16);
        v52 = v87;
      }
LABEL_110:
      while ( 1 )
      {
        v17 = *(_QWORD *)(v17 + 32);
        if ( !v17 )
          break;
        while ( (*(_BYTE *)(v17 + 3) & 0x10) == 0 && (*(_BYTE *)(v17 + 4) & 8) == 0 )
        {
          v51 = *(_QWORD *)(v17 + 16);
          if ( v51 != v55 )
            goto LABEL_104;
          v17 = *(_QWORD *)(v17 + 32);
          if ( !v17 )
            goto LABEL_115;
        }
      }
LABEL_115:
      v16 = a1[3];
      break;
    }
    v17 = *(_QWORD *)(v17 + 32);
  }
  v96 = 0x800000000LL;
  v99 = 0x800000000LL;
  v95 = v97;
  src = v100;
  if ( v74 < 0 )
    v18 = *(_QWORD *)(*(_QWORD *)(v16 + 56) + 16LL * (v74 & 0x7FFFFFFF) + 8);
  else
    v18 = *(_QWORD *)(*(_QWORD *)(v16 + 304) + 8LL * (unsigned int)v74);
  if ( v18 )
  {
    if ( (*(_BYTE *)(v18 + 3) & 0x10) == 0 && (*(_BYTE *)(v18 + 4) & 8) == 0 )
      goto LABEL_24;
    v18 = *(_QWORD *)(v18 + 32);
    if ( v18 )
    {
      while ( (*(_BYTE *)(v18 + 3) & 0x10) != 0 || (*(_BYTE *)(v18 + 4) & 8) != 0 )
      {
        v18 = *(_QWORD *)(v18 + 32);
        if ( !v18 )
          goto LABEL_79;
      }
LABEL_24:
      v19 = v70;
      v20 = a2;
      v68 = a4;
      v21 = v7;
      v22 = a1;
      v23 = v20;
      v24 = v70;
LABEL_25:
      v25 = *(_QWORD *)(v18 + 16);
      if ( v23 == v25 )
        goto LABEL_40;
      v26 = *(unsigned __int16 *)(v25 + 68);
      if ( !*(_WORD *)(v25 + 68) || v26 == 68 )
      {
        v24 = 0;
        goto LABEL_40;
      }
      if ( v71 && v76 != ((*(_DWORD *)v18 >> 8) & 0xFFF) || v26 == 12 )
        goto LABEL_40;
      v27 = *(_QWORD *)(v25 + 24);
      if ( a3 == v27 )
      {
        if ( *(_BYTE *)(v68 + 28) )
        {
          v49 = *(_QWORD **)(v68 + 8);
          v50 = &v49[*(unsigned int *)(v68 + 20)];
          if ( v49 != v50 )
          {
            while ( v25 != *v49 )
            {
              if ( v50 == ++v49 )
                goto LABEL_37;
            }
            goto LABEL_40;
          }
        }
        else if ( sub_C8CA60(v68, v25) )
        {
          goto LABEL_40;
        }
      }
      else if ( v87 )
      {
        v28 = v84;
        v29 = (__int64 (__fastcall *)(__int64, __int64))((char *)v84 + 8 * HIDWORD(v85));
        if ( v84 == v29 )
        {
LABEL_92:
          if ( !(_BYTE)qword_5022BE8 || !(unsigned __int8)sub_2E6D360(v22[4], a3, v27) )
          {
            v32 = (unsigned int)v96;
            v7 = v21;
            v33 = v22;
            goto LABEL_46;
          }
          v47 = (unsigned int)v99;
          v48 = (unsigned int)v99 + 1LL;
          if ( v48 > HIDWORD(v99) )
          {
            sub_C8D5F0((__int64)&src, v100, v48, 8u, v19, v11);
            v47 = (unsigned int)v99;
          }
          *((_QWORD *)src + v47) = v18;
          LODWORD(v99) = v99 + 1;
LABEL_40:
          while ( 1 )
          {
            v18 = *(_QWORD *)(v18 + 32);
            if ( !v18 )
              break;
            while ( (*(_BYTE *)(v18 + 3) & 0x10) == 0 )
            {
              if ( (*(_BYTE *)(v18 + 4) & 8) == 0 )
                goto LABEL_25;
              v18 = *(_QWORD *)(v18 + 32);
              if ( !v18 )
                goto LABEL_44;
            }
          }
LABEL_44:
          v19 = v24;
          v32 = (unsigned int)v96;
          v7 = v21;
          v33 = v22;
          if ( (_BYTE)v19 )
          {
            v34 = (unsigned int)v99;
            if ( (_DWORD)v99 )
            {
              v56 = src;
              if ( (unsigned int)v99 + (unsigned __int64)(unsigned int)v96 > HIDWORD(v96) )
              {
                sub_C8D5F0((__int64)&v95, v97, (unsigned int)v99 + (unsigned __int64)(unsigned int)v96, 8u, v19, v11);
                v32 = (unsigned int)v96;
              }
              memcpy(&v95[8 * v32], v56, 8 * v34);
              LODWORD(v96) = v96 + v34;
              v32 = (unsigned int)v96;
            }
          }
LABEL_46:
          if ( !(_DWORD)v32 )
          {
LABEL_79:
            if ( src != v100 )
              _libc_free((unsigned __int64)src);
            goto LABEL_81;
          }
          v35 = (__int64 *)v33[3];
          v89 = 0;
          v90 = (__int64 *)&v94;
          v91 = 4;
          v92 = 0;
          v93 = 1;
          if ( v75 < 0 )
            v36 = *(_QWORD *)(v35[7] + 16LL * (v75 & 0x7FFFFFFF) + 8);
          else
            v36 = *(_QWORD *)(v35[38] + 8LL * (unsigned int)v75);
          LOBYTE(v37) = v70;
          if ( !v36 )
          {
LABEL_69:
            v66 = *(_QWORD *)(v35[7] + 16LL * (v74 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL;
            v65 = &v95[8 * v32];
            if ( v65 != v95 )
            {
              v41 = (unsigned __int64)v95;
              while ( 1 )
              {
                v42 = *(__int64 **)(*(_QWORD *)v41 + 16LL);
                v73 = *(_DWORD **)v41;
                v43 = v42[3];
                if ( (_BYTE)v37 )
                  break;
                if ( !sub_C8CA60((__int64)&v89, v42[3]) )
                  goto LABEL_122;
                LOBYTE(v37) = v93;
LABEL_76:
                v41 += 8LL;
                if ( v65 == (_BYTE *)v41 )
                  goto LABEL_77;
              }
              v44 = v90;
              v45 = &v90[HIDWORD(v91)];
              if ( v90 != v45 )
              {
                while ( v43 != *v44 )
                {
                  if ( v45 == ++v44 )
                    goto LABEL_122;
                }
                goto LABEL_76;
              }
LABEL_122:
              if ( !(_BYTE)v7 )
              {
                sub_2EBF120(v33[3], v75);
                sub_2EBE590(v33[3], v75, v69, 0);
              }
              v57 = v33[3];
              if ( v71 )
                v66 = *(_QWORD *)(*(_QWORD *)(v57 + 56) + 16LL * (*(_DWORD *)(v42[4] + 8) & 0x7FFFFFFF))
                    & 0xFFFFFFFFFFFFFFF8LL;
              v58 = sub_2EC06C0(v57, v66, byte_3F871B3, 0, v19, v11);
              v59 = (unsigned __int8 *)v42[7];
              v60 = v58;
              v61 = v33[1];
              v77 = v59;
              v64 = *(_QWORD *)(v61 + 8) - 800LL;
              if ( v59 )
              {
                sub_B96E90((__int64)&v77, (__int64)v59, 1);
                v78[0] = (__int64)v77;
                if ( v77 )
                {
                  sub_B976B0((__int64)&v77, v77, (__int64)v78);
                  v77 = 0;
                }
              }
              else
              {
                v78[0] = 0;
              }
              v78[1] = 0;
              v78[2] = 0;
              v62 = sub_2F2A600(v43, (__int64)v42, v78, v64, v60);
              v80 = 0;
              v79.m128i_i32[2] = v75;
              v81 = 0;
              v82 = 0;
              v79.m128i_i64[0] = (unsigned __int16)(v76 & 0xFFF) << 8;
              sub_2E8EAD0(v63, (__int64)v62, &v79);
              if ( v78[0] )
                sub_B91220((__int64)v78, v78[0]);
              if ( v77 )
                sub_B91220((__int64)&v77, (__int64)v77);
              if ( v71 )
                *v73 &= 0xFFF000FF;
              sub_2EAB0C0((__int64)v73, v60);
              LOBYTE(v37) = v93;
              v7 = v70;
              goto LABEL_76;
            }
LABEL_77:
            if ( !(_BYTE)v37 )
              _libc_free((unsigned __int64)v90);
            goto LABEL_79;
          }
          while ( (*(_BYTE *)(v36 + 3) & 0x10) != 0 || (*(_BYTE *)(v36 + 4) & 8) != 0 )
          {
            v36 = *(_QWORD *)(v36 + 32);
            if ( !v36 )
            {
              LOBYTE(v37) = v70;
              goto LABEL_69;
            }
          }
          v38 = *(_QWORD *)(v36 + 16);
          v37 = v70;
LABEL_55:
          if ( *(_WORD *)(v38 + 68) != 68 )
          {
            v35 = (__int64 *)v38;
            if ( *(_WORD *)(v38 + 68) )
            {
LABEL_63:
              while ( 1 )
              {
                v36 = *(_QWORD *)(v36 + 32);
                if ( !v36 )
                  break;
                while ( (*(_BYTE *)(v36 + 3) & 0x10) == 0 && (*(_BYTE *)(v36 + 4) & 8) == 0 )
                {
                  v38 = *(_QWORD *)(v36 + 16);
                  if ( v35 != (__int64 *)v38 )
                    goto LABEL_55;
                  v36 = *(_QWORD *)(v36 + 32);
                  if ( !v36 )
                    goto LABEL_68;
                }
              }
LABEL_68:
              v35 = (__int64 *)v33[3];
              v32 = (unsigned int)v96;
              goto LABEL_69;
            }
          }
          v39 = *(_QWORD *)(v38 + 24);
          if ( (_BYTE)v37 )
          {
            v40 = v90;
            v35 = &v90[HIDWORD(v91)];
            if ( v90 != v35 )
            {
              while ( v39 != *v40 )
              {
                if ( v35 == ++v40 )
                  goto LABEL_140;
              }
              goto LABEL_62;
            }
LABEL_140:
            if ( HIDWORD(v91) < (unsigned int)v91 )
            {
              ++HIDWORD(v91);
              *v35 = v39;
              v37 = v93;
              ++v89;
LABEL_62:
              v35 = *(__int64 **)(v36 + 16);
              goto LABEL_63;
            }
          }
          sub_C8CC70((__int64)&v89, v39, v37, (__int64)v35, v19, v11);
          v37 = v93;
          v35 = *(__int64 **)(v36 + 16);
          goto LABEL_63;
        }
        while ( v27 != *(_QWORD *)v28 )
        {
          v28 = (__int64 (__fastcall *)(__int64, __int64))((char *)v28 + 8);
          if ( v29 == v28 )
            goto LABEL_92;
        }
      }
      else
      {
        v67 = *(_QWORD *)(v25 + 24);
        v46 = sub_C8CA60((__int64)&v83, v27);
        v27 = v67;
        if ( !v46 )
          goto LABEL_92;
      }
LABEL_37:
      v30 = (unsigned int)v96;
      v31 = (unsigned int)v96 + 1LL;
      if ( v31 > HIDWORD(v96) )
      {
        sub_C8D5F0((__int64)&v95, v97, v31, 8u, v19, v11);
        v30 = (unsigned int)v96;
      }
      *(_QWORD *)&v95[8 * v30] = v18;
      LODWORD(v96) = v96 + 1;
      goto LABEL_40;
    }
  }
LABEL_81:
  if ( v95 != v97 )
    _libc_free((unsigned __int64)v95);
  if ( !v87 )
    _libc_free((unsigned __int64)v84);
  return v7;
}
