// Function: sub_2D4BB90
// Address: 0x2d4bb90
//
__int64 __fastcall sub_2D4BB90(unsigned __int64 *a1, _QWORD *a2)
{
  unsigned int v2; // r14d
  _QWORD **v4; // rax
  unsigned __int64 v5; // rdi
  _QWORD *v6; // r13
  __int64 (__fastcall *v7)(__int64, __int64); // rax
  unsigned int v9; // ebx
  unsigned int v10; // ebx
  int v11; // eax
  unsigned __int64 v12; // rax
  unsigned __int16 v13; // ax
  __int64 v14; // rbx
  __int64 v15; // r14
  __int64 v16; // rbx
  __int64 v17; // rsi
  unsigned __int64 v18; // rdi
  unsigned int v19; // ebx
  _QWORD *v20; // r13
  unsigned __int64 v21; // rdx
  __int16 v22; // bx
  __int64 v23; // rax
  _BYTE *v24; // rax
  size_t v25; // r15
  __int8 *v26; // rbx
  __int64 v27; // rsi
  __int64 v28; // r13
  __int64 v29; // rax
  __int8 *v30; // rax
  size_t v31; // rdx
  __int64 v32; // rdx
  __int64 v33; // rcx
  __int64 v34; // r8
  __int64 v35; // r9
  __m128i v36; // xmm0
  __m128i v37; // xmm1
  __m128i v38; // xmm2
  int v39; // r13d
  unsigned __int64 *v40; // r13
  unsigned __int64 *v41; // rbx
  unsigned __int64 *v42; // r13
  unsigned __int64 v43; // rdi
  __int64 *v44; // r13
  __int64 *v45; // r14
  __int64 v46; // rdx
  unsigned int v47; // esi
  int v48; // ebx
  __int64 v49; // rdx
  __int64 *v50; // rbx
  __int64 v51; // rdx
  unsigned int v52; // esi
  __int64 v53; // rax
  __int64 v54; // rax
  __int64 v55; // rax
  __int64 v56; // rsi
  __int64 *v57; // rax
  unsigned __int64 *v58; // rbx
  unsigned __int64 *v59; // rcx
  __int64 v60; // r8
  unsigned __int64 v61; // rdi
  __int64 v62; // [rsp+8h] [rbp-458h]
  __int64 v63; // [rsp+10h] [rbp-450h]
  unsigned __int64 *v64; // [rsp+10h] [rbp-450h]
  unsigned int v65; // [rsp+18h] [rbp-448h]
  __int64 v66; // [rsp+18h] [rbp-448h]
  __int64 *v67; // [rsp+18h] [rbp-448h]
  __m128i *v68; // [rsp+18h] [rbp-448h]
  __int64 v69; // [rsp+20h] [rbp-440h]
  __int64 *v70; // [rsp+20h] [rbp-440h]
  unsigned __int64 *v71; // [rsp+20h] [rbp-440h]
  _QWORD *v72; // [rsp+28h] [rbp-438h] BYREF
  char v73[32]; // [rsp+30h] [rbp-430h] BYREF
  __int16 v74; // [rsp+50h] [rbp-410h]
  __int64 v75[2]; // [rsp+60h] [rbp-400h] BYREF
  __int64 *v76; // [rsp+70h] [rbp-3F0h]
  char v77; // [rsp+80h] [rbp-3E0h]
  char v78; // [rsp+81h] [rbp-3DFh]
  _QWORD v79[2]; // [rsp+90h] [rbp-3D0h] BYREF
  _BYTE v80[16]; // [rsp+A0h] [rbp-3C0h] BYREF
  __int16 v81; // [rsp+B0h] [rbp-3B0h]
  _QWORD **v82; // [rsp+D0h] [rbp-390h] BYREF
  _QWORD **v83; // [rsp+D8h] [rbp-388h]
  __int64 v84; // [rsp+E0h] [rbp-380h]
  __m128i v85; // [rsp+E8h] [rbp-378h]
  __int64 v86; // [rsp+F8h] [rbp-368h]
  __m128i v87; // [rsp+100h] [rbp-360h]
  __m128i v88; // [rsp+110h] [rbp-350h]
  __int64 *v89; // [rsp+120h] [rbp-340h] BYREF
  __int64 v90; // [rsp+128h] [rbp-338h]
  _BYTE v91[320]; // [rsp+130h] [rbp-330h] BYREF
  char v92; // [rsp+270h] [rbp-1F0h]
  int v93; // [rsp+274h] [rbp-1ECh]
  __int64 v94; // [rsp+278h] [rbp-1E8h]
  __int64 *v95; // [rsp+280h] [rbp-1E0h] BYREF
  unsigned int v96; // [rsp+288h] [rbp-1D8h]
  char v97; // [rsp+28Ch] [rbp-1D4h]
  __int64 v98; // [rsp+290h] [rbp-1D0h] BYREF
  __m128i v99; // [rsp+298h] [rbp-1C8h] BYREF
  __int64 v100; // [rsp+2A8h] [rbp-1B8h]
  __m128i v101; // [rsp+2B0h] [rbp-1B0h] BYREF
  __m128i v102; // [rsp+2C0h] [rbp-1A0h] BYREF
  unsigned __int64 *v103; // [rsp+2D0h] [rbp-190h]
  __int64 v104; // [rsp+2D8h] [rbp-188h]
  __int64 v105; // [rsp+2E0h] [rbp-180h] BYREF
  int v106; // [rsp+2E8h] [rbp-178h]
  void *v107; // [rsp+300h] [rbp-160h]
  void *v108; // [rsp+308h] [rbp-158h]
  _QWORD v109[24]; // [rsp+360h] [rbp-100h] BYREF
  char v110; // [rsp+420h] [rbp-40h]
  int v111; // [rsp+424h] [rbp-3Ch]
  __int64 v112; // [rsp+428h] [rbp-38h]

  v72 = a2;
  v4 = (_QWORD **)sub_B43CA0((__int64)a2);
  v5 = *a1;
  v6 = *v4;
  v7 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)*a1 + 1184LL);
  if ( v7 != sub_2D42AB0 )
  {
    switch ( (unsigned int)v7(v5, (__int64)a2) )
    {
      case 0u:
        return 0;
      case 2u:
        v19 = *(_DWORD *)(*a1 + 96) >> 3;
        if ( v19 > (unsigned int)sub_2D44250((__int64)v72) )
        {
          v2 = 1;
          sub_2D4FA00(a1, v72, 2);
          return v2;
        }
        v20 = v72;
        v21 = a1[1];
        v82 = &v72;
        v63 = *(v72 - 8);
        v66 = v72[1];
        v22 = *((_WORD *)v72 + 1) >> 1;
        sub_2D46B10((__int64)&v95, (__int64)v72, v21);
        v23 = sub_2D46690(a1, (__int64)&v95, v66, v63, v22 & 7, (__int64)&v95, sub_2D42EB0, (__int64)&v82);
        sub_BD84D0((__int64)v20, v23);
        sub_B43D60(v20);
        sub_B32BF0(v109);
        v107 = &unk_49E5698;
        v108 = &unk_49D94D0;
        nullsub_63();
        nullsub_63();
        v18 = (unsigned __int64)v95;
        if ( v95 != &v98 )
          goto LABEL_22;
        return 1;
      case 4u:
        v5 = *a1;
        a2 = v72;
        goto LABEL_7;
      case 5u:
        v10 = *(_DWORD *)(*a1 + 96);
        if ( v10 >> 3 > (unsigned int)sub_2D44250((__int64)v72) )
        {
          v11 = (*((_WORD *)v72 + 1) >> 4) & 0x1F;
          LOBYTE(v2) = v11 == 3 || (unsigned int)(v11 - 5) <= 1;
          if ( (_BYTE)v2 )
          {
            v53 = sub_2D47CC0(a1, (__int64)v72);
            sub_2D4BB90(a1, v53);
            return v2;
          }
        }
        sub_2D46B10((__int64)&v95, (__int64)v72, a1[1]);
        _BitScanReverse64(&v12, 1LL << (*((_WORD *)v72 + 1) >> 9));
        sub_2D44EF0(
          (__int64)&v82,
          (__int64)&v95,
          (__int64)v72,
          v72[1],
          *(v72 - 8),
          63 - (v12 ^ 0x3F),
          *(_DWORD *)(*a1 + 96) >> 3);
        v13 = *((_WORD *)v72 + 1);
        v78 = 1;
        v77 = 3;
        v74 = 257;
        v75[0] = (__int64)"ValOperand_Shifted";
        v14 = *(v72 - 4);
        v69 = v86;
        if ( v82 == *(_QWORD ***)(v14 + 8) )
        {
          v15 = *(v72 - 4);
        }
        else
        {
          v62 = (__int64)v82;
          v65 = (((v13 >> 4) & 0x1Fu) - 7 < 2) + 39;
          v15 = (*(__int64 (__fastcall **)(unsigned __int64 *, _QWORD, __int64, _QWORD **))(*v103 + 120))(
                  v103,
                  v65,
                  v14,
                  v82);
          if ( !v15 )
          {
            v81 = 257;
            v15 = sub_B51D30(v65, v14, v62, (__int64)v79, 0, 0);
            if ( (unsigned __int8)sub_920620(v15) )
            {
              v48 = v106;
              if ( v105 )
                sub_B99FD0(v15, 3u, v105);
              sub_B45150(v15, v48);
            }
            (*(void (__fastcall **)(__int64, __int64, char *, __int64, __int64))(*(_QWORD *)v104 + 16LL))(
              v104,
              v15,
              v73,
              v101.m128i_i64[1],
              v102.m128i_i64[0]);
            v49 = 2LL * v96;
            v67 = &v95[v49];
            if ( v95 != &v95[v49] )
            {
              v50 = v95;
              do
              {
                v51 = v50[1];
                v52 = *(_DWORD *)v50;
                v50 += 2;
                sub_B99FD0(v15, v52, v51);
              }
              while ( v67 != v50 );
            }
          }
        }
        v16 = (*(__int64 (__fastcall **)(unsigned __int64 *, __int64, __int64, __int64, _QWORD, _QWORD))(*v103 + 32))(
                v103,
                25,
                v15,
                v69,
                0,
                0);
        if ( !v16 )
        {
          v81 = 257;
          v16 = sub_B504D0(25, v15, v69, (__int64)v79, 0, 0);
          (*(void (__fastcall **)(__int64, __int64, __int64 *, __int64, __int64))(*(_QWORD *)v104 + 16LL))(
            v104,
            v16,
            v75,
            v101.m128i_i64[1],
            v102.m128i_i64[0]);
          v45 = v95;
          v70 = &v95[2 * v96];
          if ( v95 != v70 )
          {
            do
            {
              v46 = v45[1];
              v47 = *(_DWORD *)v45;
              v45 += 2;
              sub_B99FD0(v16, v47, v46);
            }
            while ( v70 != v45 );
          }
        }
        v17 = (*(__int64 (__fastcall **)(unsigned __int64, __int64 **, _QWORD *, __int64, __int64, __int64, __int64, _QWORD))(*(_QWORD *)*a1 + 1048LL))(
                *a1,
                &v95,
                v72,
                v85.m128i_i64[0],
                v16,
                v87.m128i_i64[0],
                v86,
                (*((_WORD *)v72 + 1) >> 1) & 7);
        if ( v82 != v83 )
          v17 = sub_2D44750((__int64 *)&v95, v17, &v82);
        sub_BD84D0((__int64)v72, v17);
        sub_B43D60(v72);
        sub_B32BF0(v109);
        v107 = &unk_49E5698;
        v108 = &unk_49D94D0;
        nullsub_63();
        nullsub_63();
        v18 = (unsigned __int64)v95;
        if ( v95 != &v98 )
          goto LABEL_22;
        return 1;
      case 6u:
        v2 = 1;
        (*(void (__fastcall **)(unsigned __int64, _QWORD *))(*(_QWORD *)*a1 + 1072LL))(*a1, v72);
        return v2;
      case 7u:
        v2 = 1;
        (*(void (__fastcall **)(unsigned __int64, _QWORD *))(*(_QWORD *)*a1 + 1080LL))(*a1, v72);
        return v2;
      case 8u:
        (*(void (__fastcall **)(unsigned __int64, _QWORD *))(*(_QWORD *)*a1 + 1056LL))(*a1, v72);
        return 1;
      case 9u:
        return (unsigned int)sub_2A2DB90((__int64)v72);
      default:
        BUG();
    }
  }
  if ( ((*((_WORD *)v72 + 1) >> 4) & 0x1Fu) - 11 <= 3 )
  {
LABEL_7:
    v9 = *(_DWORD *)(v5 + 96) >> 3;
    if ( v9 <= (unsigned int)sub_2D44250((__int64)a2) )
    {
      v79[1] = 0x300000000LL;
      v79[0] = v80;
      sub_B6F820(v6);
      v24 = &v80[16 * *((unsigned __int8 *)v72 + 72)];
      v25 = *((_QWORD *)v24 + 1);
      if ( v25 )
      {
        v26 = *(__int8 **)v24;
      }
      else
      {
        v25 = 6;
        v26 = "system";
      }
      v27 = sub_B43CB0((__int64)v72);
      sub_1049690(v75, v27);
      v28 = v75[0];
      v29 = sub_B2BE50(v75[0]);
      if ( sub_B6EA50(v29)
        || (v54 = sub_B2BE50(v28),
            v55 = sub_B6F970(v54),
            (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v55 + 48LL))(v55)) )
      {
        sub_B174A0((__int64)&v95, (__int64)"atomic-expand", (__int64)"Passed", 6, (__int64)v72);
        sub_B18290((__int64)&v95, "A compare and swap loop was generated for an atomic ", 0x34u);
        v30 = sub_B4D7D0((*((_WORD *)v72 + 1) >> 4) & 0x1F);
        sub_B18290((__int64)&v95, v30, v31);
        sub_B18290((__int64)&v95, " operation at ", 0xEu);
        sub_B18290((__int64)&v95, v26, v25);
        sub_B18290((__int64)&v95, " memory scope", 0xDu);
        v36 = _mm_loadu_si128(&v99);
        v37 = _mm_loadu_si128(&v101);
        v38 = _mm_loadu_si128(&v102);
        v89 = (__int64 *)v91;
        LODWORD(v83) = v96;
        v39 = v104;
        v85 = v36;
        BYTE4(v83) = v97;
        v87 = v37;
        v84 = v98;
        v88 = v38;
        v82 = (_QWORD **)&unk_49D9D40;
        v86 = v100;
        v90 = 0x400000000LL;
        if ( (_DWORD)v104 )
        {
          v56 = (unsigned int)v104;
          v57 = (__int64 *)v91;
          if ( (unsigned int)v104 > 4 )
          {
            sub_11F02D0((__int64)&v89, (unsigned int)v104, v32, v33, v34, v35);
            v57 = v89;
            v56 = (unsigned int)v104;
          }
          v58 = v103;
          v59 = v103;
          v71 = &v103[10 * v56];
          if ( v103 != v71 )
          {
            do
            {
              if ( v57 )
              {
                v64 = v59;
                *v57 = (__int64)(v57 + 2);
                v68 = (__m128i *)v57;
                sub_2D42DE0(v57, (_BYTE *)*v59, *v59 + v59[1]);
                v68[2].m128i_i64[0] = (__int64)v68[3].m128i_i64;
                sub_2D42DE0(v68[2].m128i_i64, (_BYTE *)v64[4], v64[4] + v64[5]);
                v59 = v64;
                v57 = (__int64 *)v68;
                v68[4] = _mm_loadu_si128((const __m128i *)v64 + 4);
              }
              v59 += 10;
              v57 += 10;
            }
            while ( v71 != v59 );
            v58 = v103;
          }
          LODWORD(v90) = v39;
          v92 = v110;
          v93 = v111;
          v94 = v112;
          v82 = (_QWORD **)&unk_49D9D78;
          v95 = (__int64 *)&unk_49D9D40;
          v60 = 10LL * (unsigned int)v104;
          v40 = &v58[v60];
          if ( v58 != &v58[v60] )
          {
            do
            {
              v40 -= 10;
              v61 = v40[4];
              if ( (unsigned __int64 *)v61 != v40 + 6 )
                j_j___libc_free_0(v61);
              if ( (unsigned __int64 *)*v40 != v40 + 2 )
                j_j___libc_free_0(*v40);
            }
            while ( v58 != v40 );
            v40 = v103;
          }
        }
        else
        {
          v40 = v103;
          v92 = v110;
          v93 = v111;
          v94 = v112;
          v82 = (_QWORD **)&unk_49D9D78;
        }
        if ( v40 != (unsigned __int64 *)&v105 )
          _libc_free((unsigned __int64)v40);
        sub_1049740(v75, (__int64)&v82);
        v41 = (unsigned __int64 *)v89;
        v82 = (_QWORD **)&unk_49D9D40;
        v42 = (unsigned __int64 *)&v89[10 * (unsigned int)v90];
        if ( v89 != (__int64 *)v42 )
        {
          do
          {
            v42 -= 10;
            v43 = v42[4];
            if ( (unsigned __int64 *)v43 != v42 + 6 )
              j_j___libc_free_0(v43);
            if ( (unsigned __int64 *)*v42 != v42 + 2 )
              j_j___libc_free_0(*v42);
          }
          while ( v41 != v42 );
          v42 = (unsigned __int64 *)v89;
        }
        if ( v42 != (unsigned __int64 *)v91 )
          _libc_free((unsigned __int64)v42);
      }
      sub_2D48D60(
        (__int64)v72,
        (void (__fastcall *)(__int64, __int64, __int64, __int64, __int64, _QWORD, _QWORD, _QWORD, _QWORD *, __int64 *, __int64))sub_2D42AF0,
        (__int64)sub_2D45870);
      v44 = v76;
      if ( v76 )
      {
        sub_FDC110(v76);
        j_j___libc_free_0((unsigned __int64)v44);
      }
      v18 = v79[0];
      if ( (_BYTE *)v79[0] != v80 )
LABEL_22:
        _libc_free(v18);
      return 1;
    }
    else
    {
      v2 = 1;
      sub_2D4FA00(a1, v72, 4);
    }
  }
  else
  {
    return 0;
  }
  return v2;
}
