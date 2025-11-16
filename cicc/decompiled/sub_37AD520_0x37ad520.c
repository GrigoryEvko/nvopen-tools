// Function: sub_37AD520
// Address: 0x37ad520
//
__m128i *__fastcall sub_37AD520(__int64 *a1, __int64 a2, __m128i a3)
{
  unsigned int v3; // r15d
  __int64 v6; // rdi
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // r9
  __int64 v10; // rsi
  __int64 v11; // rax
  __int64 (__fastcall *v12)(__int64, __int64, unsigned int, __int64); // rax
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // rsi
  char v15; // cl
  __int64 *v16; // rdi
  int v17; // eax
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // rcx
  __int64 v21; // rdi
  __int64 v22; // rsi
  __int64 v23; // r9
  __m128i **v24; // rdi
  __int64 v25; // rsi
  _QWORD *v26; // r15
  __int64 v27; // r13
  __int64 v28; // r12
  __m128i *v29; // r12
  __int64 v31; // rsi
  unsigned int v32; // edx
  __int128 v33; // rax
  __int64 v34; // r15
  unsigned __int64 v35; // rdx
  unsigned __int64 v36; // rax
  __int64 (*v37)(void); // rdx
  unsigned __int16 v38; // ax
  __int128 v39; // rax
  __int64 *v40; // r12
  const __m128i *v41; // r15
  __int64 v42; // r8
  unsigned int v43; // ecx
  _QWORD *v44; // rax
  __int64 v45; // rdx
  _QWORD *v46; // r8
  __int64 v47; // r9
  __int64 v48; // rdx
  __int64 v49; // rdx
  __int128 v50; // [rsp-60h] [rbp-250h]
  __int128 v51; // [rsp-10h] [rbp-200h]
  _QWORD *v52; // [rsp+0h] [rbp-1F0h]
  __int64 v53; // [rsp+8h] [rbp-1E8h]
  unsigned __int64 v54; // [rsp+18h] [rbp-1D8h]
  __int128 v55; // [rsp+20h] [rbp-1D0h]
  __int64 v56; // [rsp+30h] [rbp-1C0h]
  __int64 v57; // [rsp+30h] [rbp-1C0h]
  __int128 v58; // [rsp+30h] [rbp-1C0h]
  __int64 v59; // [rsp+40h] [rbp-1B0h]
  int v60; // [rsp+40h] [rbp-1B0h]
  unsigned __int64 v61; // [rsp+48h] [rbp-1A8h]
  __int64 v62; // [rsp+50h] [rbp-1A0h]
  __int64 v63; // [rsp+78h] [rbp-178h]
  __int64 v64; // [rsp+80h] [rbp-170h] BYREF
  __int64 v65; // [rsp+88h] [rbp-168h]
  int v66; // [rsp+90h] [rbp-160h] BYREF
  __int64 v67; // [rsp+98h] [rbp-158h]
  __int64 v68; // [rsp+A0h] [rbp-150h] BYREF
  int v69; // [rsp+A8h] [rbp-148h]
  __m128i **v70; // [rsp+B0h] [rbp-140h] BYREF
  __int64 v71; // [rsp+B8h] [rbp-138h]
  _QWORD v72[38]; // [rsp+C0h] [rbp-130h] BYREF

  v6 = *a1;
  v7 = a1[1];
  if ( (*(_BYTE *)(a2 + 33) & 4) != 0 )
    return sub_3461110(a3, v6, a2, v7);
  v8 = *(_QWORD *)(a2 + 40);
  v9 = *(_QWORD *)(v7 + 64);
  v10 = *(_QWORD *)(v8 + 48);
  v61 = *(_QWORD *)(v8 + 40);
  v11 = *(_QWORD *)(v61 + 48) + 16LL * *(unsigned int *)(v8 + 48);
  v59 = v10;
  LOWORD(v10) = *(_WORD *)v11;
  v65 = *(_QWORD *)(v11 + 8);
  LOWORD(v64) = v10;
  v12 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v6 + 592LL);
  if ( v12 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v70, v6, v9, v64, v65);
    LOWORD(v13) = v71;
    LOWORD(v66) = v71;
    v67 = v72[0];
  }
  else
  {
    LODWORD(v13) = ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v12)(v6, v9, (unsigned int)v64);
    v66 = v13;
    v67 = v49;
  }
  if ( (_WORD)v13 )
  {
    v15 = (unsigned __int16)(v13 - 176) <= 0x34u;
    LODWORD(v14) = word_4456340[(unsigned __int16)v13 - 1];
    LOBYTE(v13) = v15;
  }
  else
  {
    v14 = sub_3007240((__int64)&v66);
    v13 = HIDWORD(v14);
    v15 = BYTE4(v14);
  }
  v16 = *(__int64 **)(a1[1] + 64);
  LODWORD(v70) = v14;
  BYTE4(v70) = v13;
  if ( !v15 )
  {
    LOWORD(v17) = sub_2D43050(2, v14);
    v20 = 0;
    if ( (_WORD)v17 )
      goto LABEL_8;
LABEL_36:
    v17 = sub_3009450(v16, 2, 0, (__int64)v70, v18, v19);
    HIWORD(v3) = HIWORD(v17);
    v20 = v48;
    goto LABEL_8;
  }
  LOWORD(v17) = sub_2D43AD0(2, v14);
  v20 = 0;
  if ( !(_WORD)v17 )
    goto LABEL_36;
LABEL_8:
  v21 = *a1;
  LOWORD(v3) = v17;
  v22 = 1;
  if ( ((_WORD)v66 == 1
     || (_WORD)v66 && (v22 = (unsigned __int16)v66, *(_QWORD *)(v21 + 8LL * (unsigned __int16)v66 + 112)))
    && (*(_BYTE *)(v21 + 500 * v22 + 6879) & 0xFB) == 0
    && (_WORD)v17
    && *(_QWORD *)(v21 + 8LL * (unsigned __int16)v17 + 112) )
  {
    v31 = *(_QWORD *)(a2 + 80);
    v68 = v31;
    if ( v31 )
    {
      v56 = v20;
      sub_B96E90((__int64)&v68, v31, 1);
      v20 = v56;
    }
    v57 = v20;
    v69 = *(_DWORD *)(a2 + 72);
    v62 = sub_379AB60((__int64)a1, v61, v59);
    v54 = v32 | v59 & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v33 = sub_34015B0(a1[1], (__int64)&v68, v3, v57, 0, 0, a3);
    v34 = a1[1];
    v58 = v33;
    if ( (_WORD)v64 )
    {
      LOBYTE(v35) = (unsigned __int16)(v64 - 176) <= 0x34u;
      LODWORD(v36) = word_4456340[(unsigned __int16)v64 - 1];
    }
    else
    {
      v36 = sub_3007240((__int64)&v64);
      HIDWORD(v63) = HIDWORD(v36);
      v35 = HIDWORD(v36);
    }
    BYTE4(v63) = v35;
    LODWORD(v63) = v36;
    v37 = *(__int64 (**)(void))(*(_QWORD *)*a1 + 80LL);
    v38 = 7;
    if ( v37 != sub_2FE2E20 )
      v38 = v37();
    *(_QWORD *)&v39 = sub_3401C20(v34, (__int64)&v68, v38, 0, v63, a3);
    v40 = (__int64 *)a1[1];
    v41 = *(const __m128i **)(a2 + 112);
    v55 = v39;
    v60 = (*(_WORD *)(a2 + 32) >> 7) & 7;
    *(_QWORD *)&v39 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 80LL) + 48LL)
                    + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 40) + 88LL);
    v42 = *(_QWORD *)(v39 + 8);
    v43 = *(unsigned __int16 *)v39;
    v70 = 0;
    LODWORD(v71) = 0;
    v44 = sub_33F17F0(v40, 51, (__int64)&v70, v43, v42);
    v46 = v44;
    v47 = v45;
    if ( v70 )
    {
      v52 = v44;
      v53 = v45;
      sub_B91220((__int64)&v70, (__int64)v70);
      v46 = v52;
      v47 = v53;
    }
    *((_QWORD *)&v50 + 1) = v47;
    *(_QWORD *)&v50 = v46;
    v29 = sub_33F51B0(
            v40,
            **(_QWORD **)(a2 + 40),
            *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL),
            (__int64)&v68,
            v62,
            v54,
            *(_QWORD *)(*(_QWORD *)(a2 + 40) + 80LL),
            *(_QWORD *)(*(_QWORD *)(a2 + 40) + 88LL),
            v50,
            v58,
            v55,
            v64,
            v65,
            v41,
            v60,
            0,
            0);
    if ( v68 )
      sub_B91220((__int64)&v68, v68);
  }
  else
  {
    v70 = (__m128i **)v72;
    v71 = 0x1000000000LL;
    if ( !(unsigned __int8)sub_37AC930(a1, (__int64)&v70, a2) )
      sub_C64ED0("Unable to widen vector store", 1u);
    v24 = v70;
    if ( (unsigned int)v71 == 1 )
    {
      v29 = *v70;
    }
    else
    {
      v25 = *(_QWORD *)(a2 + 80);
      v26 = (_QWORD *)a1[1];
      v27 = (unsigned int)v71;
      v28 = (__int64)v70;
      v68 = v25;
      if ( v25 )
        sub_B96E90((__int64)&v68, v25, 1);
      *((_QWORD *)&v51 + 1) = v27;
      *(_QWORD *)&v51 = v28;
      v69 = *(_DWORD *)(a2 + 72);
      v29 = (__m128i *)sub_33FC220(v26, 2, (__int64)&v68, 1, 0, v23, v51);
      if ( v68 )
        sub_B91220((__int64)&v68, v68);
      v24 = v70;
    }
    if ( v24 != v72 )
      _libc_free((unsigned __int64)v24);
  }
  return v29;
}
