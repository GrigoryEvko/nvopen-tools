// Function: sub_3313C70
// Address: 0x3313c70
//
__int64 __fastcall sub_3313C70(__int64 a1, __int64 a2)
{
  const __m128i *v4; // rax
  __int64 v5; // rsi
  __int64 v6; // rbx
  __int64 v7; // r13
  __int64 v8; // r14
  __int64 v9; // r13
  __int16 v11; // ax
  __int64 v12; // rsi
  int v13; // ebx
  __int64 v14; // rax
  __int16 v15; // dx
  __int64 v16; // rax
  __int64 v17; // rax
  unsigned int v18; // ebx
  unsigned __int64 *v19; // r11
  unsigned __int64 v20; // rax
  __int64 v21; // r9
  __int64 v22; // r8
  __int64 v23; // rbx
  _QWORD *v24; // rax
  __int64 v25; // rsi
  __int64 v26; // r12
  __int64 *v27; // r13
  _QWORD *v28; // r8
  __int16 v29; // cx
  char v30; // r14
  __int64 v31; // rax
  unsigned int *v32; // r8
  _DWORD *v33; // r13
  __int64 v34; // rdi
  __int64 v35; // r9
  unsigned __int16 *v36; // rsi
  __int64 v37; // rdx
  bool (__fastcall *v38)(__int64, unsigned __int16, __int64, unsigned __int16, __int64, bool); // rax
  __int64 v39; // rbx
  __int64 v40; // rax
  __int64 v41; // rax
  int v42; // edx
  int v43; // r9d
  unsigned int v44; // eax
  __int64 v45; // rdx
  __int64 v46; // r10
  __int64 v47; // rbx
  __int64 v48; // r12
  __int64 v49; // r13
  __int64 v50; // rcx
  _QWORD *v51; // rax
  __int16 v52; // ax
  __int64 v53; // rax
  __int64 v54; // rax
  __int128 v55; // rax
  unsigned __int64 v56; // rax
  __int64 v57; // rdx
  __int64 v58; // r8
  __int64 v59; // r9
  __int64 v60; // rax
  __int64 v61; // rax
  __int64 v62; // rdx
  __int64 v63; // rdx
  __int64 v64; // [rsp+8h] [rbp-138h]
  __m128i v65; // [rsp+10h] [rbp-130h]
  __int32 v66; // [rsp+20h] [rbp-120h]
  unsigned __int16 v67; // [rsp+20h] [rbp-120h]
  __int64 v68; // [rsp+20h] [rbp-120h]
  __int32 v69; // [rsp+28h] [rbp-118h]
  _QWORD *v70; // [rsp+28h] [rbp-118h]
  __int64 v71; // [rsp+30h] [rbp-110h]
  __int64 v72; // [rsp+30h] [rbp-110h]
  int v73; // [rsp+30h] [rbp-110h]
  __int128 v74; // [rsp+30h] [rbp-110h]
  __int64 v75; // [rsp+40h] [rbp-100h]
  __m128i v76; // [rsp+40h] [rbp-100h]
  int v77; // [rsp+40h] [rbp-100h]
  __int64 v78; // [rsp+50h] [rbp-F0h]
  __int64 (__fastcall *v79)(_DWORD *, __int64, __int64, _QWORD); // [rsp+50h] [rbp-F0h]
  int v80; // [rsp+50h] [rbp-F0h]
  int v81; // [rsp+50h] [rbp-F0h]
  unsigned int v82; // [rsp+58h] [rbp-E8h]
  _QWORD *v83; // [rsp+58h] [rbp-E8h]
  __int64 v84; // [rsp+60h] [rbp-E0h]
  __int16 v85; // [rsp+60h] [rbp-E0h]
  __int64 v86; // [rsp+68h] [rbp-D8h]
  __int64 v87; // [rsp+70h] [rbp-D0h] BYREF
  int v88; // [rsp+78h] [rbp-C8h]
  unsigned __int16 v89; // [rsp+80h] [rbp-C0h] BYREF
  __int64 v90; // [rsp+88h] [rbp-B8h]
  unsigned __int64 v91; // [rsp+90h] [rbp-B0h]
  __int64 v92; // [rsp+98h] [rbp-A8h]
  unsigned __int16 v93; // [rsp+A0h] [rbp-A0h] BYREF
  __int64 v94; // [rsp+A8h] [rbp-98h]
  unsigned __int64 v95; // [rsp+B0h] [rbp-90h]
  __int64 v96; // [rsp+B8h] [rbp-88h]
  unsigned __int16 v97; // [rsp+C0h] [rbp-80h] BYREF
  __int64 v98; // [rsp+C8h] [rbp-78h]
  unsigned __int64 v99; // [rsp+D0h] [rbp-70h]
  __int64 v100; // [rsp+D8h] [rbp-68h]
  unsigned __int64 v101; // [rsp+E0h] [rbp-60h] BYREF
  __int64 v102; // [rsp+E8h] [rbp-58h]
  _OWORD v103[5]; // [rsp+F0h] [rbp-50h] BYREF

  v4 = *(const __m128i **)(a2 + 40);
  v5 = *(_QWORD *)(a2 + 80);
  v6 = v4[10].m128i_i64[0];
  v66 = v4[10].m128i_i32[2];
  v84 = v4->m128i_i64[1];
  v86 = v4->m128i_i64[0];
  v7 = v4->m128i_i64[0];
  v71 = v4[3].m128i_i64[0];
  v75 = v4[2].m128i_i64[1];
  v8 = v75;
  v78 = v4[5].m128i_i64[0];
  v82 = v4[3].m128i_u32[0];
  v69 = v4[5].m128i_i32[2];
  v87 = v5;
  v65 = _mm_loadu_si128(v4 + 5);
  if ( v5 )
    sub_B96E90((__int64)&v87, v5, 1);
  v88 = *(_DWORD *)(a2 + 72);
  if ( (unsigned __int8)sub_33D1AE0(v6, 0) )
  {
    v9 = v86;
    goto LABEL_5;
  }
  if ( *(_DWORD *)(v7 + 24) == 363 )
  {
    v11 = *(_WORD *)(a2 + 32);
    if ( (v11 & 0x380) == 0 && (*(_BYTE *)(*(_QWORD *)(a2 + 112) + 37LL) & 0xF) == 0 && (v11 & 8) == 0 )
    {
      v52 = *(_WORD *)(v7 + 32);
      if ( (v52 & 0x380) == 0 && (*(_BYTE *)(*(_QWORD *)(v7 + 112) + 37LL) & 0xF) == 0 && (v52 & 8) == 0 )
      {
        v53 = *(_QWORD *)(v7 + 40);
        if ( v78 == *(_QWORD *)(v53 + 80)
          && v69 == *(_DWORD *)(v53 + 88)
          && *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 80LL) + 24LL) != 51 )
        {
          if ( *(_QWORD *)(v53 + 160) == v6 && v66 == *(_DWORD *)(v53 + 168) )
          {
            v60 = *(_QWORD *)(v7 + 104);
            v93 = *(_WORD *)(v7 + 96);
            v94 = v60;
            v95 = sub_3285A00(&v93);
            v61 = *(_QWORD *)(a2 + 104);
            v96 = v62;
            LOWORD(v62) = *(_WORD *)(a2 + 96);
            v90 = v61;
            v89 = v62;
            v91 = sub_3285A00(&v89);
            v92 = v63;
            if ( v95 == v91 && (_BYTE)v96 == (_BYTE)v92 )
              goto LABEL_70;
          }
          if ( (unsigned __int8)sub_33D1720(v6, 0) )
          {
LABEL_70:
            v54 = *(_QWORD *)(a2 + 104);
            LOWORD(v101) = *(_WORD *)(a2 + 96);
            v102 = v54;
            *(_QWORD *)&v55 = sub_3285A00((unsigned __int16 *)&v101);
            v103[0] = v55;
            WORD4(v55) = *(_WORD *)(v7 + 96);
            v98 = *(_QWORD *)(v7 + 104);
            v97 = WORD4(v55);
            v56 = sub_3285A00(&v97);
            v100 = v57;
            v99 = v56;
            if ( (!(_BYTE)v57 || BYTE8(v103[0])) && *(_QWORD *)&v103[0] >= v99 )
            {
              v103[0] = _mm_loadu_si128((const __m128i *)*(_QWORD *)(v7 + 40));
              sub_32EB790(a1, v7, (__int64 *)v103, 1, 1);
              if ( *(_DWORD *)(a2 + 24) )
                sub_32B3E80(a1, a2, 1, 0, v58, v59);
              goto LABEL_18;
            }
          }
        }
      }
    }
  }
  v12 = 0;
  if ( (unsigned __int8)sub_33D1720(v6, 0) && (*(_WORD *)(a2 + 32) & 0x380) == 0 && (*(_BYTE *)(a2 + 33) & 0xC) == 0 )
  {
    v23 = *(_QWORD *)(a2 + 112);
    v24 = *(_QWORD **)(a2 + 40);
    v25 = *(_QWORD *)(a2 + 80);
    v26 = *(_QWORD *)a1;
    v27 = v24 + 10;
    v28 = v24 + 5;
    v103[0] = _mm_loadu_si128((const __m128i *)(v23 + 40));
    v103[1] = _mm_loadu_si128((const __m128i *)(v23 + 56));
    v29 = *(_WORD *)(v23 + 32);
    v30 = *(_BYTE *)(v23 + 34);
    v101 = v25;
    if ( v25 )
    {
      v83 = v24 + 5;
      v85 = v29;
      sub_B96E90((__int64)&v101, v25, 1);
      v24 = *(_QWORD **)(a2 + 40);
      v28 = v83;
      v29 = v85;
    }
    LODWORD(v102) = *(_DWORD *)(a2 + 72);
    v9 = sub_33F4560(
           v26,
           *v24,
           v24[1],
           (unsigned int)&v101,
           *v28,
           v28[1],
           *v27,
           v27[1],
           *(_OWORD *)v23,
           *(_QWORD *)(v23 + 16),
           v30,
           v29,
           (__int64)v103);
    if ( v101 )
      sub_B91220((__int64)&v101, v101);
    goto LABEL_5;
  }
  if ( *(int *)(a1 + 24) > 2 )
  {
    v12 = a2;
    if ( (unsigned __int8)sub_3312A90((__int64 *)a1, a2)
      || *(int *)(a1 + 24) > 2 && (v12 = a2, (unsigned __int8)sub_3312210((__int64 *)a1, a2)) )
    {
LABEL_18:
      v9 = a2;
      goto LABEL_5;
    }
  }
  v13 = *(_DWORD *)(v75 + 24);
  if ( (*(_BYTE *)(a2 + 33) & 4) == 0 || (*(_WORD *)(a2 + 32) & 0x380) != 0 )
    goto LABEL_38;
  v14 = *(_QWORD *)(v75 + 48) + 16LL * v82;
  v15 = *(_WORD *)v14;
  v16 = *(_QWORD *)(v14 + 8);
  LOWORD(v103[0]) = v15;
  *((_QWORD *)&v103[0] + 1) = v16;
  if ( v15 )
  {
    if ( (unsigned __int16)(v15 - 2) > 7u
      && (unsigned __int16)(v15 - 17) > 0x6Cu
      && (unsigned __int16)(v15 - 176) > 0x1Fu )
    {
      goto LABEL_38;
    }
  }
  else if ( !sub_3007070((__int64)v103) )
  {
    goto LABEL_38;
  }
  if ( (v13 == 11 || v13 == 35) && (*(_BYTE *)(v75 + 32) & 8) != 0 )
  {
LABEL_38:
    if ( v13 == 216 )
    {
      v31 = *(_QWORD *)(v75 + 56);
      if ( v31 )
      {
        if ( !*(_QWORD *)(v31 + 32) && (*(_WORD *)(a2 + 32) & 0x380) == 0 && (*(_BYTE *)(a2 + 33) & 8) == 0 )
        {
          v32 = *(unsigned int **)(v75 + 40);
          v33 = *(_DWORD **)(a1 + 8);
          v34 = *(unsigned __int16 *)(a2 + 96);
          v35 = *(unsigned __int8 *)(a1 + 33);
          v36 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)v32 + 48LL) + 16LL * v32[2]);
          v37 = *v36;
          v38 = *(bool (__fastcall **)(__int64, unsigned __int16, __int64, unsigned __int16, __int64, bool))(*(_QWORD *)v33 + 688LL);
          if ( v38 == sub_2FE31C0 )
          {
            if ( (_BYTE)v35 )
            {
              if ( (_WORD)v37
                && *(_QWORD *)&v33[2 * v37 + 28]
                && (_WORD)v34
                && !*((_BYTE *)v33 + 274 * (unsigned __int16)v37 + v34 + 443718) )
              {
                goto LABEL_55;
              }
            }
            else if ( (_WORD)v37
                   && *(_QWORD *)&v33[2 * v37 + 28]
                   && (_WORD)v34
                   && (*((_BYTE *)v33 + 274 * (unsigned __int16)v37 + v34 + 443718) & 0xFB) == 0 )
            {
LABEL_55:
              v39 = *(_QWORD *)a1;
              v40 = *(_QWORD *)(*(_QWORD *)v32 + 48LL) + 16LL * v32[2];
              v67 = *(_WORD *)v40;
              v64 = *(_QWORD *)(v40 + 8);
              v76 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a2 + 40) + 160LL));
              sub_3285E70((__int64)&v101, v76.m128i_i64[0]);
              v72 = *(_QWORD *)(v39 + 64);
              v79 = *(__int64 (__fastcall **)(_DWORD *, __int64, __int64, _QWORD))(*(_QWORD *)v33 + 528LL);
              v41 = sub_2E79000(*(__int64 **)(v39 + 40));
              v80 = v79(v33, v41, v72, v67);
              v73 = v42;
              LOWORD(v103[0]) = v67;
              *((_QWORD *)&v103[0] + 1) = v64;
              if ( sub_32801E0((__int64)v103) )
              {
                v44 = v33[17];
              }
              else if ( (unsigned __int8)sub_3280140((__int64)v103) )
              {
                v44 = v33[16];
              }
              else
              {
                v44 = v33[15];
              }
              if ( v44 > 2 )
                BUG();
              *(_QWORD *)&v74 = sub_33FAF80(v39, 215 - v44, (unsigned int)&v101, v80, v73, v43, *(_OWORD *)&v76);
              *((_QWORD *)&v74 + 1) = v45;
              sub_9C6650(&v101);
              v46 = *(_QWORD *)a1;
              v47 = *(_QWORD *)(a2 + 112);
              v48 = *(unsigned __int16 *)(a2 + 96);
              v49 = *(_QWORD *)(a2 + 104);
              v50 = *(_QWORD *)(a2 + 40);
              v77 = (*(_WORD *)(a2 + 32) >> 7) & 7;
              v51 = *(_QWORD **)(v8 + 40);
              *(_QWORD *)&v103[0] = *(_QWORD *)(a2 + 80);
              if ( *(_QWORD *)&v103[0] )
              {
                v68 = v50;
                v70 = v51;
                v81 = v46;
                sub_325F5D0((__int64 *)v103);
                v50 = v68;
                v51 = v70;
                LODWORD(v46) = v81;
              }
              DWORD2(v103[0]) = *(_DWORD *)(a2 + 72);
              v9 = sub_33F65D0(
                     v46,
                     v86,
                     v84,
                     (unsigned int)v103,
                     *v51,
                     v51[1],
                     v65.m128i_i64[0],
                     v65.m128i_i64[1],
                     *(_OWORD *)(v50 + 120),
                     v74,
                     v48,
                     v49,
                     v47,
                     v77,
                     1,
                     0);
              sub_9C6650(v103);
              goto LABEL_5;
            }
          }
          else if ( v38(*(_QWORD *)(a1 + 8), v37, *((_QWORD *)v36 + 1), *(_WORD *)(a2 + 96), *(_QWORD *)(a2 + 104), v35) )
          {
            v33 = *(_DWORD **)(a1 + 8);
            v32 = *(unsigned int **)(v75 + 40);
            goto LABEL_55;
          }
        }
      }
    }
    v9 = 0;
    goto LABEL_5;
  }
  v17 = *(_QWORD *)(a2 + 104);
  LOWORD(v103[0]) = *(_WORD *)(a2 + 96);
  *((_QWORD *)&v103[0] + 1) = v17;
  v18 = sub_32844A0((unsigned __int16 *)v103, v12);
  LODWORD(v102) = sub_3263630(v75, v82);
  if ( (unsigned int)v102 > 0x40 )
    sub_C43690((__int64)&v101, 0, 0);
  else
    v101 = 0;
  v19 = &v101;
  if ( v18 )
  {
    if ( v18 > 0x40 )
    {
      sub_C43C90(&v101, 0, v18);
      LODWORD(v19) = (unsigned int)&v101;
    }
    else
    {
      v20 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v18);
      if ( (unsigned int)v102 > 0x40 )
        *(_QWORD *)v101 |= v20;
      else
        v101 |= v20;
    }
  }
  if ( !(unsigned __int8)sub_32D08B0(a1, v75, v71, (int)v19) )
  {
    if ( (unsigned int)v102 > 0x40 && v101 )
      j_j___libc_free_0_0(v101);
    v13 = *(_DWORD *)(v75 + 24);
    goto LABEL_38;
  }
  v22 = *(unsigned int *)(a2 + 24);
  if ( (_DWORD)v22 )
    sub_32B3E80(a1, a2, 1, 0, v22, v21);
  v9 = a2;
  if ( (unsigned int)v102 > 0x40 && v101 )
    j_j___libc_free_0_0(v101);
LABEL_5:
  if ( v87 )
    sub_B91220((__int64)&v87, v87);
  return v9;
}
