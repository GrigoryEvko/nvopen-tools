// Function: sub_5FB5C0
// Address: 0x5fb5c0
//
__int64 __fastcall sub_5FB5C0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v3; // r14
  __int64 v4; // rbx
  __int64 v5; // r12
  __int64 v6; // rax
  __int64 v7; // r15
  __int64 v8; // r13
  __int64 v9; // r15
  char v10; // al
  __int64 v11; // r8
  __int64 v12; // r9
  char v13; // al
  __int64 v14; // rdx
  __m128i v15; // xmm2
  __m128i v16; // xmm4
  __m128i v17; // xmm6
  __m128i v18; // xmm0
  __m128i v19; // xmm2
  __int64 v20; // rax
  char v21; // dl
  char v22; // dl
  _QWORD *v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rsi
  __int64 v26; // rdx
  __int64 *v27; // r14
  __int64 v28; // rbx
  __int64 v29; // rdi
  __int64 v30; // rax
  __int64 v32; // r14
  __int64 v33; // r14
  char v34; // al
  char v35; // si
  __int64 v36; // rax
  __int64 v37; // rbx
  _BYTE *v38; // rax
  __int64 v39; // [rsp+0h] [rbp-5A0h]
  __int64 v40; // [rsp+8h] [rbp-598h]
  __int64 v41; // [rsp+18h] [rbp-588h]
  __int64 *v42; // [rsp+28h] [rbp-578h]
  unsigned int v43; // [rsp+30h] [rbp-570h]
  char v44; // [rsp+37h] [rbp-569h]
  __int64 v47; // [rsp+48h] [rbp-558h]
  __int64 v48; // [rsp+50h] [rbp-550h]
  __int64 v49; // [rsp+50h] [rbp-550h]
  __int16 v50; // [rsp+6Eh] [rbp-532h] BYREF
  _BYTE v51[64]; // [rsp+70h] [rbp-530h] BYREF
  __m128i v52[4]; // [rsp+B0h] [rbp-4F0h] BYREF
  __m128i v53; // [rsp+F0h] [rbp-4B0h]
  __m128i v54; // [rsp+100h] [rbp-4A0h]
  __int64 v55; // [rsp+110h] [rbp-490h]
  _QWORD v56[10]; // [rsp+120h] [rbp-480h] BYREF
  _BOOL4 v57; // [rsp+174h] [rbp-42Ch]
  _BOOL4 v58; // [rsp+178h] [rbp-428h]
  __int16 *v59; // [rsp+1D8h] [rbp-3C8h]
  _QWORD *v60; // [rsp+1E0h] [rbp-3C0h]
  __int64 v61; // [rsp+1F8h] [rbp-3A8h]
  __int64 v62; // [rsp+270h] [rbp-330h]
  __int64 v63; // [rsp+320h] [rbp-280h] BYREF
  __int64 v64; // [rsp+328h] [rbp-278h]
  char v65; // [rsp+39Fh] [rbp-201h]
  char v66; // [rsp+3A2h] [rbp-1FEh]
  char v67; // [rsp+3A3h] [rbp-1FDh]
  __int64 v68; // [rsp+440h] [rbp-160h]
  __int64 v69; // [rsp+460h] [rbp-140h]
  char v70; // [rsp+550h] [rbp-50h]

  v3 = *(__int64 **)a3;
  v4 = *(_QWORD *)(a1 + 176);
  v44 = *(_BYTE *)(a3 + 12);
  v48 = *(_QWORD *)(a2 + 24);
  v47 = **(_QWORD **)a3;
  v42 = *(__int64 **)a3;
  v5 = **(_QWORD **)(a1 + 328);
  v6 = sub_7259C0(7);
  v7 = *(_QWORD *)(v6 + 168);
  v8 = v6;
  sub_73BCD0(*(_QWORD *)(v4 + 152), v6, 0);
  *(_BYTE *)(v7 + 21) |= 1u;
  *(_QWORD *)(v7 + 8) = 0;
  *(_QWORD *)(v7 + 56) = 0;
  *(_QWORD *)(v7 + 40) = v3;
  v9 = *(_QWORD *)(*(_QWORD *)(v47 + 96) + 8LL);
  if ( !v9 )
    goto LABEL_4;
  v10 = *(_BYTE *)(v9 + 80);
  if ( v10 != 17 )
    goto LABEL_36;
  v9 = *(_QWORD *)(v9 + 88);
  if ( v9 )
  {
    while ( 1 )
    {
      v10 = *(_BYTE *)(v9 + 80);
LABEL_36:
      if ( v10 == 20 )
      {
        v32 = *(_QWORD *)(v9 + 88);
        if ( (unsigned int)sub_89B3C0(v5, **(_QWORD **)(v32 + 328), 0, 2, 0, 3) )
        {
          v33 = *(_QWORD *)(v32 + 176);
          if ( (*(_BYTE *)(v33 + 193) & 0x10) == 0
            && (unsigned int)sub_8DED30(*(_QWORD *)(v33 + 152), v8, 1114244)
            && (unsigned int)sub_739400(*(_QWORD *)(v33 + 216), *(_QWORD *)(v4 + 216)) )
          {
            break;
          }
          v34 = *(_BYTE *)(v33 + 194);
          if ( (v34 & 0x40) != 0 && ((*(_BYTE *)(v48 + 96) & 2) != 0 || *(char *)(v4 + 194) < 0) && v34 < 0 )
          {
            v35 = *(_BYTE *)(v4 + 194);
            v36 = v4;
            if ( (v35 & 0x40) != 0 )
            {
              do
                v36 = *(_QWORD *)(v36 + 232);
              while ( (v35 >= 0 || *(char *)(v36 + 194) < 0) && (*(_BYTE *)(v36 + 194) & 0x40) != 0 );
            }
            do
              v33 = *(_QWORD *)(v33 + 232);
            while ( *(char *)(v33 + 194) < 0 && (*(_BYTE *)(v33 + 194) & 0x40) != 0 );
            if ( v33 == v36 )
              break;
          }
        }
      }
      v9 = *(_QWORD *)(v9 + 8);
      if ( !v9 )
        goto LABEL_4;
    }
  }
  else
  {
LABEL_4:
    v50 = 75;
    v41 = unk_4D03B88;
    v43 = dword_4F04C3C;
    *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 7) |= 0x10u;
    dword_4F04C3C = 1;
    sub_5E4C60((__int64)&v63, (_QWORD *)(a2 + 8));
    v70 |= 2u;
    v66 |= 0x80u;
    v69 = v4;
    if ( (*(_BYTE *)(v48 + 96) & 2) != 0 || *(char *)(v4 + 194) < 0 )
      v67 |= 1u;
    v65 |= 0x10u;
    v68 = v8;
    v13 = *(_BYTE *)(v4 + 193);
    if ( (v13 & 2) != 0 )
      v64 |= 0x80000uLL;
    if ( (v13 & 4) != 0 )
      v64 |= 0x100000uLL;
    v39 = v12;
    v40 = v11;
    sub_87E3B0(v52);
    v14 = *(_QWORD *)(a1 + 280);
    v15 = _mm_loadu_si128((const __m128i *)(a1 + 200));
    v16 = _mm_loadu_si128((const __m128i *)(a1 + 216));
    v17 = _mm_loadu_si128((const __m128i *)(a1 + 232));
    v52[0] = _mm_loadu_si128((const __m128i *)(a1 + 184));
    v18 = _mm_loadu_si128((const __m128i *)(a1 + 248));
    v52[1] = v15;
    v19 = _mm_loadu_si128((const __m128i *)(a1 + 264));
    v53 = v18;
    v52[2] = v16;
    v52[3] = v17;
    v54 = v19;
    v53.m128i_i8[0] = v18.m128i_i8[0] | 2;
    v55 = v14;
    sub_878710(v47, v51);
    sub_87A680(v51, v40, 0);
    sub_89EF00(v56, v39);
    v20 = v56[0];
    *(_BYTE *)(v56[0] + 130LL) |= 0x80u;
    v21 = *(_BYTE *)(v20 + 131);
    *(_QWORD *)(v20 + 320) = v4;
    *(_BYTE *)(v20 + 131) = v67 & 1 | v21 & 0xFE;
    v59 = &v50;
    v22 = *(_BYTE *)(a1 + 160);
    v58 = (v22 & 0x10) != 0;
    v57 = (v22 & 8) != 0;
    sub_85FC20(v5, 0);
    sub_860400(v60, 0);
    v23 = (_QWORD *)sub_89EE30(v5);
    *v60 = v23;
    v60[9] = *(_QWORD *)(*(_QWORD *)(a1 + 328) + 72LL);
    ++v61;
    v24 = v62;
    *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 616) = v56;
    *(_BYTE *)(a3 + 12) = *(_BYTE *)(v4 + 88) & 3;
    sub_5FA450((__int64)v51, v23, v24, v52, a3, v39);
    v25 = v63;
    v49 = *(_QWORD *)(v63 + 88);
    sub_897580(v56, v63, v49);
    v26 = v49;
    v27 = *(__int64 **)(v49 + 176);
    *((_DWORD *)v27 + 48) |= 0x401000u;
    v27[45] = a2;
    if ( *(char *)(v4 + 193) < 0 )
      *((_BYTE *)v27 + 193) |= 0x80u;
    if ( (*(_BYTE *)(v4 + 194) & 0x10) != 0 )
    {
      *((_BYTE *)v27 + 194) |= 0x10u;
      *(_BYTE *)(v42[21] + 110) |= 2u;
    }
    if ( dword_4D048B8 )
    {
      v37 = *(_QWORD *)(v8 + 168);
      v38 = (_BYTE *)sub_725E60(v56, v25, v49);
      v26 = v49;
      *v38 |= 0xAu;
      *(_QWORD *)(v37 + 56) = v38;
      *(_QWORD *)(v37 + 8) = v27;
    }
    *(_QWORD *)(v26 + 328) = v60;
    unk_4D03B88 = 0;
    sub_89F220(v56, v52, v63);
    v28 = *v27;
    v29 = sub_5E4B20((__int64)v42);
    *(_QWORD *)(v29 + 16) = v28;
    *(_BYTE *)(v29 + 184) = *(_BYTE *)(v29 + 184) & 0xCF | 0x10;
    sub_5E9580(v29);
    sub_863FC0();
    if ( dword_4F04C64 == -1
      || (v30 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v30 + 7) & 1) == 0)
      || dword_4F04C44 == -1 && (*(_BYTE *)(v30 + 6) & 2) == 0 )
    {
      if ( (v53.m128i_i8[1] & 8) == 0 )
        sub_87E280(&v52[0].m128i_u64[1]);
    }
    sub_85FB90(v5);
    dword_4F04C3C = v43;
    *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 7) = *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 7)
                                                             & 0xEF
                                                             | (16 * (v43 & 1));
    unk_4D03B88 = v41;
  }
  *(_BYTE *)(a3 + 12) = v44;
  return a3;
}
