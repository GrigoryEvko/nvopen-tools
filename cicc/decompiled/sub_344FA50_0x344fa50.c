// Function: sub_344FA50
// Address: 0x344fa50
//
unsigned __int8 *__fastcall sub_344FA50(__int64 a1, __int64 a2, char a3, _QWORD *a4)
{
  unsigned __int16 *v6; // rdx
  int v7; // eax
  __int64 v8; // rbx
  __int64 v9; // rdx
  __int64 v10; // rdx
  __int64 v11; // rsi
  __int64 v12; // rax
  __int128 v13; // xmm0
  __int128 v14; // xmm1
  __int64 v15; // r14
  unsigned int v16; // ebx
  unsigned __int16 *v17; // rax
  __int64 v18; // r14
  unsigned int v19; // ebx
  __int64 v20; // rdx
  __int64 v21; // r9
  __int64 v22; // rdx
  __int64 v23; // rdx
  unsigned int v24; // r15d
  __int128 v25; // rax
  __int128 v26; // rax
  __int64 v27; // r9
  __int128 v28; // rax
  __int64 v29; // r9
  unsigned int v30; // edx
  __int64 v31; // r9
  __int128 v32; // rax
  __int128 v33; // rax
  __int64 v34; // r9
  __int128 v35; // rax
  __int64 v36; // r9
  __int64 v37; // r9
  unsigned __int8 *v38; // r10
  unsigned int v39; // edx
  __int64 v40; // r11
  unsigned __int8 *v41; // rax
  unsigned __int8 *v42; // r14
  __int64 v44; // rcx
  _BYTE *v45; // rax
  __int128 v46; // rax
  __int64 v47; // rdx
  __int64 v48; // rcx
  __int64 v49; // r8
  __int128 v50; // rax
  __int64 v51; // r9
  __int128 v52; // rax
  __int64 v53; // r9
  unsigned int v54; // edx
  __int128 v55; // rax
  __int64 v56; // r9
  unsigned int v57; // edx
  unsigned __int16 v58; // r15
  __int128 v59; // [rsp-10h] [rbp-130h]
  __int128 v60; // [rsp+0h] [rbp-120h]
  unsigned int v61; // [rsp+28h] [rbp-F8h]
  __int128 v63; // [rsp+30h] [rbp-F0h]
  __int64 v64; // [rsp+30h] [rbp-F0h]
  __int64 v65; // [rsp+50h] [rbp-D0h]
  __int128 v66; // [rsp+60h] [rbp-C0h]
  __int128 v67; // [rsp+60h] [rbp-C0h]
  __int128 v68; // [rsp+60h] [rbp-C0h]
  int v69; // [rsp+70h] [rbp-B0h]
  __int128 v70; // [rsp+70h] [rbp-B0h]
  unsigned int v71; // [rsp+C0h] [rbp-60h] BYREF
  __int64 v72; // [rsp+C8h] [rbp-58h]
  __int64 v73; // [rsp+D0h] [rbp-50h] BYREF
  __int64 v74; // [rsp+D8h] [rbp-48h]
  __int64 v75; // [rsp+E0h] [rbp-40h]
  __int64 v76; // [rsp+E8h] [rbp-38h]

  v6 = *(unsigned __int16 **)(a2 + 48);
  v7 = *v6;
  v8 = *((_QWORD *)v6 + 1);
  LOWORD(v71) = v7;
  v72 = v8;
  if ( (_WORD)v7 )
  {
    if ( (unsigned __int16)(v7 - 17) > 0xD3u )
    {
      LOWORD(v73) = v7;
      v74 = v8;
      goto LABEL_4;
    }
    LOWORD(v7) = word_4456580[v7 - 1];
    v9 = 0;
  }
  else
  {
    if ( !sub_30070B0((__int64)&v71) )
    {
      v74 = v8;
      LOWORD(v73) = 0;
      goto LABEL_9;
    }
    LOWORD(v7) = sub_3009970((__int64)&v71, a2, v47, v48, v49);
  }
  LOWORD(v73) = v7;
  v74 = v9;
  if ( !(_WORD)v7 )
  {
LABEL_9:
    v75 = sub_3007260((__int64)&v73);
    v76 = v10;
    LODWORD(v65) = v75;
    goto LABEL_10;
  }
LABEL_4:
  if ( (_WORD)v7 == 1 || (unsigned __int16)(v7 - 504) <= 7u )
    BUG();
  v65 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v7 - 16];
LABEL_10:
  v11 = *(_QWORD *)(a2 + 80);
  v69 = *(_DWORD *)(a2 + 24);
  v12 = *(_QWORD *)(a2 + 40);
  v13 = (__int128)_mm_loadu_si128((const __m128i *)v12);
  v14 = (__int128)_mm_loadu_si128((const __m128i *)(v12 + 40));
  v15 = *(_QWORD *)(v12 + 40);
  v16 = *(_DWORD *)(v12 + 48);
  v73 = v11;
  if ( v11 )
    sub_B96E90((__int64)&v73, v11, 1);
  LODWORD(v74) = *(_DWORD *)(a2 + 72);
  v17 = (unsigned __int16 *)(*(_QWORD *)(v15 + 48) + 16LL * v16);
  v18 = *((_QWORD *)v17 + 1);
  v19 = *v17;
  *(_QWORD *)&v60 = sub_3400BD0((__int64)a4, 0, (__int64)&v73, *v17, v18, 0, (__m128i)v13, 0);
  *((_QWORD *)&v60 + 1) = v20;
  v21 = (unsigned int)(v69 == 193) + 193;
  if ( (_WORD)v71 != 1 )
  {
    if ( !(_WORD)v71 )
    {
      if ( a3 || !sub_30070B0((__int64)&v71) )
        goto LABEL_20;
      goto LABEL_34;
    }
    if ( !*(_QWORD *)(a1 + 8LL * (unsigned __int16)v71 + 112) )
      goto LABEL_28;
  }
  v22 = *(unsigned int *)(a2 + 24);
  if ( (unsigned int)v22 > 0x1F3 )
    goto LABEL_19;
  if ( (*(_BYTE *)(v22 + a1 + 500LL * (unsigned __int16)v71 + 6414) & 0xFB) == 0 )
    goto LABEL_28;
  v23 = 1;
  if ( (_WORD)v71 != 1 )
  {
    if ( *(_QWORD *)(a1 + 8LL * (unsigned __int16)v71 + 112) )
    {
      v23 = (unsigned __int16)v71;
      goto LABEL_18;
    }
LABEL_28:
    if ( a3 )
    {
LABEL_20:
      v61 = 2 * (v69 != 193) + 190;
      v24 = 2 * (v69 == 193) + 190;
      *(_QWORD *)&v25 = sub_3400BD0((__int64)a4, (unsigned int)(v65 - 1), (__int64)&v73, v19, v18, 0, (__m128i)v13, 0);
      v63 = v25;
      if ( (_DWORD)v65 && ((unsigned int)v65 & ((_DWORD)v65 - 1)) == 0 )
      {
        *(_QWORD *)&v50 = sub_3406EB0(a4, 0x39u, (__int64)&v73, v19, v18, 0xFFFFFFFF00000000LL, v60, v14);
        v68 = v50;
        *(_QWORD *)&v52 = sub_3406EB0(a4, 0xBAu, (__int64)&v73, v19, v18, v51, v14, v63);
        *(_QWORD *)&v70 = sub_3406EB0(a4, v61, (__int64)&v73, v71, v72, v53, v13, v52);
        *((_QWORD *)&v70 + 1) = v54;
        *(_QWORD *)&v55 = sub_3406EB0(a4, 0xBAu, (__int64)&v73, v19, v18, 0xFFFFFFFF00000000LL, v68, v63);
        v38 = sub_3406EB0(a4, v24, (__int64)&v73, v71, v72, v56, v13, v55);
        v37 = v57;
        v40 = v57;
      }
      else
      {
        *(_QWORD *)&v26 = sub_3400BD0((__int64)a4, (unsigned int)v65, (__int64)&v73, v19, v18, 0, (__m128i)v13, 0);
        *(_QWORD *)&v28 = sub_3406EB0(a4, 0x3Eu, (__int64)&v73, v19, v18, v27, v14, v26);
        v66 = v28;
        *(_QWORD *)&v70 = sub_3406EB0(a4, v61, (__int64)&v73, v71, v72, v29, v13, v28);
        *((_QWORD *)&v70 + 1) = v30;
        *(_QWORD *)&v32 = sub_3406EB0(a4, 0x39u, (__int64)&v73, v19, v18, v31, v63, v66);
        v67 = v32;
        *(_QWORD *)&v33 = sub_3400BD0((__int64)a4, 1, (__int64)&v73, v19, v18, 0, (__m128i)v13, 0);
        *(_QWORD *)&v35 = sub_3406EB0(a4, v24, (__int64)&v73, v71, v72, v34, v13, v33);
        v38 = sub_3406EB0(a4, v24, (__int64)&v73, v71, v72, v36, v35, v67);
        v40 = v39;
      }
      *((_QWORD *)&v59 + 1) = v40;
      *(_QWORD *)&v59 = v38;
      v41 = sub_3406EB0(a4, 0xBBu, (__int64)&v73, v71, v72, v37, v70, v59);
      goto LABEL_24;
    }
    goto LABEL_29;
  }
LABEL_18:
  if ( (*(_BYTE *)((unsigned int)v21 + a1 + 500 * v23 + 6414) & 0xFB) == 0
    && (_DWORD)v65
    && (((_DWORD)v65 - 1) & (unsigned int)v65) == 0 )
  {
    *(_QWORD *)&v46 = sub_3406EB0(a4, 0x39u, (__int64)&v73, v19, v18, v21, v60, v14);
    v41 = sub_3406EB0(
            a4,
            (unsigned int)(v69 == 193) + 193,
            (__int64)&v73,
            v71,
            v72,
            (unsigned int)(v69 == 193) + 193,
            v13,
            v46);
LABEL_24:
    v42 = v41;
    goto LABEL_25;
  }
LABEL_19:
  if ( a3 )
    goto LABEL_20;
LABEL_29:
  if ( (unsigned __int16)(v71 - 17) > 0xD3u )
    goto LABEL_20;
  v44 = (unsigned __int16)v71 + 14LL;
  if ( *(_QWORD *)(a1 + 8LL * (unsigned __int16)v71 + 112) )
  {
    v45 = (_BYTE *)(a1 + 500LL * (unsigned __int16)v71);
    if ( (v45[6604] & 0xFB) == 0 )
    {
      if ( *(_QWORD *)(a1 + 8 * v44) )
      {
        if ( (v45[6606] & 0xFB) == 0 )
        {
          if ( *(_QWORD *)(a1 + 8 * v44) )
          {
            if ( (v45[6471] & 0xFB) == 0 )
            {
              v58 = v71;
              v64 = v72;
              if ( (unsigned __int8)sub_328C7F0(a1, 0xBBu, v71, v72, 0) )
              {
                if ( (unsigned __int8)sub_328C7F0(a1, 0xBAu, v58, v64, 0) )
                  goto LABEL_20;
              }
            }
          }
        }
      }
    }
  }
LABEL_34:
  v42 = 0;
LABEL_25:
  if ( v73 )
    sub_B91220((__int64)&v73, v73);
  return v42;
}
