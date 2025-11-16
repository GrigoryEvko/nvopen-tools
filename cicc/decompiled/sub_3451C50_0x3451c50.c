// Function: sub_3451C50
// Address: 0x3451c50
//
__int64 __fastcall sub_3451C50(__int64 a1, __int64 a2, __int64 a3, __m128i a4, __int64 a5, _QWORD *a6, int a7)
{
  int v9; // eax
  unsigned int v10; // r8d
  _QWORD *v11; // rcx
  __int64 v12; // r14
  __int64 v13; // r15
  unsigned __int16 *v14; // rax
  int v15; // edx
  unsigned __int16 *v16; // rax
  __int64 v17; // rsi
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rsi
  unsigned __int8 *v21; // rax
  __int64 v22; // rsi
  int v23; // edx
  __int16 v25; // ax
  __int64 v26; // rdx
  __int64 v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // r8
  __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // r8
  __int64 v33; // rsi
  __int128 v34; // rax
  __int128 v35; // rax
  __int128 v36; // rax
  __int128 v37; // rax
  __int128 v38; // rax
  __int64 v39; // r9
  __int128 v40; // rax
  __int64 v41; // r9
  __int128 v42; // rax
  __int64 v43; // r9
  unsigned __int8 *v44; // rax
  unsigned int v45; // edx
  unsigned int v46; // r14d
  __int64 v47; // r15
  __int64 v48; // r9
  unsigned __int8 *v49; // rax
  unsigned int v50; // edx
  unsigned int v51; // ebx
  unsigned __int8 *v52; // rax
  __int64 v53; // rdx
  __int64 v54; // r15
  unsigned __int8 *v55; // r14
  __int128 v56; // rax
  __int64 v57; // r9
  __int128 v58; // rax
  __int64 v59; // r9
  unsigned __int8 *v60; // rax
  __int64 v61; // rsi
  int v62; // edx
  __int128 v63; // [rsp-20h] [rbp-110h]
  __int128 v64; // [rsp-10h] [rbp-100h]
  __int128 v65; // [rsp-10h] [rbp-100h]
  __int128 v66; // [rsp+10h] [rbp-E0h]
  __int128 v67; // [rsp+10h] [rbp-E0h]
  __int128 v68; // [rsp+20h] [rbp-D0h]
  __int128 v69; // [rsp+20h] [rbp-D0h]
  __int128 v70; // [rsp+30h] [rbp-C0h]
  unsigned __int16 v71; // [rsp+40h] [rbp-B0h]
  __int128 v72; // [rsp+40h] [rbp-B0h]
  __int64 v73; // [rsp+50h] [rbp-A0h]
  __int128 v74; // [rsp+50h] [rbp-A0h]
  unsigned int v76; // [rsp+90h] [rbp-60h] BYREF
  __int64 v77; // [rsp+98h] [rbp-58h]
  unsigned int v78; // [rsp+A0h] [rbp-50h] BYREF
  __int64 v79; // [rsp+A8h] [rbp-48h]
  __int64 v80; // [rsp+B0h] [rbp-40h] BYREF
  int v81; // [rsp+B8h] [rbp-38h]

  v9 = *(_DWORD *)(a2 + 24);
  if ( v9 > 239 )
  {
    v10 = 0;
    if ( (unsigned int)(v9 - 242) <= 1 )
      return v10;
  }
  else
  {
    v10 = 0;
    if ( v9 > 237 || (unsigned int)(v9 - 101) <= 0x2F )
      return v10;
  }
  v11 = *(_QWORD **)(a2 + 40);
  v12 = *v11;
  v13 = v11[1];
  v14 = (unsigned __int16 *)(*(_QWORD *)(*v11 + 48LL) + 16LL * *((unsigned int *)v11 + 2));
  v15 = *v14;
  v77 = *((_QWORD *)v14 + 1);
  v16 = *(unsigned __int16 **)(a2 + 48);
  LOWORD(v76) = v15;
  v17 = *v16;
  v18 = *((_QWORD *)v16 + 1);
  LOWORD(v78) = v17;
  v79 = v18;
  if ( (*(_BYTE *)(a2 + 28) & 0x10) != 0 )
  {
    v19 = 1;
    if ( (_WORD)v15 == 1 )
      goto LABEL_6;
    if ( (_WORD)v15 )
    {
      v17 = (unsigned __int16)v15;
      v19 = (unsigned __int16)v15;
      if ( *(_QWORD *)(a1 + 8LL * (unsigned __int16)v15 + 112) )
      {
LABEL_6:
        if ( (*(_BYTE *)(a1 + 500 * v19 + 6634) & 0xFB) == 0 )
        {
          v20 = *(_QWORD *)(a2 + 80);
          v80 = v20;
          if ( v20 )
            sub_B96E90((__int64)&v80, v20, 1);
          v81 = *(_DWORD *)(a2 + 72);
          v21 = sub_33FAF80((__int64)a6, 220, (__int64)&v80, v78, v79, a7, a4);
          v22 = v80;
          *(_QWORD *)a3 = v21;
          *(_DWORD *)(a3 + 8) = v23;
          if ( v22 )
            sub_B91220((__int64)&v80, v22);
          return 1;
        }
      }
LABEL_19:
      if ( (unsigned __int16)(v15 - 17) <= 0xD3u )
        LOWORD(v15) = word_4456580[v15 - 1];
      goto LABEL_21;
    }
  }
  else if ( (_WORD)v15 )
  {
    goto LABEL_19;
  }
  if ( !sub_30070B0((__int64)&v76) )
    return 0;
  LOWORD(v15) = sub_3009970((__int64)&v76, v17, v27, v28, v29);
LABEL_21:
  if ( (_WORD)v15 != 8 )
    return 0;
  v25 = v78;
  if ( (_WORD)v78 )
  {
    if ( (unsigned __int16)(v78 - 17) <= 0xD3u )
      v25 = word_4456580[(unsigned __int16)v78 - 1];
  }
  else
  {
    if ( !sub_30070B0((__int64)&v78) )
      return 0;
    v25 = sub_3009970((__int64)&v78, v17, v30, v31, v32);
  }
  if ( v25 != 13 )
    return 0;
  if ( (_WORD)v76 )
  {
    if ( (unsigned __int16)(v76 - 17) <= 0xD3u )
    {
      if ( *(_QWORD *)(a1 + 8LL * (unsigned __int16)v76 + 112) )
      {
        if ( (*(_BYTE *)(a1 + 500LL * (unsigned __int16)v76 + 6606) & 0xFB) == 0 )
        {
          v26 = 1;
          if ( (_WORD)v78 == 1
            || (_WORD)v78 && (v26 = (unsigned __int16)v78, *(_QWORD *)(a1 + 8LL * (unsigned __int16)v78 + 112)) )
          {
            if ( (*(_BYTE *)(a1 + 500 * v26 + 6510) & 0xFB) == 0 )
            {
              v10 = sub_328A020(a1, 0x61u, v78, v79, 0);
              if ( !(_BYTE)v10 )
                return v10;
              v71 = v76;
              v73 = v77;
              v10 = sub_328C7F0(a1, 0xBBu, v76, v77, 0);
              if ( !(_BYTE)v10 )
                return v10;
              v10 = sub_328C7F0(a1, 0xBAu, v71, v73, 0);
              if ( !(_BYTE)v10 )
                return v10;
              goto LABEL_43;
            }
          }
        }
      }
      return 0;
    }
  }
  else if ( sub_30070B0((__int64)&v76) )
  {
    return 0;
  }
LABEL_43:
  v33 = *(_QWORD *)(a2 + 80);
  v80 = v33;
  if ( v33 )
    sub_B96E90((__int64)&v80, v33, 1);
  v81 = *(_DWORD *)(a2 + 72);
  *(_QWORD *)&v34 = sub_3400BD0((__int64)a6, 0x4330000000000000LL, (__int64)&v80, v76, v77, 0, a4, 0);
  v74 = v34;
  *(_QWORD *)&v35 = sub_33FE730((__int64)a6, (__int64)&v80, v78, v79, 0, (__m128i)0x4530000000100000uLL);
  v72 = v35;
  *(_QWORD *)&v36 = sub_3400BD0(
                      (__int64)a6,
                      0x4530000000000000LL,
                      (__int64)&v80,
                      v76,
                      v77,
                      0,
                      (__m128i)0x4530000000100000uLL,
                      0);
  v70 = v36;
  *(_QWORD *)&v37 = sub_3400BD0(
                      (__int64)a6,
                      0xFFFFFFFFLL,
                      (__int64)&v80,
                      v76,
                      v77,
                      0,
                      (__m128i)0x4530000000100000uLL,
                      0);
  v66 = v37;
  *(_QWORD *)&v38 = sub_3400E40((__int64)a6, 32, v76, v77, (__int64)&v80, (__m128i)0x4530000000100000uLL);
  v68 = v38;
  *((_QWORD *)&v63 + 1) = v13;
  *(_QWORD *)&v63 = v12;
  *(_QWORD *)&v40 = sub_3406EB0(a6, 0xBAu, (__int64)&v80, v76, v77, v39, v63, v66);
  *((_QWORD *)&v64 + 1) = v13;
  *(_QWORD *)&v64 = v12;
  v67 = v40;
  *(_QWORD *)&v42 = sub_3406EB0(a6, 0xC0u, (__int64)&v80, v76, v77, v41, v64, v68);
  v69 = v42;
  v44 = sub_3406EB0(a6, 0xBBu, (__int64)&v80, v76, v77, v43, v67, v74);
  v46 = v45;
  v47 = (__int64)v44;
  v49 = sub_3406EB0(a6, 0xBBu, (__int64)&v80, v76, v77, v48, v69, v70);
  v51 = v50;
  *(_QWORD *)&v74 = v49;
  v52 = sub_33FB890((__int64)a6, v78, v79, v47, v46, (__m128i)0x4530000000100000uLL);
  v54 = v53;
  v55 = v52;
  *(_QWORD *)&v56 = sub_33FB890((__int64)a6, v78, v79, v74, v51, (__m128i)0x4530000000100000uLL);
  *(_QWORD *)&v58 = sub_3406EB0(a6, 0x61u, (__int64)&v80, v78, v79, v57, v56, v72);
  *((_QWORD *)&v65 + 1) = v54;
  *(_QWORD *)&v65 = v55;
  v60 = sub_3406EB0(a6, 0x60u, (__int64)&v80, v78, v79, v59, v65, v58);
  v61 = v80;
  v10 = 1;
  *(_QWORD *)a3 = v60;
  *(_DWORD *)(a3 + 8) = v62;
  if ( v61 )
  {
    sub_B91220((__int64)&v80, v61);
    return 1;
  }
  return v10;
}
