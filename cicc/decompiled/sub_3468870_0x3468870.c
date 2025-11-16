// Function: sub_3468870
// Address: 0x3468870
//
__int64 __fastcall sub_3468870(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r8
  const __m128i *v8; // rax
  __int64 v9; // rsi
  __int128 v10; // xmm0
  __int64 v11; // r14
  __int64 v12; // r15
  __int64 v13; // rax
  __int64 v14; // rcx
  __int64 v15; // rbx
  __int64 v16; // rbx
  __int64 v17; // rax
  unsigned int v18; // eax
  __int64 v19; // rdx
  __int64 v20; // r9
  __int64 v21; // rdx
  __int64 v22; // r9
  __int128 v23; // rax
  unsigned __int64 v24; // rax
  __int64 v25; // r14
  unsigned int v26; // edx
  unsigned __int64 v27; // rbx
  __int128 v28; // rax
  __int64 v29; // r9
  __int64 v30; // rax
  unsigned int v31; // ecx
  __int64 v32; // rsi
  __int64 v33; // r10
  __int64 v34; // r15
  unsigned int v35; // edx
  __int64 v36; // rax
  __int64 v37; // r11
  __int16 v38; // dx
  __int64 v39; // rax
  __int128 v40; // kr00_16
  bool v41; // al
  __int64 v42; // r12
  bool v44; // al
  __int64 v45; // rdx
  unsigned int v46; // r15d
  __int128 v47; // rax
  __int64 v48; // r14
  __int64 v49; // rdx
  unsigned __int64 v50; // rdi
  __int64 v51; // r14
  unsigned __int64 v52; // rax
  unsigned __int8 *v53; // r14
  __int64 v54; // rdx
  __int64 v55; // r15
  __int64 v56; // rdx
  __int128 v57; // rax
  __int64 v58; // r9
  __int64 v59; // rax
  unsigned int v60; // ecx
  __int64 v61; // rbx
  __int64 v62; // r10
  unsigned int v63; // edx
  __int64 v64; // rsi
  __int64 v65; // rdx
  __int16 v66; // ax
  __int64 v67; // rdx
  __int64 v68; // r11
  unsigned int v69; // esi
  unsigned int v70; // edx
  unsigned int v71; // ecx
  __int64 v72; // r8
  __int64 v73; // rdx
  bool v74; // al
  __int128 v75; // rax
  __int128 v76; // [rsp-20h] [rbp-110h]
  __int128 v77; // [rsp-20h] [rbp-110h]
  __int128 v78; // [rsp-10h] [rbp-100h]
  __int128 v79; // [rsp-10h] [rbp-100h]
  __int128 v80; // [rsp-10h] [rbp-100h]
  __int128 v81; // [rsp-10h] [rbp-100h]
  unsigned int v82; // [rsp+8h] [rbp-E8h]
  __int128 v83; // [rsp+10h] [rbp-E0h]
  __int128 v84; // [rsp+20h] [rbp-D0h]
  __int64 v85; // [rsp+20h] [rbp-D0h]
  int v86; // [rsp+30h] [rbp-C0h]
  __int128 v87; // [rsp+30h] [rbp-C0h]
  __int64 (__fastcall *v88)(__int64, __int64, __int64, _QWORD, __int64); // [rsp+40h] [rbp-B0h]
  __int64 v89; // [rsp+40h] [rbp-B0h]
  __int64 v90; // [rsp+48h] [rbp-A8h]
  __int64 v91; // [rsp+50h] [rbp-A0h]
  __int64 v92; // [rsp+50h] [rbp-A0h]
  __int128 v93; // [rsp+50h] [rbp-A0h]
  __int128 v94; // [rsp+60h] [rbp-90h]
  unsigned int v95; // [rsp+70h] [rbp-80h]
  unsigned int v96; // [rsp+70h] [rbp-80h]
  __int16 v97; // [rsp+78h] [rbp-78h]
  __int64 v98; // [rsp+78h] [rbp-78h]
  __int64 v99; // [rsp+80h] [rbp-70h] BYREF
  __int64 v100; // [rsp+88h] [rbp-68h]
  __int64 v101; // [rsp+90h] [rbp-60h] BYREF
  int v102; // [rsp+98h] [rbp-58h]
  __int64 v103; // [rsp+A0h] [rbp-50h]
  __int64 v104; // [rsp+A8h] [rbp-48h]
  __int64 v105; // [rsp+B0h] [rbp-40h] BYREF
  __int64 v106; // [rsp+B8h] [rbp-38h]

  v6 = a2;
  v86 = *(_DWORD *)(a2 + 24);
  v8 = *(const __m128i **)(a2 + 40);
  v9 = *(_QWORD *)(a2 + 80);
  v10 = (__int128)_mm_loadu_si128(v8);
  v11 = v8[2].m128i_i64[1];
  v12 = v8[3].m128i_i64[0];
  v13 = *(_QWORD *)(v8->m128i_i64[0] + 48) + 16LL * v8->m128i_u32[2];
  LOWORD(v14) = *(_WORD *)v13;
  v15 = *(_QWORD *)(v13 + 8);
  v101 = v9;
  LOWORD(v99) = v14;
  v100 = v15;
  if ( v9 )
  {
    v91 = v6;
    v97 = v14;
    sub_B96E90((__int64)&v101, v9, 1);
    v6 = v91;
    LOWORD(v14) = v97;
  }
  v102 = *(_DWORD *)(v6 + 72);
  if ( (_WORD)v14 )
  {
    if ( (unsigned __int16)(v14 - 17) > 0xD3u )
    {
      LOWORD(v105) = v14;
      v106 = v15;
      goto LABEL_10;
    }
    v14 = (unsigned __int16)v14;
    if ( *(_QWORD *)(a1 + 8LL * (unsigned __int16)v14 + 112)
      && (*(_BYTE *)(a1 + 500LL * (unsigned __int16)v14 + 6620) & 0xFB) == 0 )
    {
      v106 = 0;
      LOWORD(v14) = word_4456580[(unsigned __int16)v14 - 1];
      LOWORD(v105) = v14;
      if ( (_WORD)v14 )
      {
LABEL_10:
        if ( (_WORD)v14 == 1 || (unsigned __int16)(v14 - 504) <= 7u )
          BUG();
        v16 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v14 - 16];
LABEL_13:
        v92 = *(_QWORD *)(a3 + 64);
        v88 = *(__int64 (__fastcall **)(__int64, __int64, __int64, _QWORD, __int64))(*(_QWORD *)a1 + 528LL);
        v17 = sub_2E79000(*(__int64 **)(a3 + 40));
        v18 = v88(a1, v17, v92, (unsigned int)v99, v100);
        *((_QWORD *)&v78 + 1) = v12;
        *(_QWORD *)&v78 = v11;
        v89 = v19;
        v95 = v18;
        *(_QWORD *)&v93 = sub_3406EB0((_QWORD *)a3, 0xBEu, (__int64)&v101, (unsigned int)v99, v100, v20, v10, v78);
        *((_QWORD *)&v93 + 1) = v21;
        if ( v86 != 86 )
        {
          *((_QWORD *)&v79 + 1) = v12;
          *(_QWORD *)&v79 = v11;
          *((_QWORD *)&v76 + 1) = v21;
          *(_QWORD *)&v76 = v93;
          *(_QWORD *)&v23 = sub_3406EB0((_QWORD *)a3, 0xC0u, (__int64)&v101, (unsigned int)v99, v100, v22, v76, v79);
          LODWORD(v106) = v16;
          v87 = v23;
          if ( (unsigned int)v16 > 0x40 )
          {
            sub_C43690((__int64)&v105, -1, 1);
          }
          else
          {
            v24 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v16;
            if ( !(_DWORD)v16 )
              v24 = 0;
            v105 = v24;
          }
          v25 = (__int64)sub_34007B0(a3, (__int64)&v105, (__int64)&v101, v99, v100, 0, (__m128i)v10, 0);
          v27 = v26;
          if ( (unsigned int)v106 > 0x40 && v105 )
            j_j___libc_free_0_0(v105);
LABEL_21:
          *(_QWORD *)&v28 = sub_33ED040((_QWORD *)a3, 0x16u);
          v30 = sub_340F900((_QWORD *)a3, 0xD0u, (__int64)&v101, v95, v89, v29, v10, v87, v28);
          v31 = v99;
          v32 = v30;
          v33 = v30;
          v34 = v100;
          v36 = *(_QWORD *)(v30 + 48) + 16LL * v35;
          v37 = v35;
          v38 = *(_WORD *)v36;
          v39 = *(_QWORD *)(v36 + 8);
          LOWORD(v105) = v38;
          v106 = v39;
          v40 = __PAIR128__(v27, v25);
          if ( v38 )
          {
            v41 = (unsigned __int16)(v38 - 17) <= 0xD3u;
          }
          else
          {
            *((_QWORD *)&v94 + 1) = v27;
            v96 = v99;
            v90 = v37;
            *(_QWORD *)&v94 = v25;
            v41 = sub_30070B0((__int64)&v105);
            v31 = v96;
            v33 = v32;
            v37 = v90;
            v40 = v94;
          }
          v42 = sub_340EC60(
                  (_QWORD *)a3,
                  205 - ((unsigned int)!v41 - 1),
                  (__int64)&v101,
                  v31,
                  v34,
                  0,
                  v33,
                  v37,
                  v40,
                  v93);
          goto LABEL_24;
        }
        *((_QWORD *)&v80 + 1) = v12;
        *(_QWORD *)&v80 = v11;
        v46 = v16 - 1;
        *((_QWORD *)&v77 + 1) = v21;
        *(_QWORD *)&v77 = v93;
        *(_QWORD *)&v47 = sub_3406EB0((_QWORD *)a3, 0xBFu, (__int64)&v101, (unsigned int)v99, v100, v22, v77, v80);
        LODWORD(v106) = v16;
        v87 = v47;
        v48 = 1LL << ((unsigned __int8)v16 - 1);
        if ( (unsigned int)v16 <= 0x40 )
        {
          v105 = 1LL << ((unsigned __int8)v16 - 1);
          *(_QWORD *)&v83 = sub_34007B0(a3, (__int64)&v105, (__int64)&v101, v99, v100, 0, (__m128i)v10, 0);
          *((_QWORD *)&v83 + 1) = v49;
          if ( (unsigned int)v106 <= 0x40 )
          {
            LODWORD(v106) = v16;
            v51 = ~v48;
            goto LABEL_37;
          }
LABEL_34:
          v50 = v105;
          if ( !v105 )
            goto LABEL_36;
          goto LABEL_35;
        }
        sub_C43690((__int64)&v105, 0, 0);
        if ( (unsigned int)v106 <= 0x40 )
        {
          v105 |= v48;
          *(_QWORD *)&v75 = sub_34007B0(a3, (__int64)&v105, (__int64)&v101, v99, v100, 0, (__m128i)v10, 0);
          v83 = v75;
          if ( (unsigned int)v106 > 0x40 )
            goto LABEL_34;
        }
        else
        {
          v71 = v99;
          v72 = v100;
          *(_QWORD *)(v105 + 8LL * (v46 >> 6)) |= v48;
          *(_QWORD *)&v83 = sub_34007B0(a3, (__int64)&v105, (__int64)&v101, v71, v72, 0, (__m128i)v10, 0);
          *((_QWORD *)&v83 + 1) = v73;
          if ( (unsigned int)v106 > 0x40 )
          {
            v50 = v105;
            if ( v105 )
            {
LABEL_35:
              j_j___libc_free_0_0(v50);
LABEL_36:
              v51 = ~v48;
              LODWORD(v106) = v16;
              if ( (unsigned int)v16 <= 0x40 )
              {
LABEL_37:
                v52 = 0xFFFFFFFFFFFFFFFFLL >> (63 - ((v16 - 1) & 0x3F));
                if ( !(_DWORD)v16 )
                  v52 = 0;
                v105 = v52;
                goto LABEL_40;
              }
LABEL_51:
              sub_C43690((__int64)&v105, -1, 1);
              if ( (unsigned int)v106 > 0x40 )
              {
                *(_QWORD *)(v105 + 8LL * (v46 >> 6)) &= v51;
LABEL_41:
                v53 = sub_34007B0(a3, (__int64)&v105, (__int64)&v101, v99, v100, 0, (__m128i)v10, 0);
                v55 = v54;
                if ( (unsigned int)v106 > 0x40 && v105 )
                  j_j___libc_free_0_0(v105);
                *(_QWORD *)&v84 = sub_3400BD0(a3, 0, (__int64)&v101, (unsigned int)v99, v100, 0, (__m128i)v10, 0);
                *((_QWORD *)&v84 + 1) = v56;
                *(_QWORD *)&v57 = sub_33ED040((_QWORD *)a3, 0x14u);
                v59 = sub_340F900((_QWORD *)a3, 0xD0u, (__int64)&v101, v95, v89, v58, v10, v84, v57);
                v60 = v99;
                v61 = v100;
                v62 = v59;
                v64 = v63;
                v65 = *(_QWORD *)(v59 + 48) + 16LL * v63;
                v66 = *(_WORD *)v65;
                v67 = *(_QWORD *)(v65 + 8);
                v68 = v64;
                LOWORD(v105) = v66;
                v106 = v67;
                if ( v66 )
                {
                  v69 = ((unsigned __int16)(v66 - 17) < 0xD4u) + 205;
                }
                else
                {
                  v82 = v99;
                  v85 = v62;
                  v74 = sub_30070B0((__int64)&v105);
                  v60 = v82;
                  v62 = v85;
                  v68 = v64;
                  v69 = 205 - (!v74 - 1);
                }
                *((_QWORD *)&v81 + 1) = v55;
                *(_QWORD *)&v81 = v53;
                v25 = sub_340EC60((_QWORD *)a3, v69, (__int64)&v101, v60, v61, 0, v62, v68, v83, v81);
                v27 = v70;
                goto LABEL_21;
              }
LABEL_40:
              v105 &= v51;
              goto LABEL_41;
            }
          }
        }
        LODWORD(v106) = v16;
        v51 = ~v48;
        goto LABEL_51;
      }
LABEL_29:
      v103 = sub_3007260((__int64)&v105);
      LODWORD(v16) = v103;
      v104 = v45;
      goto LABEL_13;
    }
  }
  else
  {
    v98 = v6;
    v44 = sub_30070B0((__int64)&v99);
    v6 = v98;
    if ( !v44 )
    {
      v106 = v15;
      LOWORD(v105) = 0;
      goto LABEL_29;
    }
  }
  v42 = (__int64)sub_3412A00((_QWORD *)a3, v6, 0, v14, v6, a6, (__m128i)v10);
LABEL_24:
  if ( v101 )
    sub_B91220((__int64)&v101, v101);
  return v42;
}
