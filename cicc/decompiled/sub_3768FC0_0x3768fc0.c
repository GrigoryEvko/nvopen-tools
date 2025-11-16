// Function: sub_3768FC0
// Address: 0x3768fc0
//
unsigned __int8 *__fastcall sub_3768FC0(__int64 *a1, __int64 a2)
{
  __int16 *v3; // rax
  __int16 v4; // dx
  __int64 v5; // rcx
  __int64 v6; // rax
  __int64 v7; // rsi
  __int64 v8; // rax
  __int64 v9; // rax
  unsigned int v10; // eax
  int v11; // r9d
  __int64 v12; // rdx
  __int64 v13; // r8
  __int64 v14; // rdx
  __int64 v15; // rax
  unsigned __int8 *result; // rax
  char v17; // dl
  __int64 v18; // rax
  __int64 v19; // rsi
  __int128 v20; // xmm0
  __int128 v21; // xmm1
  __int64 v22; // rdx
  int v23; // r9d
  unsigned __int8 *v24; // rax
  __int64 v25; // r15
  __int64 v26; // rdx
  __int64 v27; // r13
  unsigned __int16 v28; // dx
  unsigned __int8 *v29; // r12
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // r8
  __int128 v34; // rax
  __int64 v35; // r9
  __int128 v36; // rax
  unsigned __int16 v37; // r13
  __int64 v38; // r12
  __int64 v39; // rdx
  __int64 v40; // rax
  __int64 v41; // rdx
  __int64 v42; // r13
  unsigned __int64 v43; // rdx
  unsigned __int8 *v44; // r12
  __int64 v45; // rdx
  __int64 v46; // r13
  __int64 v47; // r9
  __int128 v48; // rax
  int v49; // r9d
  __int64 v50; // rsi
  char v51; // cl
  __int64 v52; // rdx
  __int64 v53; // rcx
  __int64 v54; // r8
  bool v55; // al
  __int64 v56; // rcx
  __int64 v57; // r8
  unsigned __int16 v58; // ax
  __int64 v59; // rdx
  __int64 v60; // r8
  __int128 v61; // [rsp-40h] [rbp-120h]
  __int128 v62; // [rsp-30h] [rbp-110h]
  __int64 v63; // [rsp-8h] [rbp-E8h]
  unsigned int v64; // [rsp+4h] [rbp-DCh]
  unsigned int v65; // [rsp+8h] [rbp-D8h]
  __int64 v66; // [rsp+8h] [rbp-D8h]
  __int128 v67; // [rsp+10h] [rbp-D0h]
  __int128 v68; // [rsp+10h] [rbp-D0h]
  __int128 v69; // [rsp+20h] [rbp-C0h]
  unsigned __int8 *v70; // [rsp+40h] [rbp-A0h]
  unsigned int v71; // [rsp+50h] [rbp-90h] BYREF
  __int64 v72; // [rsp+58h] [rbp-88h]
  unsigned int v73; // [rsp+60h] [rbp-80h] BYREF
  __int64 v74; // [rsp+68h] [rbp-78h]
  __int64 v75; // [rsp+70h] [rbp-70h] BYREF
  int v76; // [rsp+78h] [rbp-68h]
  unsigned __int64 v77; // [rsp+80h] [rbp-60h] BYREF
  __int64 v78; // [rsp+88h] [rbp-58h]
  __int64 v79; // [rsp+90h] [rbp-50h]
  __int64 v80; // [rsp+98h] [rbp-48h]
  unsigned __int64 v81; // [rsp+A0h] [rbp-40h] BYREF
  __int64 v82; // [rsp+A8h] [rbp-38h]

  v3 = *(__int16 **)(a2 + 48);
  v4 = *v3;
  v5 = *((_QWORD *)v3 + 1);
  v6 = *(_QWORD *)(a2 + 40);
  LOWORD(v71) = v4;
  v7 = *(_QWORD *)(v6 + 40);
  v8 = *(unsigned int *)(v6 + 48);
  v72 = v5;
  v9 = *(_QWORD *)(v7 + 48) + 16 * v8;
  if ( *(_WORD *)v9 != v4 || *(_QWORD *)(v9 + 8) != v5 && !v4 )
    return 0;
  v10 = sub_327FDF0((unsigned __int16 *)&v71, v7);
  v13 = v12;
  v14 = (unsigned __int16)v10;
  v73 = v10;
  v15 = a1[1];
  v74 = v13;
  if ( (_WORD)v14 == 1 )
  {
    v17 = *(_BYTE *)(v15 + 7310);
    if ( v17 && v17 != 4 )
      return 0;
    v14 = 1;
  }
  else
  {
    if ( !(_WORD)v14 )
      return 0;
    v50 = v14 + 14;
    if ( !*(_QWORD *)(v15 + 8 * v14 + 112) )
      return 0;
    v14 = (unsigned __int16)v14;
    v51 = *(_BYTE *)(v15 + 500LL * (unsigned __int16)v14 + 6810);
    if ( v51 )
    {
      if ( v51 != 4 || !*(_QWORD *)(v15 + 8 * v50) )
        return 0;
    }
  }
  if ( (*(_BYTE *)(v15 + 500 * v14 + 6821) & 0xFB) != 0 )
    return 0;
  v18 = *(_QWORD *)(a2 + 40);
  v19 = *(_QWORD *)(a2 + 80);
  v20 = (__int128)_mm_loadu_si128((const __m128i *)(v18 + 80));
  v21 = (__int128)_mm_loadu_si128((const __m128i *)(v18 + 120));
  v75 = v19;
  if ( v19 )
    sub_B96E90((__int64)&v75, v19, 1);
  v76 = *(_DWORD *)(a2 + 72);
  *(_QWORD *)&v69 = sub_33FAF80(*a1, 234, (__int64)&v75, v73, v74, v11, (__m128i)v20);
  *((_QWORD *)&v69 + 1) = v22;
  v24 = sub_33FAF80(*a1, 234, (__int64)&v75, v73, v74, v23, (__m128i)v20);
  v25 = *a1;
  v27 = v26;
  v28 = v73;
  v29 = v24;
  if ( (_WORD)v73 )
  {
    if ( (unsigned __int16)(v73 - 17) > 0xD3u )
    {
LABEL_16:
      v30 = v74;
      goto LABEL_17;
    }
    v28 = word_4456580[(unsigned __int16)v73 - 1];
    v30 = 0;
  }
  else
  {
    v55 = sub_30070B0((__int64)&v73);
    v28 = 0;
    if ( !v55 )
      goto LABEL_16;
    v58 = sub_3009970((__int64)&v73, 234, 0, v56, v57);
    v60 = v59;
    v28 = v58;
    v30 = v60;
  }
LABEL_17:
  LOWORD(v81) = v28;
  v82 = v30;
  if ( v28 )
  {
    if ( v28 == 1 || (unsigned __int16)(v28 - 504) <= 7u )
      goto LABEL_61;
    v31 = *(_QWORD *)&byte_444C4A0[16 * v28 - 16];
  }
  else
  {
    v31 = sub_3007260((__int64)&v81);
    v79 = v31;
    v80 = v32;
  }
  LODWORD(v82) = v31;
  v33 = 1LL << ((unsigned __int8)v31 - 1);
  if ( (unsigned int)v31 > 0x40 )
  {
    v66 = 1LL << ((unsigned __int8)v31 - 1);
    v64 = v31 - 1;
    sub_C43690((__int64)&v81, 0, 0);
    v33 = v66;
    if ( (unsigned int)v82 > 0x40 )
    {
      *(_QWORD *)(v81 + 8LL * (v64 >> 6)) |= v66;
      goto LABEL_22;
    }
  }
  else
  {
    v81 = 0;
  }
  v81 |= v33;
LABEL_22:
  *(_QWORD *)&v34 = sub_34007B0(v25, (__int64)&v81, (__int64)&v75, v73, v74, 0, (__m128i)v20, 0);
  if ( (unsigned int)v82 > 0x40 && v81 )
  {
    v67 = v34;
    j_j___libc_free_0_0(v81);
    v34 = v67;
  }
  *((_QWORD *)&v61 + 1) = v27;
  *(_QWORD *)&v61 = v29;
  *(_QWORD *)&v36 = sub_33FC130((_QWORD *)*a1, 396, (__int64)&v75, v73, v74, v35, v61, v34, v20, v21);
  v37 = v73;
  v38 = *a1;
  v68 = v36;
  if ( (_WORD)v73 )
  {
    if ( (unsigned __int16)(v73 - 17) > 0xD3u )
    {
LABEL_27:
      v39 = v74;
      goto LABEL_28;
    }
    v37 = word_4456580[(unsigned __int16)v73 - 1];
    v39 = 0;
  }
  else
  {
    if ( !sub_30070B0((__int64)&v73) )
      goto LABEL_27;
    v37 = sub_3009970((__int64)&v73, 396, v52, v53, v54);
  }
LABEL_28:
  LOWORD(v77) = v37;
  v78 = v39;
  if ( !v37 )
  {
    v40 = sub_3007260((__int64)&v77);
    v81 = v40;
    v82 = v41;
    goto LABEL_30;
  }
  if ( v37 == 1 || (unsigned __int16)(v37 - 504) <= 7u )
LABEL_61:
    BUG();
  v40 = *(_QWORD *)&byte_444C4A0[16 * v37 - 16];
LABEL_30:
  LODWORD(v78) = v40;
  v42 = ~(1LL << ((unsigned __int8)v40 - 1));
  if ( (unsigned int)v40 <= 0x40 )
  {
    v43 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v40;
    if ( !(_DWORD)v40 )
      v43 = 0;
    v77 = v43;
    goto LABEL_34;
  }
  v65 = v40 - 1;
  sub_C43690((__int64)&v77, -1, 1);
  if ( (unsigned int)v78 <= 0x40 )
  {
LABEL_34:
    v77 &= v42;
    goto LABEL_35;
  }
  *(_QWORD *)(v77 + 8LL * (v65 >> 6)) &= v42;
LABEL_35:
  v44 = sub_34007B0(v38, (__int64)&v77, (__int64)&v75, v73, v74, 0, (__m128i)v20, 0);
  v46 = v45;
  v47 = v63;
  if ( (unsigned int)v78 > 0x40 && v77 )
    j_j___libc_free_0_0(v77);
  *((_QWORD *)&v62 + 1) = v46;
  *(_QWORD *)&v62 = v44;
  *(_QWORD *)&v48 = sub_33FC130((_QWORD *)*a1, 396, (__int64)&v75, v73, v74, v47, v69, v62, v20, v21);
  sub_33FC0E0((_QWORD *)*a1, 400, (__int64)&v75, v73, v74, 8, v48, v68, v20, v21);
  result = sub_33FAF80(*a1, 234, (__int64)&v75, v71, v72, v49, (__m128i)v20);
  if ( v75 )
  {
    v70 = result;
    sub_B91220((__int64)&v75, v75);
    return v70;
  }
  return result;
}
