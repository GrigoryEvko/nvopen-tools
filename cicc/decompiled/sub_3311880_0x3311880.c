// Function: sub_3311880
// Address: 0x3311880
//
__int64 __fastcall sub_3311880(_QWORD *a1, __int64 a2)
{
  const __m128i *v4; // rax
  __int64 v5; // r14
  __int64 v6; // r15
  __int64 v7; // rax
  __int16 v8; // dx
  __int64 v9; // rax
  __int64 v10; // rsi
  unsigned __int16 v11; // cx
  __int64 v12; // rax
  __int64 v13; // r13
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rsi
  int v17; // r9d
  __int64 v18; // r15
  int v19; // edx
  __int64 v20; // rax
  int v21; // edx
  __int64 v22; // r13
  char v24; // al
  __int64 v25; // r8
  char v26; // al
  int v27; // r9d
  int v28; // r9d
  __int64 v29; // rcx
  __int64 v30; // rax
  int v31; // eax
  __int64 v32; // rax
  int v33; // edx
  int v34; // r15d
  __int64 v35; // r13
  __int64 v36; // rax
  int v37; // edx
  __int64 v38; // rdi
  __int64 v39; // rcx
  int v40; // esi
  int v41; // r8d
  unsigned __int16 v42; // r13
  __int64 v43; // rdx
  __int64 v44; // rax
  __int64 v45; // rdx
  __int64 v46; // rdi
  __int64 v47; // rax
  int v48; // edx
  int v49; // r9d
  int v50; // edx
  __int64 v51; // rdx
  __int64 v52; // rcx
  __int64 v53; // r8
  __int64 v54; // rax
  __int64 v55; // r13
  __int64 v56; // rdx
  __int64 v57; // rdx
  __int128 v58; // rax
  int v59; // edx
  __int128 v60; // [rsp-30h] [rbp-120h]
  __int128 v61; // [rsp-10h] [rbp-100h]
  __int128 v62; // [rsp-10h] [rbp-100h]
  __int128 v63; // [rsp+0h] [rbp-F0h]
  __int64 v64; // [rsp+0h] [rbp-F0h]
  __int128 v65; // [rsp+0h] [rbp-F0h]
  __int64 v66; // [rsp+10h] [rbp-E0h]
  __int64 v67; // [rsp+18h] [rbp-D8h]
  __int64 v68; // [rsp+18h] [rbp-D8h]
  int v69; // [rsp+18h] [rbp-D8h]
  __int64 v70; // [rsp+20h] [rbp-D0h]
  __int128 v71; // [rsp+20h] [rbp-D0h]
  __int64 v72; // [rsp+30h] [rbp-C0h]
  unsigned __int16 v73; // [rsp+38h] [rbp-B8h]
  __int64 v74; // [rsp+40h] [rbp-B0h]
  __int64 v75; // [rsp+40h] [rbp-B0h]
  int v76; // [rsp+48h] [rbp-A8h]
  int v77; // [rsp+48h] [rbp-A8h]
  __int64 v78; // [rsp+48h] [rbp-A8h]
  __int128 v79; // [rsp+50h] [rbp-A0h]
  int v80; // [rsp+50h] [rbp-A0h]
  __int64 v81; // [rsp+50h] [rbp-A0h]
  bool v82; // [rsp+6Fh] [rbp-81h] BYREF
  int v83; // [rsp+70h] [rbp-80h] BYREF
  __int64 v84; // [rsp+78h] [rbp-78h]
  __int64 v85; // [rsp+80h] [rbp-70h] BYREF
  int v86; // [rsp+88h] [rbp-68h]
  unsigned __int64 v87; // [rsp+90h] [rbp-60h] BYREF
  __int64 v88; // [rsp+98h] [rbp-58h]
  __int64 v89; // [rsp+A0h] [rbp-50h] BYREF
  __int64 v90; // [rsp+A8h] [rbp-48h]
  __int64 v91; // [rsp+B0h] [rbp-40h]
  int v92; // [rsp+B8h] [rbp-38h]

  v4 = *(const __m128i **)(a2 + 40);
  v5 = v4[2].m128i_i64[1];
  v6 = v4[3].m128i_i64[0];
  v70 = v4->m128i_i64[0];
  v79 = (__int128)_mm_loadu_si128(v4);
  v72 = v4->m128i_u32[2];
  v7 = *(_QWORD *)(v4->m128i_i64[0] + 48) + 16 * v72;
  v8 = *(_WORD *)v7;
  v84 = *(_QWORD *)(v7 + 8);
  LODWORD(v7) = *(_DWORD *)(a2 + 24);
  LOWORD(v83) = v8;
  v76 = v7;
  v9 = *(_QWORD *)(a2 + 48);
  v10 = *(_QWORD *)(a2 + 80);
  v11 = *(_WORD *)(v9 + 16);
  v12 = *(_QWORD *)(v9 + 24);
  v85 = v10;
  v73 = v11;
  v74 = v12;
  if ( v10 )
    sub_B96E90((__int64)&v85, v10, 1);
  v86 = *(_DWORD *)(a2 + 72);
  v13 = sub_33DFBC0(v79, *((_QWORD *)&v79 + 1), 0, 0);
  v14 = sub_33DFBC0(v5, v6, 0, 0);
  if ( !v13 || !v14 )
  {
    v67 = v14;
    v24 = sub_33E2390(*a1, v79, *((_QWORD *)&v79 + 1), 1);
    v25 = v67;
    if ( v24 )
    {
      v26 = sub_33E2390(*a1, v5, v6, 1);
      v25 = v67;
      if ( !v26 )
      {
        *((_QWORD *)&v61 + 1) = v6;
        *(_QWORD *)&v61 = v5;
        v22 = sub_3411F20(
                *a1,
                *(_DWORD *)(a2 + 24),
                (unsigned int)&v85,
                *(_QWORD *)(a2 + 48),
                *(_DWORD *)(a2 + 68),
                v27,
                v61,
                v79);
        goto LABEL_10;
      }
    }
    v68 = v25;
    if ( (unsigned __int8)sub_33E0720(v5, v6, 0) )
    {
      v32 = sub_3400BD0(*a1, 0, (unsigned int)&v85, v73, v74, 0, 0);
      LODWORD(v64) = 0;
      v34 = v33;
      v35 = v32;
      v36 = sub_3400BD0(*a1, 0, (unsigned int)&v85, v83, v84, 0, v64);
      LODWORD(v90) = v37;
      v91 = v35;
      v89 = v36;
      v92 = v34;
      v22 = sub_32EB790((__int64)a1, a2, &v89, 2, 1);
      goto LABEL_10;
    }
    if ( v68 )
    {
      v29 = *(_QWORD *)(v68 + 96);
      if ( *(_DWORD *)(v29 + 32) > 0x40u )
      {
        v66 = *(_QWORD *)(v68 + 96);
        v69 = *(_DWORD *)(v29 + 32);
        if ( v69 - (unsigned int)sub_C444A0(v29 + 24) > 0x40 )
          goto LABEL_22;
        v30 = **(_QWORD **)(v66 + 24);
      }
      else
      {
        v30 = *(_QWORD *)(v29 + 24);
      }
      if ( v30 == 2 )
      {
        if ( v76 == 80 )
        {
          if ( (unsigned __int64)sub_32844A0((unsigned __int16 *)&v83, v6) <= 2 )
            goto LABEL_32;
          v38 = *a1;
          v39 = *(_QWORD *)(a2 + 48);
          v40 = 76;
          v41 = *(_DWORD *)(a2 + 68);
        }
        else
        {
          v38 = *a1;
          v39 = *(_QWORD *)(a2 + 48);
          v40 = 77;
          v41 = *(_DWORD *)(a2 + 68);
        }
        v22 = sub_3411F20(v38, v40, (unsigned int)&v85, v39, v41, v28, v79, v79);
        goto LABEL_10;
      }
    }
LABEL_22:
    if ( v76 != 80 )
    {
      v31 = sub_33DDA20(*a1, v70, v72, v5, (unsigned int)v6);
      goto LABEL_24;
    }
LABEL_32:
    v42 = v83;
    if ( (_WORD)v83 )
    {
      if ( (unsigned __int16)(v83 - 17) <= 0xD3u )
      {
        v42 = word_4456580[(unsigned __int16)v83 - 1];
        v43 = 0;
        goto LABEL_35;
      }
    }
    else if ( sub_30070B0((__int64)&v83) )
    {
      v42 = sub_3009970((__int64)&v83, v6, v51, v52, v53);
      goto LABEL_35;
    }
    v43 = v84;
LABEL_35:
    LOWORD(v89) = v42;
    v90 = v43;
    if ( v42 )
    {
      if ( v42 == 1 || (unsigned __int16)(v42 - 504) <= 7u )
        BUG();
      v44 = *(_QWORD *)&byte_444C4A0[16 * v42 - 16];
    }
    else
    {
      v44 = sub_3007260((__int64)&v89);
      v87 = v44;
      v88 = v45;
    }
    v46 = *a1;
    if ( v44 == 1 )
    {
      *((_QWORD *)&v65 + 1) = v6;
      *(_QWORD *)&v65 = v5;
      v54 = sub_3406EB0(v46, 186, (unsigned int)&v85, v83, v84, v28, v79, v65);
      v55 = *a1;
      v78 = v56;
      v81 = v54;
      *(_QWORD *)&v71 = sub_3400BD0(*a1, 0, (unsigned int)&v85, v83, v84, 0, 0);
      *((_QWORD *)&v71 + 1) = v57;
      *(_QWORD *)&v58 = sub_33ED040(v55, 22);
      *((_QWORD *)&v60 + 1) = v78;
      *(_QWORD *)&v60 = v81;
      v91 = sub_340F900(v55, 208, (unsigned int)&v85, v73, v74, DWORD2(v71), v60, v71, v58);
      v89 = v81;
      v92 = v59;
      LODWORD(v90) = v78;
LABEL_40:
      v22 = sub_32EB790((__int64)a1, a2, &v89, 2, 1);
      goto LABEL_10;
    }
    v31 = sub_33DF7F0(v46, v70, v72, v5, (unsigned int)v6);
LABEL_24:
    if ( v31 )
    {
      v22 = 0;
      goto LABEL_10;
    }
    v47 = sub_3400BD0(*a1, 0, (unsigned int)&v85, v73, v74, 0, 0);
    *((_QWORD *)&v62 + 1) = v6;
    *(_QWORD *)&v62 = v5;
    v77 = v48;
    v75 = v47;
    v89 = sub_3406EB0(*a1, 58, (unsigned int)&v85, v83, v84, v49, v79, v62);
    LODWORD(v90) = v50;
    v91 = v75;
    v92 = v77;
    goto LABEL_40;
  }
  v15 = *(_QWORD *)(v14 + 96) + 24LL;
  v16 = *(_QWORD *)(v13 + 96) + 24LL;
  if ( v76 == 80 )
    sub_C4A7C0((__int64)&v87, v16, v15, &v82);
  else
    sub_C49BE0((__int64)&v87, v16, v15, &v82);
  *((_QWORD *)&v63 + 1) = v74;
  *(_QWORD *)&v63 = v73;
  v18 = sub_3401740(*a1, v82, (unsigned int)&v85, v73, v74, v17, v63);
  v80 = v19;
  v20 = sub_34007B0(*a1, (unsigned int)&v87, (unsigned int)&v85, v83, v84, 0, 0);
  LODWORD(v90) = v21;
  v89 = v20;
  v91 = v18;
  v92 = v80;
  v22 = sub_32EB790((__int64)a1, a2, &v89, 2, 1);
  if ( (unsigned int)v88 > 0x40 && v87 )
    j_j___libc_free_0_0(v87);
LABEL_10:
  if ( v85 )
    sub_B91220((__int64)&v85, v85);
  return v22;
}
