// Function: sub_32DC600
// Address: 0x32dc600
//
__int64 __fastcall sub_32DC600(__int64 *a1, __int64 a2)
{
  const __m128i *v4; // rax
  __int64 v5; // r14
  __int64 v6; // r15
  __int32 v7; // ecx
  __int16 *v8; // rax
  __int64 v9; // rsi
  __int16 v10; // dx
  __int64 v11; // rax
  __int64 v12; // rdi
  __m128i si128; // xmm1
  __int64 v14; // rax
  __int64 v16; // rcx
  __int64 v17; // r8
  int v18; // r9d
  char v19; // al
  __int64 v20; // r9
  __int64 v21; // rax
  __int64 v22; // rdx
  int v23; // eax
  __int64 v24; // r14
  __int64 v25; // rcx
  __int128 v26; // rax
  int v27; // r9d
  __int64 v28; // r14
  __int64 v29; // rdx
  __int64 v30; // rbx
  unsigned __int16 *v31; // rax
  unsigned int v32; // eax
  __int64 v33; // rdx
  __int128 v34; // rax
  int v35; // r9d
  __int128 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // rdx
  __int64 v39; // rdi
  __int64 v40; // rdx
  unsigned __int64 v41; // r15
  __int64 v42; // rdi
  __int64 v43; // r14
  __int64 v44; // rsi
  __int64 v45; // rdx
  __int64 v46; // rcx
  unsigned __int64 v47; // r15
  __int128 v48; // rax
  __int64 v49; // rdx
  int v50; // r9d
  __int128 v51; // [rsp-20h] [rbp-110h]
  __int128 v52; // [rsp-20h] [rbp-110h]
  __int128 v53; // [rsp-20h] [rbp-110h]
  __int128 v54; // [rsp-20h] [rbp-110h]
  __int128 v55; // [rsp-10h] [rbp-100h]
  __int128 v56; // [rsp-10h] [rbp-100h]
  unsigned __int32 v57; // [rsp+10h] [rbp-E0h]
  __int64 v58; // [rsp+10h] [rbp-E0h]
  __int64 v59; // [rsp+18h] [rbp-D8h]
  unsigned int v60; // [rsp+18h] [rbp-D8h]
  __int32 v61; // [rsp+18h] [rbp-D8h]
  __int64 v62; // [rsp+20h] [rbp-D0h]
  __int128 v63; // [rsp+20h] [rbp-D0h]
  unsigned int v64; // [rsp+20h] [rbp-D0h]
  __int128 v65; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v66; // [rsp+40h] [rbp-B0h]
  __int64 v67; // [rsp+48h] [rbp-A8h]
  __int64 v68; // [rsp+50h] [rbp-A0h]
  __int64 v69; // [rsp+58h] [rbp-98h]
  __int64 v70; // [rsp+60h] [rbp-90h]
  __int64 v71; // [rsp+68h] [rbp-88h]
  __int64 v72; // [rsp+70h] [rbp-80h]
  __int64 v73; // [rsp+78h] [rbp-78h]
  __int64 v74; // [rsp+80h] [rbp-70h] BYREF
  __int64 v75; // [rsp+88h] [rbp-68h]
  __int64 v76; // [rsp+90h] [rbp-60h] BYREF
  int v77; // [rsp+98h] [rbp-58h]
  __int128 v78; // [rsp+A0h] [rbp-50h] BYREF
  __int64 v79; // [rsp+B0h] [rbp-40h]
  __int64 v80; // [rsp+B8h] [rbp-38h]

  v4 = *(const __m128i **)(a2 + 40);
  v5 = v4[2].m128i_i64[1];
  v6 = v4[3].m128i_i64[0];
  v59 = v4->m128i_i64[0];
  v7 = v4->m128i_i32[2];
  v65 = (__int128)_mm_loadu_si128(v4);
  v57 = v7;
  v8 = *(__int16 **)(a2 + 48);
  v9 = *(_QWORD *)(a2 + 80);
  v10 = *v8;
  v11 = *((_QWORD *)v8 + 1);
  v76 = v9;
  LOWORD(v74) = v10;
  v75 = v11;
  if ( v9 )
    sub_B96E90((__int64)&v76, v9, 1);
  v12 = *a1;
  v77 = *(_DWORD *)(a2 + 72);
  si128 = _mm_load_si128((const __m128i *)&v65);
  v79 = v5;
  v80 = v6;
  v78 = (__int128)si128;
  v14 = sub_3402EA0(v12, 172, (unsigned int)&v76, v74, v75, 0, (__int64)&v78, 2);
  if ( v14 )
    goto LABEL_4;
  if ( (unsigned __int8)sub_33E2390(*a1, v65, *((_QWORD *)&v65 + 1), 1) && !(unsigned __int8)sub_33E2390(*a1, v5, v6, 1) )
  {
    *((_QWORD *)&v51 + 1) = v6;
    *(_QWORD *)&v51 = v5;
    v5 = sub_3411F20(*a1, 172, (unsigned int)&v76, *(_QWORD *)(a2 + 48), *(_DWORD *)(a2 + 68), v18, v51, v65);
    goto LABEL_5;
  }
  if ( (_WORD)v74 )
  {
    if ( (unsigned __int16)(v74 - 17) > 0xD3u )
      goto LABEL_11;
  }
  else if ( !sub_30070B0((__int64)&v74) )
  {
    goto LABEL_11;
  }
  v14 = sub_3295970(a1, a2, (__int64)&v76, v16, v17);
  if ( v14 )
  {
LABEL_4:
    v5 = v14;
    goto LABEL_5;
  }
  if ( (unsigned __int8)sub_33D1AE0(v5, 0) )
    goto LABEL_16;
LABEL_11:
  if ( (unsigned __int8)sub_33CF170(v5, v6) )
    goto LABEL_5;
  if ( (unsigned __int8)sub_33CF4D0(v5, v6) || *(_DWORD *)(v59 + 24) == 51 || *(_DWORD *)(v5 + 24) == 51 )
  {
LABEL_16:
    v5 = sub_3400BD0(*a1, 0, (unsigned int)&v76, v74, v75, 0, 0);
    goto LABEL_5;
  }
  if ( (unsigned __int8)sub_326A930(v5, v6, 1u) )
  {
    v62 = a1[1];
    v19 = sub_328A020(v62, 0xC0u, v74, v75, *((unsigned __int8 *)a1 + 33));
    v20 = v62;
    if ( !v19 )
      goto LABEL_24;
    *(_QWORD *)&v63 = sub_3289780(a1, v5, v6, (__int64)&v76, 0, 0, v78, 0);
    *((_QWORD *)&v63 + 1) = v22;
    if ( (_QWORD)v63 )
    {
      v23 = sub_32844A0((unsigned __int16 *)&v74, v5);
      v24 = *a1;
      *(_QWORD *)&v26 = sub_3400BD0(*a1, v23, (unsigned int)&v76, v74, v75, 0, 0, v25);
      v28 = sub_3406EB0(v24, 57, (unsigned int)&v76, v74, v75, v27, v26, v63);
      v30 = v29;
      v31 = (unsigned __int16 *)(*(_QWORD *)(v59 + 48) + 16LL * v57);
      v32 = sub_325F340(*a1, a1[1], *v31, *((_QWORD *)v31 + 1));
      *(_QWORD *)&v34 = sub_33FB310(*a1, v28, v30, &v76, v32, v33);
      v5 = sub_3406EB0(*a1, 192, (unsigned int)&v76, v74, v75, v35, v65, v34);
      goto LABEL_5;
    }
  }
  v20 = a1[1];
LABEL_24:
  v21 = 1;
  if ( (_WORD)v74 != 1 )
  {
    if ( !(_WORD)v74 )
      goto LABEL_29;
    v21 = (unsigned __int16)v74;
    if ( !*(_QWORD *)(v20 + 8LL * (unsigned __int16)v74 + 112) )
    {
LABEL_28:
      if ( (unsigned __int16)(v74 - 17) > 0xD3u )
      {
        *(_QWORD *)&v36 = sub_325F590(v74);
        v78 = v36;
        v60 = sub_CA1930(&v78);
        v64 = sub_327FC40(*(_QWORD **)(*a1 + 64), 2 * v60);
        v58 = v37;
        if ( sub_328D6E0(a1[1], 0x3Au, v64) )
        {
          *((_QWORD *)&v52 + 1) = v6;
          v72 = sub_33FAF80(*a1, 214, (unsigned int)&v76, v64, v58, v58, v65);
          *(_QWORD *)&v65 = v72;
          *(_QWORD *)&v52 = v5;
          v73 = v38;
          v39 = *a1;
          *((_QWORD *)&v65 + 1) = (unsigned int)v38 | *((_QWORD *)&v65 + 1) & 0xFFFFFFFF00000000LL;
          v70 = sub_33FAF80(v39, 214, (unsigned int)&v76, v64, v58, v58, v52);
          v71 = v40;
          v41 = (unsigned int)v40 | v6 & 0xFFFFFFFF00000000LL;
          *((_QWORD *)&v55 + 1) = v41;
          *(_QWORD *)&v55 = v70;
          v53 = v65;
          v42 = *a1;
          *(_QWORD *)&v65 = v58;
          v68 = sub_3406EB0(v42, 58, (unsigned int)&v76, v64, v58, v58, v53, v55);
          v43 = v68;
          v44 = v60;
          v69 = v45;
          v46 = v65;
          v61 = v65;
          v47 = (unsigned int)v45 | v41 & 0xFFFFFFFF00000000LL;
          *(_QWORD *)&v65 = *a1;
          *(_QWORD *)&v48 = sub_3400E40(v65, v44, v64, v46, &v76);
          *((_QWORD *)&v54 + 1) = v47;
          *(_QWORD *)&v54 = v43;
          v66 = sub_3406EB0(v65, 192, (unsigned int)&v76, v64, v61, v61, v54, v48);
          v67 = v49;
          *((_QWORD *)&v56 + 1) = (unsigned int)v49 | v47 & 0xFFFFFFFF00000000LL;
          *(_QWORD *)&v56 = v66;
          v5 = sub_33FAF80(*a1, 216, (unsigned int)&v76, v74, v75, v50, v56);
          goto LABEL_5;
        }
      }
      goto LABEL_29;
    }
  }
  if ( (*(_BYTE *)(v20 + 500 * v21 + 6586) & 0xFB) != 0 )
    goto LABEL_28;
LABEL_29:
  v5 = 0;
  if ( (unsigned __int8)sub_32D0FE0((__int64)a1, a2, 0) )
    v5 = a2;
LABEL_5:
  if ( v76 )
    sub_B91220((__int64)&v76, v76);
  return v5;
}
