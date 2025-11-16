// Function: sub_3324380
// Address: 0x3324380
//
__int64 __fastcall sub_3324380(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // r14d
  __int64 v9; // rdx
  __int64 v10; // rcx
  char v11; // bl
  __int64 v12; // r8
  __int64 v13; // r9
  unsigned __int16 *v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rbx
  __int64 v17; // r14
  __int64 v18; // r10
  int v19; // ecx
  __int64 v20; // r11
  __int64 v21; // rsi
  __int64 v22; // rax
  unsigned int v23; // edx
  __int64 v24; // r14
  unsigned int v25; // ebx
  __int64 v27; // rsi
  __int64 v28; // rcx
  int v29; // r9d
  __int64 v30; // rsi
  __int64 v31; // r14
  __int64 v32; // r10
  __int64 v33; // rdx
  __int64 v34; // rbx
  int v35; // ecx
  __int64 v36; // r11
  unsigned __int64 v37; // rdx
  unsigned __int64 v38; // rcx
  size_t v39; // rbx
  unsigned __int64 v40; // r9
  __int64 v41; // r8
  unsigned __int64 v42; // rax
  unsigned int v43; // edx
  __int64 v44; // rdi
  __int64 v45; // rcx
  __int64 v46; // rsi
  __int64 v47; // rax
  unsigned __int16 v48; // dx
  __int64 v49; // r8
  __int64 v50; // rbx
  __int64 v51; // r10
  int v52; // ecx
  __int64 v53; // r11
  __int64 v54; // rsi
  __int64 v55; // rsi
  __int64 v56; // rcx
  __int64 v57; // r8
  __int64 v58; // rbx
  __int64 v59; // r10
  int v60; // ecx
  __int64 v61; // r11
  __int64 v62; // rsi
  __int64 v63; // rax
  __int64 v64; // r8
  unsigned __int64 v65; // r9
  int v66; // edx
  int v67; // ebx
  size_t v68; // r14
  unsigned __int64 v69; // rdx
  unsigned __int64 v70; // rcx
  unsigned __int64 v71; // rax
  __int64 v72; // rcx
  __int64 v73; // rdi
  __int64 v74; // rax
  __int64 v75; // rsi
  __int64 v76; // rax
  __int64 v77; // rax
  __int128 v78; // [rsp-10h] [rbp-90h]
  __int128 v79; // [rsp-10h] [rbp-90h]
  __int128 v80; // [rsp-10h] [rbp-90h]
  __int128 v81; // [rsp-10h] [rbp-90h]
  unsigned __int64 v82; // [rsp-10h] [rbp-90h]
  __int64 v83; // [rsp-8h] [rbp-88h]
  unsigned __int64 v84; // [rsp-8h] [rbp-88h]
  int v85; // [rsp+8h] [rbp-78h]
  int v86; // [rsp+8h] [rbp-78h]
  int v87; // [rsp+10h] [rbp-70h]
  int v88; // [rsp+10h] [rbp-70h]
  __int64 v89; // [rsp+10h] [rbp-70h]
  __int64 v90; // [rsp+10h] [rbp-70h]
  __int64 v91; // [rsp+18h] [rbp-68h]
  __int64 v92; // [rsp+18h] [rbp-68h]
  __int64 v93; // [rsp+20h] [rbp-60h]
  __int64 v94; // [rsp+20h] [rbp-60h]
  int v95; // [rsp+20h] [rbp-60h]
  int v96; // [rsp+20h] [rbp-60h]
  __int64 v97; // [rsp+28h] [rbp-58h]
  __int64 v98; // [rsp+28h] [rbp-58h]
  __int64 v99; // [rsp+30h] [rbp-50h] BYREF
  unsigned int v100; // [rsp+38h] [rbp-48h]
  unsigned __int64 v101; // [rsp+40h] [rbp-40h]
  unsigned int v102; // [rsp+48h] [rbp-38h]

  v6 = a3;
  v11 = sub_33CF8A0(a2, 1, a3, a4, a5, a6);
  if ( v11 )
  {
    if ( (unsigned __int8)sub_33CF8A0(a2, 0, v9, v10, v12, v13) )
      return 0;
  }
  else
  {
    v14 = *(unsigned __int16 **)(a2 + 48);
    v15 = *v14;
    if ( !*(_BYTE *)(a1 + 33)
      || ((v27 = *(_QWORD *)(a1 + 8), v28 = 1, (_WORD)v15 == 1)
       || (_WORD)v15 && (v28 = (unsigned __int16)v15, *(_QWORD *)(v27 + 8LL * (unsigned __int16)v15 + 112)))
      && (v28 *= 500, (*(_BYTE *)(v27 + v28 + 6472) & 0xFB) == 0) )
    {
      v16 = *((_QWORD *)v14 + 1);
      v17 = *(_QWORD *)a1;
      v18 = *(_QWORD *)(a2 + 40);
      v19 = (unsigned __int16)v15;
      v20 = *(unsigned int *)(a2 + 64);
      v21 = *(_QWORD *)(a2 + 80);
      v99 = v21;
      if ( v21 )
      {
        v87 = (unsigned __int16)v15;
        v93 = v18;
        v97 = v20;
        sub_B96E90((__int64)&v99, v21, 1);
        v19 = v87;
        v18 = v93;
        v20 = v97;
      }
      *((_QWORD *)&v78 + 1) = v20;
      *(_QWORD *)&v78 = v18;
      v100 = *(_DWORD *)(a2 + 72);
      v22 = sub_34102A0(v17, 58, (unsigned int)&v99, v19, v16, v13, v78);
      goto LABEL_6;
    }
    if ( (unsigned __int8)sub_33CF8A0(a2, 0, v15, v28, v12, v13) )
    {
      v30 = *(_QWORD *)(a2 + 80);
      v31 = *(_QWORD *)a1;
      v32 = *(_QWORD *)(a2 + 40);
      v33 = *(unsigned int *)(a2 + 64);
      v34 = *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL);
      v35 = **(unsigned __int16 **)(a2 + 48);
      v99 = v30;
      v36 = v33;
      if ( v30 )
      {
        v88 = v35;
        v94 = v32;
        v98 = v33;
        sub_B96E90((__int64)&v99, v30, 1);
        v35 = v88;
        v32 = v94;
        v36 = v98;
      }
      *((_QWORD *)&v79 + 1) = v36;
      *(_QWORD *)&v79 = v32;
      v100 = *(_DWORD *)(a2 + 72);
      v39 = sub_34102A0(v31, 58, (unsigned int)&v99, v35, v34, v29, v79);
      v41 = v83;
      if ( v99 )
        sub_B91220((__int64)&v99, v99);
      if ( *(_DWORD *)(v39 + 24) != 328 )
      {
        v99 = v39;
        sub_32B3B20(a1 + 568, &v99);
        if ( *(int *)(v39 + 88) < 0 )
        {
          *(_DWORD *)(v39 + 88) = *(_DWORD *)(a1 + 48);
          v76 = *(unsigned int *)(a1 + 48);
          v38 = *(unsigned int *)(a1 + 52);
          if ( v76 + 1 > v38 )
          {
            sub_C8D5F0(a1 + 40, (const void *)(a1 + 56), v76 + 1, 8u, v41, v40);
            v76 = *(unsigned int *)(a1 + 48);
          }
          v37 = *(_QWORD *)(a1 + 40);
          *(_QWORD *)(v37 + 8 * v76) = v39;
          ++*(_DWORD *)(a1 + 48);
        }
      }
      v42 = sub_3322160((__int64 *)a1, v39, v37, v38, v41, v40);
      if ( v42 && v39 != v42 )
      {
        if ( !*(_BYTE *)(a1 + 33)
          || ((v44 = *(_QWORD *)(a1 + 8),
               v45 = *(unsigned __int16 *)(*(_QWORD *)(v42 + 48) + 16LL * v43),
               (_WORD)v45 == 1)
           || (_WORD)v45 && *(_QWORD *)(v44 + 8LL * (unsigned __int16)v45 + 112))
          && ((v46 = *(unsigned int *)(v42 + 24), (unsigned int)v46 > 0x1F3)
           || (*(_BYTE *)(v46 + v44 + 500 * v45 + 6414) & 0xFB) == 0) )
        {
          v99 = v42;
          v100 = v43;
          v101 = v42;
LABEL_55:
          v102 = v43;
          return sub_32EB790(a1, a2, &v99, 2, 1);
        }
      }
      return 0;
    }
  }
  v47 = *(_QWORD *)(a2 + 48);
  v48 = *(_WORD *)(v47 + 16);
  if ( !*(_BYTE *)(a1 + 33)
    || ((v55 = *(_QWORD *)(a1 + 8), v56 = 1, v48 == 1) || v48 && (v56 = v48, *(_QWORD *)(v55 + 8LL * v48 + 112)))
    && (*(_BYTE *)(v6 + v55 + 500 * v56 + 6414) & 0xFB) == 0 )
  {
    v49 = *(_QWORD *)(v47 + 24);
    v50 = *(_QWORD *)a1;
    v51 = *(_QWORD *)(a2 + 40);
    v52 = v48;
    v53 = *(unsigned int *)(a2 + 64);
    v54 = *(_QWORD *)(a2 + 80);
    v99 = v54;
    if ( v54 )
    {
      v85 = v48;
      v89 = v51;
      v91 = v53;
      v95 = v49;
      sub_B96E90((__int64)&v99, v54, 1);
      v52 = v85;
      v51 = v89;
      v53 = v91;
      LODWORD(v49) = v95;
    }
    *((_QWORD *)&v80 + 1) = v53;
    *(_QWORD *)&v80 = v51;
    v100 = *(_DWORD *)(a2 + 72);
    v22 = sub_34102A0(v50, v6, (unsigned int)&v99, v52, v49, v29, v80);
LABEL_6:
    v24 = v22;
    v25 = v23;
    if ( v99 )
      sub_B91220((__int64)&v99, v99);
    v99 = v24;
    v100 = v25;
    v101 = v24;
    v102 = v25;
    return sub_32EB790(a1, a2, &v99, 2, 1);
  }
  if ( v11 )
  {
    v57 = *(_QWORD *)(v47 + 24);
    v58 = *(_QWORD *)a1;
    v59 = *(_QWORD *)(a2 + 40);
    v60 = v48;
    v61 = *(unsigned int *)(a2 + 64);
    v62 = *(_QWORD *)(a2 + 80);
    v99 = v62;
    if ( v62 )
    {
      v86 = v48;
      v90 = v59;
      v92 = v61;
      v96 = v57;
      sub_B96E90((__int64)&v99, v62, 1);
      v60 = v86;
      v59 = v90;
      v61 = v92;
      LODWORD(v57) = v96;
    }
    *((_QWORD *)&v81 + 1) = v61;
    *(_QWORD *)&v81 = v59;
    v100 = *(_DWORD *)(a2 + 72);
    v63 = sub_34102A0(v58, v6, (unsigned int)&v99, v60, v57, v29, v81);
    v67 = v66;
    v68 = v63;
    v69 = v82;
    v70 = v84;
    if ( v99 )
      sub_B91220((__int64)&v99, v99);
    if ( *(_DWORD *)(v68 + 24) != 328 )
    {
      v99 = v68;
      sub_32B3B20(a1 + 568, &v99);
      if ( *(int *)(v68 + 88) < 0 )
      {
        *(_DWORD *)(v68 + 88) = *(_DWORD *)(a1 + 48);
        v77 = *(unsigned int *)(a1 + 48);
        v70 = *(unsigned int *)(a1 + 52);
        if ( v77 + 1 > v70 )
        {
          sub_C8D5F0(a1 + 40, (const void *)(a1 + 56), v77 + 1, 8u, v64, v65);
          v77 = *(unsigned int *)(a1 + 48);
        }
        v69 = *(_QWORD *)(a1 + 40);
        *(_QWORD *)(v69 + 8 * v77) = v68;
        ++*(_DWORD *)(a1 + 48);
      }
    }
    v71 = sub_3322160((__int64 *)a1, v68, v69, v70, v64, v65);
    v72 = v71;
    if ( v71 )
    {
      if ( v68 != v71 || v43 != v67 )
      {
        if ( !*(_BYTE *)(a1 + 33)
          || ((v73 = *(_QWORD *)(a1 + 8),
               v74 = *(unsigned __int16 *)(*(_QWORD *)(v71 + 48) + 16LL * v43),
               (_WORD)v74 == 1)
           || (_WORD)v74 && *(_QWORD *)(v73 + 8LL * (unsigned __int16)v74 + 112))
          && ((v75 = *(unsigned int *)(v72 + 24), (unsigned int)v75 > 0x1F3)
           || (*(_BYTE *)(v75 + v73 + 500 * v74 + 6414) & 0xFB) == 0) )
        {
          v99 = v72;
          v100 = v43;
          v101 = v72;
          goto LABEL_55;
        }
      }
    }
  }
  return 0;
}
