// Function: sub_3393440
// Address: 0x3393440
//
void __fastcall sub_3393440(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r12
  __int64 v5; // rax
  int v6; // edx
  __int64 v7; // rsi
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r15
  __int64 v11; // r14
  __int64 v12; // r13
  __int64 v13; // rax
  __int64 v14; // r8
  __int128 v15; // rax
  int v16; // r9d
  __int64 v17; // rax
  __int64 v18; // rsi
  __int64 v19; // rax
  unsigned __int64 v20; // rdx
  __int64 (__fastcall *v21)(__int64, __int64, unsigned int); // r13
  __int64 v22; // rax
  int v23; // edx
  __int16 v24; // ax
  __int64 v25; // rdi
  __int64 v26; // rax
  __int64 v27; // rcx
  unsigned int v28; // edx
  unsigned __int64 v29; // r13
  __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // r8
  __int128 v33; // rax
  __int128 v34; // rax
  __int64 v35; // rax
  __int64 v36; // r8
  __int64 v37; // r9
  __int64 v38; // rdx
  __int64 v39; // r14
  __int64 v40; // r13
  __int64 v41; // r15
  __int64 v42; // r14
  __int128 v43; // rax
  int v44; // r9d
  int v45; // edx
  __int64 v46; // r12
  __int64 v47; // r14
  __int64 v48; // r12
  __int64 v49; // r14
  unsigned __int64 v50; // r13
  __int64 v51; // rdx
  __int64 v52; // rdx
  char v53; // al
  unsigned int v54; // eax
  unsigned __int64 v55; // rdx
  __int64 v56; // rax
  __int64 v57; // r13
  __int64 v58; // r14
  unsigned __int16 *v59; // r14
  __int64 v60; // rdx
  __int64 v61; // rax
  __int64 v62; // r14
  __int64 v63; // rax
  int v64; // eax
  int v65; // edx
  __int128 v66; // rax
  __int64 v67; // r13
  __int64 v68; // rdx
  __int64 v69; // r14
  __int128 v70; // rax
  int v71; // r9d
  unsigned int v72; // edx
  __int128 v73; // [rsp-40h] [rbp-170h]
  __int128 v74; // [rsp-30h] [rbp-160h]
  __int128 v75; // [rsp-20h] [rbp-150h]
  __int128 v76; // [rsp-20h] [rbp-150h]
  __int128 v77; // [rsp-10h] [rbp-140h]
  __int128 v78; // [rsp+0h] [rbp-130h]
  __int64 (__fastcall *v79)(__int64, __int64, __int64, __int64, __int64); // [rsp+10h] [rbp-120h]
  __int64 v80; // [rsp+18h] [rbp-118h]
  __int64 v81; // [rsp+18h] [rbp-118h]
  __int128 v82; // [rsp+20h] [rbp-110h]
  __int64 v83; // [rsp+20h] [rbp-110h]
  __int64 v84; // [rsp+20h] [rbp-110h]
  __int64 v85; // [rsp+30h] [rbp-100h]
  int v86; // [rsp+30h] [rbp-100h]
  unsigned __int64 v87; // [rsp+38h] [rbp-F8h]
  __int64 v88; // [rsp+40h] [rbp-F0h]
  __int64 v89; // [rsp+40h] [rbp-F0h]
  __int64 v91; // [rsp+60h] [rbp-D0h]
  __int128 v92; // [rsp+60h] [rbp-D0h]
  __int64 v93; // [rsp+C0h] [rbp-70h] BYREF
  int v94; // [rsp+C8h] [rbp-68h]
  unsigned int v95; // [rsp+D0h] [rbp-60h] BYREF
  __int64 v96; // [rsp+D8h] [rbp-58h]
  __int64 v97; // [rsp+E0h] [rbp-50h] BYREF
  char v98; // [rsp+E8h] [rbp-48h]
  __int64 v99; // [rsp+F0h] [rbp-40h]
  __int64 v100; // [rsp+F8h] [rbp-38h]

  v3 = a1;
  v5 = *(_QWORD *)a1;
  v6 = *(_DWORD *)(a1 + 848);
  v93 = 0;
  v94 = v6;
  if ( v5 )
  {
    if ( &v93 != (__int64 *)(v5 + 48) )
    {
      v7 = *(_QWORD *)(v5 + 48);
      v93 = v7;
      if ( v7 )
        sub_B96E90((__int64)&v93, v7, 1);
    }
  }
  v8 = sub_338B750(a1, *(_QWORD *)(a2 + 32));
  v10 = v9;
  v11 = v8;
  v13 = *(_QWORD *)(v8 + 48) + 16LL * (unsigned int)v9;
  v12 = *(_QWORD *)(a1 + 864);
  v14 = *(_QWORD *)(v13 + 8);
  LOWORD(v95) = *(_WORD *)v13;
  v96 = v14;
  *(_QWORD *)&v15 = sub_34007B0(v12, a2, (unsigned int)&v93, v95, v14, 0, 0);
  *((_QWORD *)&v74 + 1) = v10;
  *(_QWORD *)&v74 = v11;
  v17 = sub_3406EB0(v12, 57, (unsigned int)&v93, v95, v96, v16, v74, v15);
  v18 = (unsigned __int16)v95;
  v88 = v17;
  v19 = *(_QWORD *)(a1 + 864);
  v87 = v20;
  v85 = *(_QWORD *)(v19 + 16);
  if ( (_WORD)v95 && *(_QWORD *)(*(_QWORD *)(v19 + 16) + 8LL * (unsigned __int16)v95 + 112) )
  {
    v47 = *(unsigned int *)(a2 + 72);
    if ( (_DWORD)v47 )
    {
      v48 = 0;
      v49 = 32 * v47;
      do
      {
        v50 = *(_QWORD *)(*(_QWORD *)(a2 + 64) + v48);
        if ( (_WORD)v18 )
        {
          if ( (_WORD)v18 == 1 || (unsigned __int16)(v18 - 504) <= 7u )
            BUG();
          v56 = 16LL * ((unsigned __int16)v18 - 1);
          v52 = *(_QWORD *)&byte_444C4A0[v56];
          v53 = byte_444C4A0[v56 + 8];
        }
        else
        {
          v99 = sub_3007260((__int64)&v95);
          v100 = v51;
          v52 = v99;
          v53 = v100;
        }
        v97 = v52;
        v98 = v53;
        v54 = sub_CA1930(&v97);
        if ( v54 <= 0x3F )
        {
          v55 = 0;
          if ( v54 )
            v55 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v54);
          if ( v50 > v55 )
          {
            v3 = a1;
            v19 = *(_QWORD *)(a1 + 864);
            goto LABEL_6;
          }
        }
        v48 += 32;
        v18 = (unsigned __int16)v95;
      }
      while ( v48 != v49 );
      v3 = a1;
    }
    v27 = v88;
    v29 = v87;
  }
  else
  {
LABEL_6:
    v21 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v85 + 32LL);
    v22 = sub_2E79000(*(__int64 **)(v19 + 40));
    if ( v21 == sub_2D42F30 )
    {
      v23 = sub_AE2980(v22, 0)[1];
      v24 = 2;
      if ( v23 != 1 )
      {
        v24 = 3;
        if ( v23 != 2 )
        {
          v24 = 4;
          if ( v23 != 4 )
          {
            v24 = 5;
            if ( v23 != 8 )
            {
              v24 = 6;
              if ( v23 != 16 )
              {
                v24 = 7;
                if ( v23 != 32 )
                {
                  v24 = 8;
                  if ( v23 != 64 )
                    v24 = 9 * (v23 == 128);
                }
              }
            }
          }
        }
      }
    }
    else
    {
      v24 = v21(v85, v22, 0);
    }
    LOWORD(v95) = v24;
    v25 = *(_QWORD *)(v3 + 864);
    v96 = 0;
    v26 = sub_33FB310(v25, v88, v87, &v93, v95, 0);
    v18 = (unsigned __int16)v95;
    v27 = v26;
    v29 = v28 | v87 & 0xFFFFFFFF00000000LL;
  }
  *(_WORD *)(a2 + 44) = v18;
  v80 = v27;
  *(_DWORD *)(a2 + 40) = sub_374C900(*(_QWORD *)(v3 + 960), v18, 0);
  v91 = *(_QWORD *)(v3 + 864);
  *(_QWORD *)&v33 = sub_3373A60(v3, v18, v30, v31, v32, v91);
  v82 = v33;
  *(_QWORD *)&v34 = sub_33F0B60(
                      v91,
                      *(unsigned int *)(a2 + 40),
                      *(unsigned __int16 *)(*(_QWORD *)(v80 + 48) + 16LL * (unsigned int)v29),
                      *(_QWORD *)(*(_QWORD *)(v80 + 48) + 16LL * (unsigned int)v29 + 8));
  *((_QWORD *)&v77 + 1) = v29;
  *(_QWORD *)&v77 = v80;
  v35 = sub_340F900(v91, 49, (unsigned int)&v93, 1, 0, v91, v82, v34, v77);
  v83 = v38;
  v39 = v35;
  v40 = v35;
  v41 = *(_QWORD *)(*(_QWORD *)(a2 + 64) + 8LL);
  if ( !*(_BYTE *)(a2 + 184) )
    sub_3373E10(v3, a3, *(_QWORD *)(a2 + 56), *(unsigned int *)(a2 + 180), v36, v37);
  sub_3373E10(v3, a3, v41, *(unsigned int *)(a2 + 176), v36, v37);
  sub_2E33470(*(unsigned int **)(a3 + 144), *(unsigned int **)(a3 + 152));
  *(_QWORD *)&v92 = v39;
  *((_QWORD *)&v92 + 1) = v83;
  if ( !*(_BYTE *)(a2 + 184) )
  {
    v57 = *(_QWORD *)(v3 + 864);
    v58 = 16LL * (unsigned int)v87;
    *(_QWORD *)&v78 = sub_34007B0(
                        v57,
                        (int)a2 + 16,
                        (unsigned int)&v93,
                        *(unsigned __int16 *)(v58 + *(_QWORD *)(v88 + 48)),
                        *(_QWORD *)(v58 + *(_QWORD *)(v88 + 48) + 8),
                        0,
                        0);
    v59 = (unsigned __int16 *)(*(_QWORD *)(v88 + 48) + v58);
    *((_QWORD *)&v78 + 1) = v60;
    v61 = *(_QWORD *)(v3 + 864);
    v81 = *((_QWORD *)v59 + 1);
    v84 = *v59;
    v62 = *(_QWORD *)(v61 + 64);
    v79 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)v85 + 528LL);
    v63 = sub_2E79000(*(__int64 **)(v61 + 40));
    v64 = v79(v85, v63, v62, v84, v81);
    LODWORD(v62) = v65;
    v86 = v64;
    *(_QWORD *)&v66 = sub_33ED040(v57, 10);
    *((_QWORD *)&v73 + 1) = v87;
    *(_QWORD *)&v73 = v88;
    v67 = sub_340F900(v57, 208, (unsigned int)&v93, v86, v62, v87, v73, v78, v66);
    v69 = v68;
    v89 = *(_QWORD *)(v3 + 864);
    *(_QWORD *)&v70 = sub_33EEAD0(v89, *(_QWORD *)(a2 + 56));
    *((_QWORD *)&v76 + 1) = v69;
    *(_QWORD *)&v76 = v67;
    v40 = sub_340F900(v89, 305, (unsigned int)&v93, 1, 0, v71, v92, v76, v70);
    *((_QWORD *)&v92 + 1) = v72 | *((_QWORD *)&v92 + 1) & 0xFFFFFFFF00000000LL;
  }
  if ( v41 != sub_3374B60(v3, a3) )
  {
    v42 = *(_QWORD *)(v3 + 864);
    *(_QWORD *)&v43 = sub_33EEAD0(v42, v41);
    *((_QWORD *)&v75 + 1) = *((_QWORD *)&v92 + 1);
    *(_QWORD *)&v75 = v40;
    v40 = sub_3406EB0(v42, 301, (unsigned int)&v93, 1, 0, v44, v75, v43);
    DWORD2(v92) = v45;
  }
  v46 = *(_QWORD *)(v3 + 864);
  if ( v40 )
  {
    nullsub_1875(v40, v46, 0);
    *(_QWORD *)(v46 + 384) = v40;
    *(_DWORD *)(v46 + 392) = DWORD2(v92);
    sub_33E2B60(v46, 0);
  }
  else
  {
    *(_QWORD *)(v46 + 384) = 0;
    *(_DWORD *)(v46 + 392) = DWORD2(v92);
  }
  if ( v93 )
    sub_B91220((__int64)&v93, v93);
}
