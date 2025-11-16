// Function: sub_75B260
// Address: 0x75b260
//
__int64 __fastcall sub_75B260(__int64 a1, unsigned __int8 a2)
{
  char v3; // dl
  __int64 (__fastcall *v4)(_QWORD, _QWORD); // r15
  __int64 (__fastcall *v5)(_QWORD, _QWORD, _QWORD); // r14
  int v6; // eax
  _QWORD *v8; // rbx
  __int64 v9; // rdi
  _QWORD *v10; // r13
  __int64 v11; // rdi
  _QWORD *v12; // r13
  __int64 v13; // rdi
  _QWORD *v14; // r13
  __int64 v15; // rdi
  _QWORD *v16; // r13
  __int64 v17; // rdi
  _QWORD *v18; // r13
  __int64 v19; // rdi
  _QWORD *v20; // r13
  __int64 v21; // rdi
  _QWORD *v22; // r13
  __int64 v23; // rdi
  _QWORD *v24; // r13
  __int64 v25; // rdi
  _QWORD *v26; // r13
  __int64 v27; // rdi
  _QWORD *v28; // r13
  __int64 v29; // rdi
  _QWORD *v30; // r13
  __int64 v31; // rdi
  _QWORD *v32; // r13
  __int64 v33; // rdi
  _QWORD *v34; // r13
  __int64 v35; // rdi
  _QWORD *v36; // r13
  __int64 v37; // rdi
  _QWORD *v38; // r13
  __int64 v39; // rdi
  _QWORD *v40; // r13
  __int64 v41; // rdi
  _QWORD *v42; // r13
  __int64 v43; // rdi
  _QWORD *v44; // r13
  __int64 v45; // rdi
  _QWORD *v46; // r13
  __int64 v47; // rdi
  _QWORD *v48; // r13
  __int64 v49; // rdi
  _QWORD *v50; // r13
  __int64 v51; // rdi
  _QWORD *v52; // r13
  __int64 v53; // rdi
  _QWORD *v54; // r13
  __int64 v55; // rdi
  _QWORD *v56; // r13
  __int64 v57; // rdi
  _QWORD *v58; // r13
  __int64 v59; // rdi
  _QWORD *v60; // r13
  __int64 v61; // rdi
  _QWORD *v62; // r13
  __int64 v63; // rdi
  _QWORD *v64; // r13
  __int64 v65; // rdi
  _QWORD *v66; // r13
  __int64 v67; // rdi
  _QWORD *v68; // r13
  __int64 v69; // rdi
  _QWORD *v70; // r13
  __int64 v71; // rdi
  _QWORD *v72; // r13
  __int64 v73; // rdi
  _QWORD *v74; // r13
  __int64 v75; // rdi
  _QWORD *v76; // r13
  __int64 v77; // rdi
  _QWORD *v78; // r13
  __int64 v79; // rdi
  _QWORD *v80; // r13
  __int64 v81; // rdi
  _QWORD *v82; // r13
  __int64 v83; // rdi
  _QWORD *v84; // r13
  __int64 v85; // rdi
  _QWORD *v86; // r13
  __int64 v87; // rdi
  _QWORD *v88; // r13
  __int64 v89; // rdi
  _QWORD *v90; // r13
  __int64 v91; // rdi
  _QWORD *v92; // r13
  __int64 v93; // rdi
  _QWORD *v94; // r13
  __int64 v95; // rdi
  _QWORD *v96; // r13
  __int64 v97; // rdi
  _QWORD *v98; // rbx
  unsigned int v99; // [rsp+8h] [rbp-58h]
  int v100; // [rsp+Ch] [rbp-54h]
  int v101; // [rsp+10h] [rbp-50h]
  int v102; // [rsp+14h] [rbp-4Ch]
  __int64 (__fastcall *v103)(_QWORD, _QWORD); // [rsp+18h] [rbp-48h]
  __int64 (__fastcall *v104)(_QWORD, _QWORD); // [rsp+20h] [rbp-40h]
  __int64 (__fastcall *v105)(_QWORD, _QWORD); // [rsp+28h] [rbp-38h]

  v3 = *(_BYTE *)(a1 - 8);
  v4 = qword_4F08040;
  v5 = qword_4F08038;
  qword_4F08040 = 0;
  v105 = qword_4F08030;
  qword_4F08038 = 0;
  v104 = qword_4F08028;
  qword_4F08028 = 0;
  v103 = qword_4F08020;
  qword_4F08020 = 0;
  v102 = dword_4F08014;
  v101 = dword_4F08010;
  v100 = dword_4D03B64;
  v6 = dword_4F08018;
  dword_4F08018 = 0;
  v99 = v6;
  qword_4F08030 = (__int64 (__fastcall *)(_QWORD, _QWORD))sub_7661C0;
  dword_4F08010 = (v3 & 2) != 0;
  if ( a2 == 23 )
  {
    if ( *(_BYTE *)(a1 + 28) )
    {
      sub_760BD0(a1, 23);
    }
    else
    {
      *(_BYTE *)(a1 - 8) = v3 & 0x7F;
      sub_760BD0(a1, 23);
      if ( dword_4F077C4 == 2 && (unk_4D048F8 || (*(_BYTE *)(a1 - 8) & 2) != 0) )
        sub_75BE40(a1);
      v8 = (_QWORD *)qword_4F072C0;
      if ( qword_4F072C0 )
      {
        do
        {
          while ( (*(_BYTE *)(v8[1] + 203LL) & 8) == 0 )
          {
            v8 = (_QWORD *)*v8;
            if ( !v8 )
              goto LABEL_11;
          }
          sub_760BD0(v8, 57);
          v8 = (_QWORD *)*v8;
        }
        while ( v8 );
      }
LABEL_11:
      v9 = qword_4F06D10[0];
      if ( qword_4F06D10[0] )
      {
        v10 = qword_4F06D10;
        do
        {
          sub_760BD0(v9, 1);
          v9 = *(_QWORD *)(*v10 - 16LL);
          v10 = (_QWORD *)(*v10 - 16LL);
        }
        while ( v9 );
      }
      v11 = qword_4F06D10[2];
      if ( qword_4F06D10[2] )
      {
        v12 = &qword_4F06D10[2];
        do
        {
          sub_760BD0(v11, 2);
          v11 = *(_QWORD *)(*v12 - 16LL);
          v12 = (_QWORD *)(*v12 - 16LL);
        }
        while ( v11 );
      }
      v13 = qword_4F06D10[4];
      if ( qword_4F06D10[4] )
      {
        v14 = &qword_4F06D10[4];
        do
        {
          sub_760BD0(v13, 3);
          v13 = *(_QWORD *)(*v14 - 16LL);
          v14 = (_QWORD *)(*v14 - 16LL);
        }
        while ( v13 );
      }
      v15 = qword_4F06D10[6];
      if ( qword_4F06D10[6] )
      {
        v16 = &qword_4F06D10[6];
        do
        {
          sub_760BD0(v15, 4);
          v15 = *(_QWORD *)(*v16 - 16LL);
          v16 = (_QWORD *)(*v16 - 16LL);
        }
        while ( v15 );
      }
      v17 = qword_4F06D10[8];
      if ( qword_4F06D10[8] )
      {
        v18 = &qword_4F06D10[8];
        do
        {
          sub_760BD0(v17, 5);
          v17 = *(_QWORD *)(*v18 - 16LL);
          v18 = (_QWORD *)(*v18 - 16LL);
        }
        while ( v17 );
      }
      v19 = qword_4F06D10[10];
      if ( qword_4F06D10[10] )
      {
        v20 = &qword_4F06D10[10];
        do
        {
          sub_760BD0(v19, 6);
          v19 = *(_QWORD *)(*v20 - 16LL);
          v20 = (_QWORD *)(*v20 - 16LL);
        }
        while ( v19 );
      }
      v21 = qword_4F06D10[12];
      if ( qword_4F06D10[12] )
      {
        v22 = &qword_4F06D10[12];
        do
        {
          sub_760BD0(v21, 7);
          v21 = *(_QWORD *)(*v22 - 16LL);
          v22 = (_QWORD *)(*v22 - 16LL);
        }
        while ( v21 );
      }
      v23 = qword_4F06D10[14];
      if ( qword_4F06D10[14] )
      {
        v24 = &qword_4F06D10[14];
        do
        {
          sub_760BD0(v23, 8);
          v23 = *(_QWORD *)(*v24 - 16LL);
          v24 = (_QWORD *)(*v24 - 16LL);
        }
        while ( v23 );
      }
      v25 = qword_4F06D10[16];
      if ( qword_4F06D10[16] )
      {
        v26 = &qword_4F06D10[16];
        do
        {
          sub_760BD0(v25, 9);
          v25 = *(_QWORD *)(*v26 - 16LL);
          v26 = (_QWORD *)(*v26 - 16LL);
        }
        while ( v25 );
      }
      v27 = qword_4F06D10[18];
      if ( qword_4F06D10[18] )
      {
        v28 = &qword_4F06D10[18];
        do
        {
          sub_760BD0(v27, 10);
          v27 = *(_QWORD *)(*v28 - 16LL);
          v28 = (_QWORD *)(*v28 - 16LL);
        }
        while ( v27 );
      }
      v29 = qword_4F06D10[20];
      if ( qword_4F06D10[20] )
      {
        v30 = &qword_4F06D10[20];
        do
        {
          sub_760BD0(v29, 11);
          v29 = *(_QWORD *)(*v30 - 16LL);
          v30 = (_QWORD *)(*v30 - 16LL);
        }
        while ( v29 );
      }
      v31 = qword_4F06D10[22];
      if ( qword_4F06D10[22] )
      {
        v32 = &qword_4F06D10[22];
        do
        {
          sub_760BD0(v31, 12);
          v31 = *(_QWORD *)(*v32 - 16LL);
          v32 = (_QWORD *)(*v32 - 16LL);
        }
        while ( v31 );
      }
      v33 = qword_4F06D10[24];
      if ( qword_4F06D10[24] )
      {
        v34 = &qword_4F06D10[24];
        do
        {
          sub_760BD0(v33, 13);
          v33 = *(_QWORD *)(*v34 - 16LL);
          v34 = (_QWORD *)(*v34 - 16LL);
        }
        while ( v33 );
      }
      v35 = qword_4F06D10[26];
      if ( qword_4F06D10[26] )
      {
        v36 = &qword_4F06D10[26];
        do
        {
          sub_760BD0(v35, 14);
          v35 = *(_QWORD *)(*v36 - 16LL);
          v36 = (_QWORD *)(*v36 - 16LL);
        }
        while ( v35 );
      }
      v37 = qword_4F06D10[28];
      if ( qword_4F06D10[28] )
      {
        v38 = &qword_4F06D10[28];
        do
        {
          sub_760BD0(v37, 15);
          v37 = *(_QWORD *)(*v38 - 16LL);
          v38 = (_QWORD *)(*v38 - 16LL);
        }
        while ( v37 );
      }
      v39 = qword_4F06D10[30];
      if ( qword_4F06D10[30] )
      {
        v40 = &qword_4F06D10[30];
        do
        {
          sub_760BD0(v39, 16);
          v39 = *(_QWORD *)(*v40 - 16LL);
          v40 = (_QWORD *)(*v40 - 16LL);
        }
        while ( v39 );
      }
      v41 = qword_4F06D10[32];
      if ( qword_4F06D10[32] )
      {
        v42 = &qword_4F06D10[32];
        do
        {
          sub_760BD0(v41, 17);
          v41 = *(_QWORD *)(*v42 - 16LL);
          v42 = (_QWORD *)(*v42 - 16LL);
        }
        while ( v41 );
      }
      v43 = qword_4F06D10[34];
      if ( qword_4F06D10[34] )
      {
        v44 = &qword_4F06D10[34];
        do
        {
          sub_760BD0(v43, 18);
          v43 = *(_QWORD *)(*v44 - 16LL);
          v44 = (_QWORD *)(*v44 - 16LL);
        }
        while ( v43 );
      }
      v45 = qword_4F06D10[36];
      if ( qword_4F06D10[36] )
      {
        v46 = &qword_4F06D10[36];
        do
        {
          sub_760BD0(v45, 19);
          v45 = *(_QWORD *)(*v46 - 16LL);
          v46 = (_QWORD *)(*v46 - 16LL);
        }
        while ( v45 );
      }
      v47 = qword_4F06D10[38];
      if ( qword_4F06D10[38] )
      {
        v48 = &qword_4F06D10[38];
        do
        {
          sub_760BD0(v47, 20);
          v47 = *(_QWORD *)(*v48 - 16LL);
          v48 = (_QWORD *)(*v48 - 16LL);
        }
        while ( v47 );
      }
      v49 = qword_4F06D10[40];
      if ( qword_4F06D10[40] )
      {
        v50 = &qword_4F06D10[40];
        do
        {
          sub_760BD0(v49, 21);
          v49 = *(_QWORD *)(*v50 - 16LL);
          v50 = (_QWORD *)(*v50 - 16LL);
        }
        while ( v49 );
      }
      v51 = qword_4F06D10[42];
      if ( qword_4F06D10[42] )
      {
        v52 = &qword_4F06D10[42];
        do
        {
          sub_760BD0(v51, 22);
          v51 = *(_QWORD *)(*v52 - 16LL);
          v52 = (_QWORD *)(*v52 - 16LL);
        }
        while ( v51 );
      }
      v53 = qword_4F06D10[44];
      if ( qword_4F06D10[44] )
      {
        v54 = &qword_4F06D10[44];
        do
        {
          sub_760BD0(v53, 23);
          v53 = *(_QWORD *)(*v54 - 16LL);
          v54 = (_QWORD *)(*v54 - 16LL);
        }
        while ( v53 );
      }
      v55 = qword_4F06D10[52];
      if ( qword_4F06D10[52] )
      {
        v56 = &qword_4F06D10[52];
        do
        {
          sub_760BD0(v55, 27);
          v55 = *(_QWORD *)(*v56 - 16LL);
          v56 = (_QWORD *)(*v56 - 16LL);
        }
        while ( v55 );
      }
      v57 = qword_4F06D10[54];
      if ( qword_4F06D10[54] )
      {
        v58 = &qword_4F06D10[54];
        do
        {
          sub_760BD0(v57, 28);
          v57 = *(_QWORD *)(*v58 - 16LL);
          v58 = (_QWORD *)(*v58 - 16LL);
        }
        while ( v57 );
      }
      v59 = qword_4F06D10[56];
      if ( qword_4F06D10[56] )
      {
        v60 = &qword_4F06D10[56];
        do
        {
          sub_760BD0(v59, 29);
          v59 = *(_QWORD *)(*v60 - 16LL);
          v60 = (_QWORD *)(*v60 - 16LL);
        }
        while ( v59 );
      }
      v61 = qword_4F06D10[58];
      if ( qword_4F06D10[58] )
      {
        v62 = &qword_4F06D10[58];
        do
        {
          sub_760BD0(v61, 30);
          v61 = *(_QWORD *)(*v62 - 16LL);
          v62 = (_QWORD *)(*v62 - 16LL);
        }
        while ( v61 );
      }
      v63 = qword_4F06D10[66];
      if ( qword_4F06D10[66] )
      {
        v64 = &qword_4F06D10[66];
        do
        {
          sub_760BD0(v63, 34);
          v63 = *(_QWORD *)(*v64 - 16LL);
          v64 = (_QWORD *)(*v64 - 16LL);
        }
        while ( v63 );
      }
      v65 = qword_4F06D10[68];
      if ( qword_4F06D10[68] )
      {
        v66 = &qword_4F06D10[68];
        do
        {
          sub_760BD0(v65, 35);
          v65 = *(_QWORD *)(*v66 - 16LL);
          v66 = (_QWORD *)(*v66 - 16LL);
        }
        while ( v65 );
      }
      v67 = qword_4F06D10[70];
      if ( qword_4F06D10[70] )
      {
        v68 = &qword_4F06D10[70];
        do
        {
          sub_760BD0(v67, 36);
          v67 = *(_QWORD *)(*v68 - 16LL);
          v68 = (_QWORD *)(*v68 - 16LL);
        }
        while ( v67 );
      }
      v69 = qword_4F06D10[72];
      if ( qword_4F06D10[72] )
      {
        v70 = &qword_4F06D10[72];
        do
        {
          sub_760BD0(v69, 37);
          v69 = *(_QWORD *)(*v70 - 16LL);
          v70 = (_QWORD *)(*v70 - 16LL);
        }
        while ( v69 );
      }
      v71 = qword_4F06D10[74];
      if ( qword_4F06D10[74] )
      {
        v72 = &qword_4F06D10[74];
        do
        {
          sub_760BD0(v71, 38);
          v71 = *(_QWORD *)(*v72 - 16LL);
          v72 = (_QWORD *)(*v72 - 16LL);
        }
        while ( v71 );
      }
      v73 = qword_4F06D10[76];
      if ( qword_4F06D10[76] )
      {
        v74 = &qword_4F06D10[76];
        do
        {
          sub_760BD0(v73, 39);
          v73 = *(_QWORD *)(*v74 - 16LL);
          v74 = (_QWORD *)(*v74 - 16LL);
        }
        while ( v73 );
      }
      v75 = qword_4F06D10[78];
      if ( qword_4F06D10[78] )
      {
        v76 = &qword_4F06D10[78];
        do
        {
          sub_760BD0(v75, 40);
          v75 = *(_QWORD *)(*v76 - 16LL);
          v76 = (_QWORD *)(*v76 - 16LL);
        }
        while ( v75 );
      }
      v77 = qword_4F06D10[80];
      if ( qword_4F06D10[80] )
      {
        v78 = &qword_4F06D10[80];
        do
        {
          sub_760BD0(v77, 41);
          v77 = *(_QWORD *)(*v78 - 16LL);
          v78 = (_QWORD *)(*v78 - 16LL);
        }
        while ( v77 );
      }
      v79 = qword_4F06D10[82];
      if ( qword_4F06D10[82] )
      {
        v80 = &qword_4F06D10[82];
        do
        {
          sub_760BD0(v79, 42);
          v79 = *(_QWORD *)(*v80 - 16LL);
          v80 = (_QWORD *)(*v80 - 16LL);
        }
        while ( v79 );
      }
      v81 = qword_4F06D10[84];
      if ( qword_4F06D10[84] )
      {
        v82 = &qword_4F06D10[84];
        do
        {
          sub_760BD0(v81, 43);
          v81 = *(_QWORD *)(*v82 - 16LL);
          v82 = (_QWORD *)(*v82 - 16LL);
        }
        while ( v81 );
      }
      v83 = qword_4F06D10[94];
      if ( qword_4F06D10[94] )
      {
        v84 = &qword_4F06D10[94];
        do
        {
          sub_760BD0(v83, 48);
          v83 = *(_QWORD *)(*v84 - 16LL);
          v84 = (_QWORD *)(*v84 - 16LL);
        }
        while ( v83 );
      }
      v85 = qword_4F06D10[96];
      if ( qword_4F06D10[96] )
      {
        v86 = &qword_4F06D10[96];
        do
        {
          sub_760BD0(v85, 49);
          v85 = *(_QWORD *)(*v86 - 16LL);
          v86 = (_QWORD *)(*v86 - 16LL);
        }
        while ( v85 );
      }
      v87 = qword_4F06D10[98];
      if ( qword_4F06D10[98] )
      {
        v88 = &qword_4F06D10[98];
        do
        {
          sub_760BD0(v87, 50);
          v87 = *(_QWORD *)(*v88 - 16LL);
          v88 = (_QWORD *)(*v88 - 16LL);
        }
        while ( v87 );
      }
      v89 = qword_4F06D10[126];
      if ( qword_4F06D10[126] )
      {
        v90 = &qword_4F06D10[126];
        do
        {
          sub_760BD0(v89, 64);
          v89 = *(_QWORD *)(*v90 - 16LL);
          v90 = (_QWORD *)(*v90 - 16LL);
        }
        while ( v89 );
      }
      v91 = qword_4F06D10[122];
      if ( qword_4F06D10[122] )
      {
        v92 = &qword_4F06D10[122];
        do
        {
          sub_760BD0(v91, 62);
          v91 = *(_QWORD *)(*v92 - 16LL);
          v92 = (_QWORD *)(*v92 - 16LL);
        }
        while ( v91 );
      }
      v93 = qword_4F06D10[128];
      if ( qword_4F06D10[128] )
      {
        v94 = &qword_4F06D10[128];
        do
        {
          sub_760BD0(v93, 65);
          v93 = *(_QWORD *)(*v94 - 16LL);
          v94 = (_QWORD *)(*v94 - 16LL);
        }
        while ( v93 );
      }
      v95 = qword_4F06D10[130];
      if ( qword_4F06D10[130] )
      {
        v96 = &qword_4F06D10[130];
        do
        {
          sub_760BD0(v95, 66);
          v95 = *(_QWORD *)(*v96 - 16LL);
          v96 = (_QWORD *)(*v96 - 16LL);
        }
        while ( v95 );
      }
      v97 = qword_4F06D10[148];
      if ( qword_4F06D10[148] )
      {
        v98 = &qword_4F06D10[148];
        do
        {
          sub_760BD0(v97, 75);
          v97 = *(_QWORD *)(*v98 - 16LL);
          v98 = (_QWORD *)(*v98 - 16LL);
        }
        while ( v97 );
      }
      sub_766120(a1);
    }
  }
  else
  {
    sub_760BD0(a1, a2);
  }
  qword_4F08040 = v4;
  qword_4F08038 = v5;
  qword_4F08030 = v105;
  qword_4F08028 = v104;
  qword_4F08020 = v103;
  dword_4F08014 = v102;
  dword_4F08010 = v101;
  dword_4D03B64 = v100;
  dword_4F08018 = v99;
  return v99;
}
