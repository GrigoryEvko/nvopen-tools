// Function: sub_17C9B10
// Address: 0x17c9b10
//
__int64 __fastcall sub_17C9B10(_QWORD *a1, double a2, double a3, double a4)
{
  __int64 result; // rax
  _QWORD *v6; // rax
  _QWORD *v7; // rbx
  unsigned __int64 *v8; // r13
  __int64 v9; // rax
  unsigned __int64 v10; // rcx
  __int64 v11; // rsi
  unsigned __int8 *v12; // rsi
  __int64 v13; // rbx
  __int64 v14; // r13
  __int64 v15; // r14
  __int64 v16; // rax
  unsigned __int8 *v17; // rsi
  __int64 v18; // rax
  __int64 *v19; // rbx
  unsigned __int64 v20; // rsi
  __int64 v21; // rax
  __int64 v22; // rsi
  __int64 v23; // rdx
  unsigned __int8 *v24; // rsi
  __int64 v25; // r9
  __int64 *v26; // rax
  __int64 *v27; // r13
  unsigned __int64 *v28; // r14
  __int64 v29; // rax
  unsigned __int64 v30; // rcx
  __int64 v31; // rsi
  unsigned __int8 *v32; // rsi
  __int64 v33; // rdx
  int v34; // eax
  __int64 v35; // rcx
  int v36; // esi
  __int64 v37; // rdx
  __int64 *v38; // rax
  __int64 v39; // rdi
  __int64 v40; // r8
  __int64 v41; // r11
  unsigned int v42; // esi
  __int64 v43; // rdx
  unsigned int v44; // r9d
  __int64 *v45; // r14
  __int64 v46; // rax
  __int64 v47; // rdx
  __int64 **v48; // rax
  __int64 v49; // rax
  __int64 v50; // r9
  __int64 *v51; // r13
  __int64 v52; // rcx
  __int64 v53; // rax
  __int64 v54; // rsi
  __int64 v55; // rdx
  unsigned __int8 *v56; // rsi
  int v57; // r9d
  int v58; // r9d
  __int64 v59; // rdi
  __int64 v60; // rdx
  __int64 v61; // rcx
  int v62; // eax
  int v63; // eax
  int v64; // r8d
  __int64 *v65; // rdi
  int v66; // eax
  int v67; // eax
  int v68; // esi
  __int64 v69; // rdi
  __int64 *v70; // rcx
  int v71; // r9d
  unsigned int v72; // r10d
  __int64 v73; // rdx
  int v74; // r10d
  __int64 *v75; // rsi
  unsigned int v76; // [rsp-110h] [rbp-110h]
  unsigned __int64 *v77; // [rsp-108h] [rbp-108h]
  __int64 v78; // [rsp-108h] [rbp-108h]
  __int64 v79; // [rsp-108h] [rbp-108h]
  __int64 v80; // [rsp-108h] [rbp-108h]
  __int64 v81; // [rsp-108h] [rbp-108h]
  __int64 v82; // [rsp-108h] [rbp-108h]
  __int64 v83; // [rsp-108h] [rbp-108h]
  __int64 v84; // [rsp-100h] [rbp-100h]
  __int64 v85; // [rsp-F8h] [rbp-F8h]
  __int64 v86; // [rsp-F8h] [rbp-F8h]
  int v87; // [rsp-F8h] [rbp-F8h]
  __int64 v88; // [rsp-F8h] [rbp-F8h]
  __int64 v89; // [rsp-E0h] [rbp-E0h]
  unsigned __int8 *v90; // [rsp-D0h] [rbp-D0h] BYREF
  __int64 v91[2]; // [rsp-C8h] [rbp-C8h] BYREF
  __int16 v92; // [rsp-B8h] [rbp-B8h]
  unsigned __int8 *v93[2]; // [rsp-A8h] [rbp-A8h] BYREF
  __int16 v94; // [rsp-98h] [rbp-98h]
  unsigned __int8 *v95; // [rsp-88h] [rbp-88h] BYREF
  __int64 v96; // [rsp-80h] [rbp-80h]
  unsigned __int64 *v97; // [rsp-78h] [rbp-78h]
  __int64 v98; // [rsp-70h] [rbp-70h]
  __int64 v99; // [rsp-68h] [rbp-68h]
  int v100; // [rsp-60h] [rbp-60h]
  __int64 v101; // [rsp-58h] [rbp-58h]
  __int64 v102; // [rsp-50h] [rbp-50h]

  result = a1[4];
  if ( (_DWORD)result )
  {
    v84 = 8LL * (unsigned int)result;
    v89 = 0;
    do
    {
      v85 = *(_QWORD *)(a1[3] + v89);
      v13 = *(_QWORD *)(a1[5] + v89);
      v14 = sub_1B40B40(a1[1]);
      v15 = *(_QWORD *)(a1[2] - 24LL);
      v16 = sub_16498A0(v13);
      v101 = 0;
      v102 = 0;
      v17 = *(unsigned __int8 **)(v13 + 48);
      v98 = v16;
      v100 = 0;
      v18 = *(_QWORD *)(v13 + 40);
      v95 = 0;
      v96 = v18;
      v99 = 0;
      v97 = (unsigned __int64 *)(v13 + 24);
      v93[0] = v17;
      if ( v17 )
      {
        sub_1623A60((__int64)v93, (__int64)v17, 2);
        if ( v95 )
          sub_161E7C0((__int64)&v95, (__int64)v95);
        v95 = v93[0];
        if ( v93[0] )
          sub_1623210((__int64)v93, v93[0], (__int64)&v95);
      }
      if ( byte_4FA38A0 )
      {
        v94 = 257;
        v6 = sub_1648A60(64, 2u);
        v7 = v6;
        if ( v6 )
          sub_15F9C10((__int64)v6, 1, v15, (__int64 *)v14, 7, 1, 0);
        if ( v96 )
        {
          v8 = v97;
          sub_157E9D0(v96 + 40, (__int64)v7);
          v9 = v7[3];
          v10 = *v8;
          v7[4] = v8;
          v10 &= 0xFFFFFFFFFFFFFFF8LL;
          v7[3] = v10 | v9 & 7;
          *(_QWORD *)(v10 + 8) = v7 + 3;
          *v8 = *v8 & 7 | (unsigned __int64)(v7 + 3);
        }
        sub_164B780((__int64)v7, (__int64 *)v93);
        if ( !v95 )
          goto LABEL_14;
        v91[0] = (__int64)v95;
        sub_1623A60((__int64)v91, (__int64)v95, 2);
        v11 = v7[6];
        if ( v11 )
          sub_161E7C0((__int64)(v7 + 6), v11);
        v12 = (unsigned __int8 *)v91[0];
        v7[6] = v91[0];
        if ( v12 )
          sub_1623210((__int64)v91, v12, (__int64)(v7 + 6));
      }
      else
      {
        v93[0] = "pgocount.promoted";
        v94 = 259;
        v19 = sub_1648A60(64, 1u);
        if ( v19 )
          sub_15F9210((__int64)v19, *(_QWORD *)(*(_QWORD *)v15 + 24LL), v15, 0, 0, 0);
        if ( v96 )
        {
          v77 = v97;
          sub_157E9D0(v96 + 40, (__int64)v19);
          v20 = *v77;
          v21 = v19[3] & 7;
          v19[4] = (__int64)v77;
          v20 &= 0xFFFFFFFFFFFFFFF8LL;
          v19[3] = v20 | v21;
          *(_QWORD *)(v20 + 8) = v19 + 3;
          *v77 = *v77 & 7 | (unsigned __int64)(v19 + 3);
        }
        sub_164B780((__int64)v19, (__int64 *)v93);
        if ( v95 )
        {
          v91[0] = (__int64)v95;
          sub_1623A60((__int64)v91, (__int64)v95, 2);
          v22 = v19[6];
          v23 = (__int64)(v19 + 6);
          if ( v22 )
          {
            sub_161E7C0((__int64)(v19 + 6), v22);
            v23 = (__int64)(v19 + 6);
          }
          v24 = (unsigned __int8 *)v91[0];
          v19[6] = v91[0];
          if ( v24 )
            sub_1623210((__int64)v91, v24, v23);
        }
        v92 = 257;
        if ( *((_BYTE *)v19 + 16) > 0x10u || *(_BYTE *)(v14 + 16) > 0x10u )
        {
          v94 = 257;
          v49 = sub_15FB440(11, v19, v14, (__int64)v93, 0);
          v50 = v49;
          if ( v96 )
          {
            v51 = (__int64 *)v97;
            v79 = v49;
            sub_157E9D0(v96 + 40, v49);
            v50 = v79;
            v52 = *v51;
            v53 = *(_QWORD *)(v79 + 24);
            *(_QWORD *)(v79 + 32) = v51;
            v52 &= 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)(v79 + 24) = v52 | v53 & 7;
            *(_QWORD *)(v52 + 8) = v79 + 24;
            *v51 = *v51 & 7 | (v79 + 24);
          }
          v80 = v50;
          sub_164B780(v50, v91);
          v25 = v80;
          if ( v95 )
          {
            v90 = v95;
            sub_1623A60((__int64)&v90, (__int64)v95, 2);
            v25 = v80;
            v54 = *(_QWORD *)(v80 + 48);
            v55 = v80 + 48;
            if ( v54 )
            {
              sub_161E7C0(v80 + 48, v54);
              v25 = v80;
              v55 = v80 + 48;
            }
            v56 = v90;
            *(_QWORD *)(v25 + 48) = v90;
            if ( v56 )
            {
              v81 = v25;
              sub_1623210((__int64)&v90, v56, v55);
              v25 = v81;
            }
          }
        }
        else
        {
          v25 = sub_15A2B30(v19, v14, 0, 0, a2, a3, a4);
        }
        v78 = v25;
        v94 = 257;
        v26 = sub_1648A60(64, 2u);
        v27 = v26;
        if ( v26 )
          sub_15F9650((__int64)v26, v78, v15, 0, 0);
        if ( v96 )
        {
          v28 = v97;
          sub_157E9D0(v96 + 40, (__int64)v27);
          v29 = v27[3];
          v30 = *v28;
          v27[4] = (__int64)v28;
          v30 &= 0xFFFFFFFFFFFFFFF8LL;
          v27[3] = v30 | v29 & 7;
          *(_QWORD *)(v30 + 8) = v27 + 3;
          *v28 = *v28 & 7 | (unsigned __int64)(v27 + 3);
        }
        sub_164B780((__int64)v27, (__int64 *)v93);
        if ( v95 )
        {
          v91[0] = (__int64)v95;
          sub_1623A60((__int64)v91, (__int64)v95, 2);
          v31 = v27[6];
          if ( v31 )
            sub_161E7C0((__int64)(v27 + 6), v31);
          v32 = (unsigned __int8 *)v91[0];
          v27[6] = v91[0];
          if ( v32 )
            sub_1623210((__int64)v91, v32, (__int64)(v27 + 6));
        }
        if ( byte_4FA3360 )
        {
          v33 = a1[8];
          v34 = *(_DWORD *)(v33 + 24);
          if ( v34 )
          {
            v35 = *(_QWORD *)(v33 + 8);
            v36 = v34 - 1;
            v37 = (v34 - 1) & (((unsigned int)v85 >> 9) ^ ((unsigned int)v85 >> 4));
            v38 = (__int64 *)(v35 + 16 * v37);
            v39 = *v38;
            if ( v85 != *v38 )
            {
              v63 = 1;
              while ( v39 != -8 )
              {
                v64 = v63 + 1;
                LODWORD(v37) = v36 & (v63 + v37);
                v38 = (__int64 *)(v35 + 16LL * (unsigned int)v37);
                v39 = *v38;
                if ( v85 == *v38 )
                  goto LABEL_45;
                v63 = v64;
              }
              goto LABEL_12;
            }
LABEL_45:
            v40 = v38[1];
            if ( v40 )
            {
              v41 = a1[7];
              v42 = *(_DWORD *)(v41 + 24);
              if ( v42 )
              {
                v43 = *(_QWORD *)(v41 + 8);
                v44 = (v42 - 1) & (((unsigned int)v40 >> 4) ^ ((unsigned int)v40 >> 9));
                v45 = (__int64 *)(v43 + 152LL * v44);
                v46 = *v45;
                if ( v40 == *v45 )
                {
LABEL_48:
                  LODWORD(v47) = *((_DWORD *)v45 + 4);
                  if ( (unsigned int)v47 >= *((_DWORD *)v45 + 5) )
                  {
                    sub_16CD150((__int64)(v45 + 1), v45 + 3, 0, 16, v40, v44);
                    v47 = *((unsigned int *)v45 + 4);
                    v48 = (__int64 **)(v45[1] + 16 * v47);
                  }
                  else
                  {
                    v48 = (__int64 **)(v45[1] + 16LL * (unsigned int)v47);
                  }
                  if ( !v48 )
                    goto LABEL_51;
                  goto LABEL_66;
                }
                v87 = 1;
                v65 = 0;
                while ( v46 != -8 )
                {
                  if ( !v65 && v46 == -16 )
                    v65 = v45;
                  v44 = (v42 - 1) & (v87 + v44);
                  v45 = (__int64 *)(v43 + 152LL * v44);
                  v46 = *v45;
                  if ( v40 == *v45 )
                    goto LABEL_48;
                  ++v87;
                }
                v66 = *(_DWORD *)(v41 + 16);
                if ( v65 )
                  v45 = v65;
                ++*(_QWORD *)v41;
                v62 = v66 + 1;
                if ( 4 * v62 < 3 * v42 )
                {
                  if ( v42 - *(_DWORD *)(v41 + 20) - v62 <= v42 >> 3 )
                  {
                    v88 = v41;
                    v76 = ((unsigned int)v40 >> 4) ^ ((unsigned int)v40 >> 9);
                    v83 = v40;
                    sub_17C7F60(v41, v42);
                    v41 = v88;
                    v67 = *(_DWORD *)(v88 + 24);
                    if ( !v67 )
                    {
LABEL_105:
                      ++*(_DWORD *)(v41 + 16);
                      BUG();
                    }
                    v68 = v67 - 1;
                    v69 = *(_QWORD *)(v88 + 8);
                    v70 = 0;
                    v40 = v83;
                    v71 = 1;
                    v72 = (v67 - 1) & v76;
                    v45 = (__int64 *)(v69 + 152LL * v72);
                    v73 = *v45;
                    v62 = *(_DWORD *)(v88 + 16) + 1;
                    if ( v83 != *v45 )
                    {
                      while ( v73 != -8 )
                      {
                        if ( !v70 && v73 == -16 )
                          v70 = v45;
                        v72 = v68 & (v71 + v72);
                        v45 = (__int64 *)(v69 + 152LL * v72);
                        v73 = *v45;
                        if ( v83 == *v45 )
                          goto LABEL_63;
                        ++v71;
                      }
                      if ( v70 )
                        v45 = v70;
                    }
                  }
                  goto LABEL_63;
                }
              }
              else
              {
                ++*(_QWORD *)v41;
              }
              v86 = v41;
              v82 = v40;
              sub_17C7F60(v41, 2 * v42);
              v41 = v86;
              v57 = *(_DWORD *)(v86 + 24);
              if ( !v57 )
                goto LABEL_105;
              v40 = v82;
              v58 = v57 - 1;
              v59 = *(_QWORD *)(v86 + 8);
              v60 = v58 & (((unsigned int)v82 >> 9) ^ ((unsigned int)v82 >> 4));
              v45 = (__int64 *)(v59 + 152 * v60);
              v61 = *v45;
              v62 = *(_DWORD *)(v86 + 16) + 1;
              if ( v82 != *v45 )
              {
                v74 = 1;
                v75 = 0;
                while ( v61 != -8 )
                {
                  if ( v61 == -16 && !v75 )
                    v75 = v45;
                  LODWORD(v60) = v58 & (v74 + v60);
                  v45 = (__int64 *)(v59 + 152LL * (unsigned int)v60);
                  v61 = *v45;
                  if ( v82 == *v45 )
                    goto LABEL_63;
                  ++v74;
                }
                if ( v75 )
                  v45 = v75;
              }
LABEL_63:
              *(_DWORD *)(v41 + 16) = v62;
              if ( *v45 != -8 )
                --*(_DWORD *)(v41 + 20);
              v48 = (__int64 **)(v45 + 3);
              *v45 = v40;
              v45[1] = (__int64)(v45 + 3);
              v45[2] = 0x800000000LL;
LABEL_66:
              *v48 = v19;
              v48[1] = v27;
              LODWORD(v47) = *((_DWORD *)v45 + 4);
LABEL_51:
              *((_DWORD *)v45 + 4) = v47 + 1;
            }
          }
        }
      }
LABEL_12:
      if ( v95 )
        sub_161E7C0((__int64)&v95, (__int64)v95);
LABEL_14:
      v89 += 8;
      result = v89;
    }
    while ( v89 != v84 );
  }
  return result;
}
