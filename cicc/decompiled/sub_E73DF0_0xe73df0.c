// Function: sub_E73DF0
// Address: 0xe73df0
//
__int64 *__fastcall sub_E73DF0(char *a1, __int64 *a2, char *a3)
{
  __int64 *result; // rax
  __int64 *v4; // r14
  __int64 *v5; // r15
  int v6; // eax
  __int64 v7; // rdi
  __int64 v8; // rsi
  __int64 v9; // rcx
  __int64 v10; // r11
  __int64 v11; // r10
  int v12; // eax
  __int64 v13; // r9
  __int64 v14; // r8
  int v15; // eax
  __int64 v16; // rax
  int v17; // ebx
  _QWORD *v18; // r12
  __int64 v19; // rax
  _QWORD *v20; // rbx
  _QWORD *v21; // r13
  _QWORD *v22; // rdi
  __int64 v23; // rdi
  __int64 v24; // rdx
  __int64 *v25; // rax
  __int64 v26; // r14
  __int64 *v27; // r15
  int v28; // eax
  __int64 v29; // rdi
  __int64 v30; // rsi
  __int64 v31; // rcx
  __int64 v32; // r11
  __int64 v33; // r10
  int v34; // eax
  __int64 v35; // r9
  __int64 v36; // r8
  int v37; // eax
  __int64 v38; // rax
  int v39; // ebx
  _QWORD *v40; // r12
  __int64 v41; // rax
  _QWORD *v42; // rbx
  _QWORD *v43; // r13
  _QWORD *v44; // rdi
  __int64 v45; // rdi
  __int64 *v46; // r13
  __int64 *v47; // r14
  int v48; // eax
  __int64 v49; // rdi
  __int64 v50; // rsi
  __int64 v51; // rcx
  __int64 v52; // r11
  __int64 v53; // r10
  int v54; // eax
  __int64 v55; // r9
  __int64 v56; // r8
  int v57; // eax
  __int64 v58; // rax
  int v59; // ebx
  _QWORD *v60; // r12
  __int64 v61; // rax
  _QWORD *v62; // rbx
  _QWORD *v63; // r15
  _QWORD *v64; // rdi
  __int64 v65; // rdi
  char *v66; // [rsp+8h] [rbp-88h]
  __int64 *v67; // [rsp+10h] [rbp-80h]
  signed __int64 v68; // [rsp+18h] [rbp-78h]
  __int64 v69; // [rsp+20h] [rbp-70h]
  __int64 v70; // [rsp+20h] [rbp-70h]
  signed __int64 v71; // [rsp+28h] [rbp-68h]
  __int64 v72; // [rsp+30h] [rbp-60h]
  __int64 v73; // [rsp+30h] [rbp-60h]
  int v74; // [rsp+3Ch] [rbp-54h]
  int v75; // [rsp+3Ch] [rbp-54h]
  int v76; // [rsp+3Ch] [rbp-54h]
  __int64 v77; // [rsp+40h] [rbp-50h]
  __int64 v78; // [rsp+40h] [rbp-50h]
  __int64 v79; // [rsp+40h] [rbp-50h]
  char v80; // [rsp+48h] [rbp-48h]
  char v81; // [rsp+48h] [rbp-48h]
  char v82; // [rsp+48h] [rbp-48h]
  char v83; // [rsp+49h] [rbp-47h]
  char v84; // [rsp+49h] [rbp-47h]
  char v85; // [rsp+49h] [rbp-47h]
  char v86; // [rsp+4Ah] [rbp-46h]
  char v87; // [rsp+4Ah] [rbp-46h]
  char v88; // [rsp+4Ah] [rbp-46h]
  char v89; // [rsp+4Bh] [rbp-45h]
  char v90; // [rsp+4Bh] [rbp-45h]
  char v91; // [rsp+4Bh] [rbp-45h]
  int v92; // [rsp+4Ch] [rbp-44h]
  int v93; // [rsp+4Ch] [rbp-44h]
  int v94; // [rsp+4Ch] [rbp-44h]
  int v95; // [rsp+50h] [rbp-40h]
  int v96; // [rsp+50h] [rbp-40h]
  int v97; // [rsp+50h] [rbp-40h]
  int v98; // [rsp+54h] [rbp-3Ch]
  int v99; // [rsp+54h] [rbp-3Ch]
  int v100; // [rsp+54h] [rbp-3Ch]
  __int64 v102; // [rsp+58h] [rbp-38h]
  __int64 v103; // [rsp+58h] [rbp-38h]

  result = (__int64 *)a3;
  v67 = (__int64 *)a1;
  if ( a1 != (char *)a2 )
  {
    result = (__int64 *)a1;
    if ( a2 != (__int64 *)a3 )
    {
      v66 = &a1[a3 - (char *)a2];
      v68 = 0xAAAAAAAAAAAAAAABLL * ((a3 - a1) >> 5);
      v72 = 0xAAAAAAAAAAAAAAABLL * (((char *)a2 - a1) >> 5);
      if ( v72 == v68 - v72 )
      {
        v46 = a2;
        v47 = (__int64 *)a1;
        do
        {
          v48 = *((_DWORD *)v47 + 14);
          v49 = v47[4];
          v47[4] = 0;
          v50 = v47[5];
          v51 = v47[6];
          v47[5] = 0;
          v52 = *v47;
          v53 = v47[1];
          v100 = v48;
          v54 = *((_DWORD *)v47 + 15);
          v55 = v47[2];
          v47[6] = 0;
          v56 = v47[3];
          v97 = v54;
          v57 = *((_DWORD *)v47 + 16);
          *v47 = *v46;
          v94 = v57;
          v58 = v47[9];
          v47[1] = v46[1];
          v79 = v58;
          v91 = *((_BYTE *)v47 + 80);
          v88 = *((_BYTE *)v47 + 81);
          v76 = *((_DWORD *)v47 + 21);
          v85 = *((_BYTE *)v47 + 88);
          v82 = *((_BYTE *)v47 + 89);
          v47[2] = v46[2];
          v47[3] = v46[3];
          v47[4] = v46[4];
          v47[5] = v46[5];
          v47[6] = v46[6];
          v59 = *((_DWORD *)v46 + 14);
          v46[4] = 0;
          v46[5] = 0;
          v46[6] = 0;
          *((_DWORD *)v47 + 14) = v59;
          *((_DWORD *)v47 + 15) = *((_DWORD *)v46 + 15);
          *((_DWORD *)v47 + 16) = *((_DWORD *)v46 + 16);
          v47[9] = v46[9];
          *((_BYTE *)v47 + 80) = *((_BYTE *)v46 + 80);
          *((_BYTE *)v47 + 81) = *((_BYTE *)v46 + 81);
          *((_DWORD *)v47 + 21) = *((_DWORD *)v46 + 21);
          *((_BYTE *)v47 + 88) = *((_BYTE *)v46 + 88);
          *((_BYTE *)v47 + 89) = *((_BYTE *)v46 + 89);
          *v46 = v52;
          v46[1] = v53;
          v60 = (_QWORD *)v46[4];
          v61 = v46[6];
          v62 = (_QWORD *)v46[5];
          v46[2] = v55;
          v46[3] = v56;
          v63 = v60;
          v73 = v61;
          v46[4] = v49;
          v46[5] = v50;
          for ( v46[6] = v51; v62 != v63; v63 += 13 )
          {
            v64 = (_QWORD *)v63[9];
            if ( v64 != v63 + 11 )
              j_j___libc_free_0(v64, v63[11] + 1LL);
            v65 = v63[6];
            if ( v65 )
              j_j___libc_free_0(v65, v63[8] - v65);
          }
          if ( v60 )
            j_j___libc_free_0(v60, v73 - (_QWORD)v60);
          v47 += 12;
          v46 += 12;
          *((_DWORD *)v46 - 10) = v100;
          *((_DWORD *)v46 - 9) = v97;
          *((_DWORD *)v46 - 8) = v94;
          *(v46 - 3) = v79;
          *((_BYTE *)v46 - 16) = v91;
          *((_BYTE *)v46 - 15) = v88;
          *((_DWORD *)v46 - 3) = v76;
          *((_BYTE *)v46 - 8) = v85;
          *((_BYTE *)v46 - 7) = v82;
        }
        while ( a2 != v47 );
        return a2;
      }
      else
      {
        while ( 1 )
        {
          while ( 1 )
          {
            v71 = v68 - v72;
            if ( v72 >= v68 - v72 )
              break;
            if ( v68 - v72 > 0 )
            {
              v102 = 0;
              v4 = v67;
              v5 = &v67[12 * v72];
              do
              {
                v6 = *((_DWORD *)v4 + 14);
                v7 = v4[4];
                v4[4] = 0;
                v8 = v4[5];
                v9 = v4[6];
                v4[5] = 0;
                v10 = *v4;
                v11 = v4[1];
                v98 = v6;
                v12 = *((_DWORD *)v4 + 15);
                v13 = v4[2];
                v4[6] = 0;
                v14 = v4[3];
                v95 = v12;
                v15 = *((_DWORD *)v4 + 16);
                *v4 = *v5;
                v92 = v15;
                v16 = v4[9];
                v4[1] = v5[1];
                v77 = v16;
                v89 = *((_BYTE *)v4 + 80);
                v86 = *((_BYTE *)v4 + 81);
                v74 = *((_DWORD *)v4 + 21);
                v83 = *((_BYTE *)v4 + 88);
                v80 = *((_BYTE *)v4 + 89);
                v4[2] = v5[2];
                v4[3] = v5[3];
                v4[4] = v5[4];
                v4[5] = v5[5];
                v4[6] = v5[6];
                v17 = *((_DWORD *)v5 + 14);
                v5[4] = 0;
                v5[5] = 0;
                v5[6] = 0;
                *((_DWORD *)v4 + 14) = v17;
                *((_DWORD *)v4 + 15) = *((_DWORD *)v5 + 15);
                *((_DWORD *)v4 + 16) = *((_DWORD *)v5 + 16);
                v4[9] = v5[9];
                *((_BYTE *)v4 + 80) = *((_BYTE *)v5 + 80);
                *((_BYTE *)v4 + 81) = *((_BYTE *)v5 + 81);
                *((_DWORD *)v4 + 21) = *((_DWORD *)v5 + 21);
                *((_BYTE *)v4 + 88) = *((_BYTE *)v5 + 88);
                *((_BYTE *)v4 + 89) = *((_BYTE *)v5 + 89);
                *v5 = v10;
                v5[1] = v11;
                v18 = (_QWORD *)v5[4];
                v19 = v5[6];
                v20 = (_QWORD *)v5[5];
                v5[2] = v13;
                v5[3] = v14;
                v21 = v18;
                v69 = v19;
                v5[4] = v7;
                v5[5] = v8;
                for ( v5[6] = v9; v20 != v21; v21 += 13 )
                {
                  v22 = (_QWORD *)v21[9];
                  if ( v22 != v21 + 11 )
                    j_j___libc_free_0(v22, v21[11] + 1LL);
                  v23 = v21[6];
                  if ( v23 )
                    j_j___libc_free_0(v23, v21[8] - v23);
                }
                if ( v18 )
                  j_j___libc_free_0(v18, v69 - (_QWORD)v18);
                ++v102;
                v4 += 12;
                v5 += 12;
                *((_DWORD *)v5 - 10) = v98;
                *((_DWORD *)v5 - 9) = v95;
                *((_DWORD *)v5 - 8) = v92;
                *(v5 - 3) = v77;
                *((_BYTE *)v5 - 16) = v89;
                *((_BYTE *)v5 - 15) = v86;
                *((_DWORD *)v5 - 3) = v74;
                *((_BYTE *)v5 - 8) = v83;
                *((_BYTE *)v5 - 7) = v80;
              }
              while ( v71 != v102 );
              v67 += 12 * v71;
            }
            v24 = v68 % v72;
            if ( !(v68 % v72) )
              return (__int64 *)v66;
            v68 = v72;
            v72 -= v24;
          }
          v25 = &v67[12 * v68];
          v67 = &v25[-12 * v71];
          if ( v72 > 0 )
          {
            v103 = 0;
            v26 = (__int64)&v25[-12 * v71 - 12];
            v27 = v25 - 12;
            do
            {
              v28 = *(_DWORD *)(v26 + 56);
              v29 = *(_QWORD *)(v26 + 32);
              *(_QWORD *)(v26 + 32) = 0;
              v30 = *(_QWORD *)(v26 + 40);
              v31 = *(_QWORD *)(v26 + 48);
              *(_QWORD *)(v26 + 40) = 0;
              v32 = *(_QWORD *)v26;
              v33 = *(_QWORD *)(v26 + 8);
              v99 = v28;
              v34 = *(_DWORD *)(v26 + 60);
              v35 = *(_QWORD *)(v26 + 16);
              *(_QWORD *)(v26 + 48) = 0;
              v36 = *(_QWORD *)(v26 + 24);
              v96 = v34;
              v37 = *(_DWORD *)(v26 + 64);
              *(_QWORD *)v26 = *v27;
              v93 = v37;
              v38 = *(_QWORD *)(v26 + 72);
              *(_QWORD *)(v26 + 8) = v27[1];
              v78 = v38;
              v90 = *(_BYTE *)(v26 + 80);
              v87 = *(_BYTE *)(v26 + 81);
              v75 = *(_DWORD *)(v26 + 84);
              v84 = *(_BYTE *)(v26 + 88);
              v81 = *(_BYTE *)(v26 + 89);
              *(_QWORD *)(v26 + 16) = v27[2];
              *(_QWORD *)(v26 + 24) = v27[3];
              *(_QWORD *)(v26 + 32) = v27[4];
              *(_QWORD *)(v26 + 40) = v27[5];
              *(_QWORD *)(v26 + 48) = v27[6];
              v39 = *((_DWORD *)v27 + 14);
              v27[4] = 0;
              v27[5] = 0;
              v27[6] = 0;
              *(_DWORD *)(v26 + 56) = v39;
              *(_DWORD *)(v26 + 60) = *((_DWORD *)v27 + 15);
              *(_DWORD *)(v26 + 64) = *((_DWORD *)v27 + 16);
              *(_QWORD *)(v26 + 72) = v27[9];
              *(_BYTE *)(v26 + 80) = *((_BYTE *)v27 + 80);
              *(_BYTE *)(v26 + 81) = *((_BYTE *)v27 + 81);
              *(_DWORD *)(v26 + 84) = *((_DWORD *)v27 + 21);
              *(_BYTE *)(v26 + 88) = *((_BYTE *)v27 + 88);
              *(_BYTE *)(v26 + 89) = *((_BYTE *)v27 + 89);
              *v27 = v32;
              v27[1] = v33;
              v40 = (_QWORD *)v27[4];
              v41 = v27[6];
              v42 = (_QWORD *)v27[5];
              v27[2] = v35;
              v27[3] = v36;
              v43 = v40;
              v70 = v41;
              v27[4] = v29;
              v27[5] = v30;
              for ( v27[6] = v31; v42 != v43; v43 += 13 )
              {
                v44 = (_QWORD *)v43[9];
                if ( v44 != v43 + 11 )
                  j_j___libc_free_0(v44, v43[11] + 1LL);
                v45 = v43[6];
                if ( v45 )
                  j_j___libc_free_0(v45, v43[8] - v45);
              }
              if ( v40 )
                j_j___libc_free_0(v40, v70 - (_QWORD)v40);
              ++v103;
              v26 -= 96;
              v27 -= 12;
              *((_DWORD *)v27 + 38) = v99;
              *((_DWORD *)v27 + 39) = v96;
              *((_DWORD *)v27 + 40) = v93;
              v27[21] = v78;
              *((_BYTE *)v27 + 176) = v90;
              *((_BYTE *)v27 + 177) = v87;
              *((_DWORD *)v27 + 45) = v75;
              *((_BYTE *)v27 + 184) = v84;
              *((_BYTE *)v27 + 185) = v81;
            }
            while ( v72 != v103 );
            v67 -= 12 * v72;
          }
          v72 = v68 % v71;
          if ( !(v68 % v71) )
            break;
          v68 = v71;
        }
        return (__int64 *)v66;
      }
    }
  }
  return result;
}
