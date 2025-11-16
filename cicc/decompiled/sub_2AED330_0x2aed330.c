// Function: sub_2AED330
// Address: 0x2aed330
//
__int64 __fastcall sub_2AED330(__int64 a1, __int64 a2, __int64 a3, int a4)
{
  __int64 v5; // rax
  unsigned int v6; // r15d
  __int64 v8; // rax
  __int64 v9; // rbx
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rdx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // rcx
  unsigned __int64 v18; // rbx
  unsigned __int64 v19; // r13
  unsigned __int64 v20; // rdi
  unsigned __int64 v21; // rdi
  int *v22; // rax
  int *v23; // rcx
  int v24; // edx
  unsigned int v25; // r14d
  int *v26; // r15
  int v27; // edx
  _QWORD *v28; // r9
  __int64 v29; // rcx
  unsigned int v30; // edi
  int *v31; // rsi
  int v32; // r10d
  __int64 v33; // rdx
  __int64 v34; // rdx
  unsigned int v35; // ebx
  unsigned int v36; // eax
  unsigned int v37; // ecx
  unsigned int v38; // r13d
  unsigned int v39; // eax
  unsigned int v40; // ecx
  __int64 v41; // r8
  unsigned int v42; // r13d
  char v43; // al
  __int64 v44; // rdx
  int v45; // ebx
  unsigned int v46; // r13d
  char v47; // si
  unsigned int v48; // ebx
  unsigned int v49; // eax
  unsigned int v50; // esi
  unsigned int v51; // eax
  char v52; // si
  unsigned __int64 v53; // rax
  unsigned __int64 v54; // rcx
  unsigned __int64 v55; // rax
  __int64 v56; // rdi
  unsigned int v57; // ecx
  unsigned int v58; // r8d
  __int64 v59; // r10
  int v60; // eax
  unsigned int v61; // r8d
  unsigned int v62; // r9d
  unsigned int v63; // eax
  unsigned int v64; // ecx
  char *v65; // rdx
  __int64 v66; // rbx
  char *v67; // r13
  __int64 v68; // rbx
  char *v69; // rdi
  __int64 v70; // rsi
  int v71; // esi
  char v72; // si
  unsigned int v73; // eax
  __int64 v74; // rax
  char *v75; // r13
  char *v76; // r14
  __int64 v77; // rbx
  char *v78; // rbx
  unsigned __int64 v79; // rax
  int v80; // edx
  unsigned int v81; // ecx
  unsigned int v82; // ecx
  _QWORD *v83; // rdi
  unsigned int v84; // r10d
  char *v85; // rcx
  unsigned int v86; // ebx
  unsigned int v87; // ecx
  signed __int64 v88; // rax
  signed __int64 v89; // rcx
  signed __int64 v90; // r10
  int v91; // [rsp+0h] [rbp-750h]
  int v92; // [rsp+4h] [rbp-74Ch]
  int v94; // [rsp+18h] [rbp-738h]
  unsigned int v95; // [rsp+1Ch] [rbp-734h]
  unsigned int v96; // [rsp+28h] [rbp-728h]
  unsigned __int8 v97; // [rsp+2Dh] [rbp-723h]
  unsigned __int8 v98; // [rsp+2Eh] [rbp-722h]
  int *v100; // [rsp+30h] [rbp-720h]
  char v102; // [rsp+44h] [rbp-70Ch]
  __int64 v103; // [rsp+48h] [rbp-708h] BYREF
  __int64 v104; // [rsp+50h] [rbp-700h] BYREF
  char v105; // [rsp+58h] [rbp-6F8h]
  _QWORD *v106; // [rsp+60h] [rbp-6F0h] BYREF
  unsigned int v107; // [rsp+68h] [rbp-6E8h]
  char *v108; // [rsp+80h] [rbp-6D0h] BYREF
  __int64 v109; // [rsp+88h] [rbp-6C8h]
  char v110; // [rsp+90h] [rbp-6C0h] BYREF
  __int64 v111; // [rsp+B0h] [rbp-6A0h] BYREF
  char v112; // [rsp+B8h] [rbp-698h]
  int *v113; // [rsp+E0h] [rbp-670h] BYREF
  __int64 v114; // [rsp+E8h] [rbp-668h]
  char v115; // [rsp+F0h] [rbp-660h] BYREF
  unsigned __int64 v116; // [rsp+110h] [rbp-640h] BYREF
  unsigned int v117; // [rsp+118h] [rbp-638h]
  char v118; // [rsp+120h] [rbp-630h] BYREF

  v96 = *(_DWORD *)(a1 + 96);
  if ( !v96 && (!*(_BYTE *)(a1 + 108) || *(_DWORD *)(a1 + 100) != 5) )
  {
    v5 = *(_QWORD *)(a1 + 440);
    if ( *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v5 + 56) + 16LL) + 216LL) == 0xFFFFFFFFLL && !*(_BYTE *)(v5 + 664) )
    {
      v91 = a4;
      v95 = a2;
      v98 = BYTE4(a2);
      v8 = sub_2AA7EC0(*(_QWORD *)(a1 + 424), *(char **)(a1 + 416), 1);
      v102 = BYTE4(v8);
      v92 = v8;
      v94 = *(_DWORD *)(*(_QWORD *)(a1 + 440) + 120LL);
      v97 = v94 != 0;
      if ( a4 || a3 || (v79 = sub_2AD1E10(a1, a2), a3 = v79, (v91 = v80) != 0) || v79 )
      {
        v103 = a2;
        sub_2AE96E0((__int64)&v116, a1, (unsigned int *)&v103, 1u);
        v9 = v116;
        v105 |= 1u;
        v104 = 0;
        sub_2AC3540((__int64)&v104);
        sub_2AC1550((__int64)&v104, v9);
        v108 = &v110;
        v109 = 0x400000000LL;
        if ( *(_DWORD *)(v9 + 56) )
          sub_2AA8A70((__int64)&v108, v9 + 48, v10, v11, v12, v13);
        v112 |= 1u;
        v111 = 0;
        sub_2AC3540((__int64)&v111);
        sub_2AC1550((__int64)&v111, v9 + 96);
        v113 = (int *)&v115;
        v114 = 0x400000000LL;
        v17 = *(unsigned int *)(v9 + 152);
        if ( (_DWORD)v17 )
          sub_2AA8A70((__int64)&v113, v9 + 144, v14, v17, v15, v16);
        v18 = v116;
        v19 = v116 + 192LL * v117;
        if ( v116 != v19 )
        {
          do
          {
            v19 -= 192LL;
            v20 = *(_QWORD *)(v19 + 144);
            if ( v20 != v19 + 160 )
              _libc_free(v20);
            if ( (*(_BYTE *)(v19 + 104) & 1) == 0 )
              sub_C7D6A0(*(_QWORD *)(v19 + 112), 8LL * *(unsigned int *)(v19 + 120), 4);
            v21 = *(_QWORD *)(v19 + 48);
            if ( v21 != v19 + 64 )
              _libc_free(v21);
            if ( (*(_BYTE *)(v19 + 8) & 1) == 0 )
              sub_C7D6A0(*(_QWORD *)(v19 + 16), 8LL * *(unsigned int *)(v19 + 24), 4);
          }
          while ( v18 != v19 );
          v19 = v116;
        }
        if ( (char *)v19 != &v118 )
          _libc_free(v19);
        v22 = v113;
        v23 = &v113[2 * (unsigned int)v114];
        if ( v113 != v23 )
        {
          do
          {
            v24 = 1;
            if ( v22[1] )
              v24 = v22[1];
            v22 += 2;
            *(v22 - 1) = v24;
          }
          while ( v22 != v23 );
          v100 = &v113[2 * (unsigned int)v114];
          if ( v113 != v100 )
          {
            v25 = -1;
            v26 = v113;
            while ( 1 )
            {
              v35 = sub_DFB120(*(_QWORD *)(a1 + 448));
              if ( (((_DWORD)a2 == 1) & (BYTE4(a2) ^ 1)) != 0 )
              {
                if ( (int)sub_23DF0D0(dword_500E148) > 0 )
                  v35 = dword_500E1C8;
              }
              else if ( (int)sub_23DF0D0(dword_500E068) > 0 )
              {
                v35 = dword_500E0E8;
              }
              v42 = v26[1];
              v43 = v105 & 1;
              if ( (v105 & 1) != 0 )
              {
                v27 = 3;
                v28 = &v106;
              }
              else
              {
                v44 = v107;
                v28 = v106;
                if ( !v107 )
                  goto LABEL_103;
                v27 = v107 - 1;
              }
              v29 = (unsigned int)*v26;
              v30 = v27 & (37 * v29);
              v31 = (int *)&v28[v30];
              v32 = *v31;
              if ( (_DWORD)v29 == *v31 )
                goto LABEL_34;
              v71 = 1;
              while ( v32 != -1 )
              {
                v41 = (unsigned int)(v71 + 1);
                v30 = v27 & (v71 + v30);
                v31 = (int *)&v28[v30];
                v32 = *v31;
                if ( (_DWORD)v29 == *v31 )
                  goto LABEL_34;
                v71 = v41;
              }
              if ( v43 )
              {
                v70 = 4;
                goto LABEL_104;
              }
              v44 = v107;
LABEL_103:
              v70 = v44;
LABEL_104:
              v31 = (int *)&v28[v70];
LABEL_34:
              v33 = 4;
              if ( !v43 )
                v33 = v107;
              v34 = (__int64)&v28[v33];
              if ( v31 != (int *)v34 && v31[1] != (_DWORD)v109 )
                v35 -= *(_DWORD *)sub_2AE5BD0((__int64)&v104, v26, v34, v29, v41, (__int64)v28);
              v36 = 0;
              if ( v42 <= v35 )
              {
                _BitScanReverse(&v37, v35 / v42);
                v36 = 0x80000000 >> (v37 ^ 0x1F);
              }
              if ( !byte_500D908 )
                goto LABEL_46;
              v38 = v42 - 1;
              v39 = v35 - 1;
              if ( !v38 )
                v38 = 1;
              if ( v39 < v38 )
              {
                v25 = 0;
                v26 += 2;
                if ( v100 == v26 )
                  goto LABEL_56;
              }
              else
              {
                _BitScanReverse(&v40, v39 / v38);
                v36 = 0x80000000 >> (v40 ^ 0x1F);
LABEL_46:
                if ( v25 > v36 )
                  v25 = v36;
                v26 += 2;
                if ( v100 == v26 )
                  goto LABEL_56;
              }
            }
          }
        }
        v25 = -1;
LABEL_56:
        v6 = sub_DFB730(*(_QWORD *)(a1 + 448));
        if ( BYTE4(a2) )
        {
          if ( (int)sub_23DF0D0(dword_500DEA8) > 0 )
            v6 = dword_500DF28;
          v46 = a2;
          if ( *(_BYTE *)(a1 + 8) )
            v46 = v95 * *(_DWORD *)(a1 + 4);
          v45 = sub_DCF980(*(__int64 **)(*(_QWORD *)(a1 + 424) + 112LL), *(char **)(a1 + 416));
          if ( v45 )
          {
            v47 = (_DWORD)a2 != 0;
            goto LABEL_62;
          }
          if ( !v102 )
            goto LABEL_73;
          v72 = (_DWORD)a2 != 0;
        }
        else
        {
          if ( (_DWORD)a2 == 1 )
          {
            if ( (int)sub_23DF0D0(dword_500DF88) > 0 )
              v6 = dword_500E008;
          }
          else if ( (int)sub_23DF0D0(dword_500DEA8) > 0 )
          {
            v6 = dword_500DF28;
          }
          v45 = sub_DCF980(*(__int64 **)(*(_QWORD *)(a1 + 424) + 112LL), *(char **)(a1 + 416));
          if ( v45 )
          {
            v46 = v95;
            v47 = v95 > 1;
LABEL_62:
            v48 = ((unsigned __int8)sub_2AB31C0(a1, v47) == 0) + v45 - 1;
            v49 = v48 / v46;
            if ( v48 / v46 > v6 )
              v49 = v6;
            if ( v49 > 1 )
            {
              _BitScanReverse(&v82, v49);
              v50 = 0x80000000 >> (v82 ^ 0x1F);
            }
            else
            {
              v50 = 1;
            }
            v51 = v48 / (2 * v46);
            if ( v51 > v6 )
              v51 = v6;
            if ( v51 > 1 )
            {
              _BitScanReverse(&v81, v51);
              v6 = 0x80000000 >> (v81 ^ 0x1F);
            }
            else
            {
              v6 = 1;
            }
            if ( v50 != v6 && v48 % (v46 * v50) == v48 % (v6 * v46) )
              v6 = v50;
            goto LABEL_73;
          }
          if ( !v102 )
            goto LABEL_73;
          v46 = v95;
          v72 = v95 > 1;
        }
        v73 = ((unsigned int)((unsigned __int8)sub_2AB31C0(a1, v72) == 0) + v92 - 1) / (2 * v46);
        if ( v73 > v6 )
          v73 = v6;
        if ( v73 > 1 )
        {
          _BitScanReverse(&v87, v73);
          v6 = 0x80000000 >> (v87 ^ 0x1F);
        }
        else
        {
          v6 = 1;
        }
LABEL_73:
        if ( v6 >= v25 )
        {
          v6 = 1;
          if ( v25 )
            v6 = v25;
        }
        if ( v98 )
        {
          if ( v95 )
            goto LABEL_78;
          goto LABEL_79;
        }
        if ( v95 > 1 )
        {
LABEL_78:
          if ( v94 )
          {
LABEL_127:
            sub_2AB4A90((__int64)&v111);
            sub_2AB4A90((__int64)&v104);
            return v6;
          }
          goto LABEL_79;
        }
        if ( v95 != 1 )
        {
LABEL_79:
          v52 = sub_DFAB90(*(__int64 **)(a1 + 448), v97);
LABEL_80:
          if ( v91 )
          {
            if ( v91 < 0 )
              goto LABEL_82;
          }
          else if ( a3 < (unsigned int)qword_500DC88 )
          {
LABEL_82:
            v53 = (unsigned int)qword_500DC88 / a3;
            if ( v53 )
            {
              _BitScanReverse64(&v54, v53);
              v55 = 0x8000000000000000LL >> ((unsigned __int8)v54 ^ 0x3Fu);
              if ( v6 < (unsigned int)v55 )
                LODWORD(v55) = v6;
              v96 = v55;
            }
            v56 = *(_QWORD *)(a1 + 440);
            v57 = 1;
            v58 = 1;
            v59 = *(_QWORD *)(v56 + 56);
            if ( *(_DWORD *)(v59 + 36) )
              v58 = *(_DWORD *)(v59 + 36);
            v60 = v6 / v58;
            if ( *(_DWORD *)(v59 + 32) )
              v57 = *(_DWORD *)(v59 + 32);
            v61 = v6 / v58;
            v62 = v60;
            v63 = v6 / v57;
            v64 = v6 / v57;
            if ( !v94 )
              goto LABEL_169;
            v65 = *(char **)(v56 + 112);
            v66 = 184LL * *(unsigned int *)(v56 + 120);
            v67 = &v65[v66];
            v68 = (__int64)(0xD37A6F4DE9BD37A7LL * (v66 >> 3)) >> 2;
            if ( v68 )
            {
              v69 = *(char **)(v56 + 112);
              while ( (unsigned int)(*((_DWORD *)v69 + 12) - 17) > 3 )
              {
                if ( (unsigned int)(*((_DWORD *)v69 + 58) - 17) <= 3 )
                {
                  v69 += 184;
                  break;
                }
                if ( (unsigned int)(*((_DWORD *)v69 + 104) - 17) <= 3 )
                {
                  v69 += 368;
                  break;
                }
                if ( (unsigned int)(*((_DWORD *)v69 + 150) - 17) <= 3 )
                {
                  v69 += 552;
                  break;
                }
                v69 += 736;
                if ( v69 == &v65[736 * v68] )
                  goto LABEL_215;
              }
LABEL_149:
              if ( v67 != v69 )
              {
LABEL_188:
                v6 = 1;
                goto LABEL_127;
              }
LABEL_150:
              v83 = **(_QWORD ***)(a1 + 416);
              if ( !v83 )
                goto LABEL_169;
              v84 = 1;
              do
              {
                v83 = (_QWORD *)*v83;
                ++v84;
              }
              while ( v83 );
              if ( v84 <= 1 )
              {
LABEL_169:
                if ( !byte_500DAC8 )
                  goto LABEL_176;
                if ( v62 >= v64 )
                  v64 = v62;
                if ( v96 < v64 )
                {
                  v6 = v64;
                }
                else
                {
LABEL_176:
                  if ( ((v95 == 1) & (v98 ^ 1)) != 0 && v52 )
                  {
                    v6 >>= 1;
                    if ( v6 < v96 )
                      v6 = v96;
                  }
                  else
                  {
                    v6 = v96;
                  }
                }
                goto LABEL_127;
              }
              if ( v68 )
              {
                v85 = &v65[736 * v68];
                while ( !v65[73] )
                {
                  if ( v65[257] )
                  {
                    v65 += 184;
                    goto LABEL_161;
                  }
                  if ( v65[441] )
                  {
                    v65 += 368;
                    goto LABEL_161;
                  }
                  if ( v65[625] )
                  {
                    v65 += 552;
                    goto LABEL_161;
                  }
                  v65 += 736;
                  if ( v85 == v65 )
                    goto LABEL_200;
                }
                goto LABEL_161;
              }
LABEL_200:
              v89 = v67 - v65;
              if ( v67 - v65 != 368 )
              {
                if ( v89 != 552 )
                {
                  if ( v89 != 184 )
                    goto LABEL_162;
LABEL_203:
                  if ( !v65[73] )
                    goto LABEL_162;
LABEL_161:
                  if ( v67 == v65 )
                  {
LABEL_162:
                    v64 = qword_500D748;
                    v86 = v96;
                    if ( v96 > (unsigned int)qword_500D748 )
                      v86 = qword_500D748;
                    if ( v61 > (unsigned int)qword_500D748 )
                      v61 = qword_500D748;
                    v96 = v86;
                    if ( v63 <= (unsigned int)qword_500D748 )
                      v64 = v63;
                    v62 = v61;
                    goto LABEL_169;
                  }
                  goto LABEL_188;
                }
                if ( v65[73] )
                  goto LABEL_161;
                v65 += 184;
              }
              if ( v65[73] )
                goto LABEL_161;
              v65 += 184;
              goto LABEL_203;
            }
            v69 = *(char **)(v56 + 112);
LABEL_215:
            v90 = v67 - v69;
            if ( v67 - v69 != 368 )
            {
              if ( v90 != 552 )
              {
                if ( v90 != 184 )
                  goto LABEL_150;
                goto LABEL_218;
              }
              if ( (unsigned int)(*((_DWORD *)v69 + 12) - 17) <= 3 )
                goto LABEL_149;
              v69 += 184;
            }
            if ( (unsigned int)(*((_DWORD *)v69 + 12) - 17) <= 3 )
              goto LABEL_149;
            v69 += 184;
LABEL_218:
            if ( (unsigned int)(*((_DWORD *)v69 + 12) - 17) > 3 )
              goto LABEL_150;
            goto LABEL_149;
          }
LABEL_125:
          if ( !v52 )
            v6 = 1;
          goto LABEL_127;
        }
        v74 = *(_QWORD *)(a1 + 416);
        v75 = *(char **)(v74 + 40);
        v76 = *(char **)(v74 + 32);
        v77 = (v75 - v76) >> 5;
        if ( v77 > 0 )
        {
          v78 = &v76[32 * v77];
          while ( !(unsigned __int8)sub_31A6C30(*(_QWORD *)(a1 + 440), *(_QWORD *)v76) )
          {
            if ( (unsigned __int8)sub_31A6C30(*(_QWORD *)(a1 + 440), *((_QWORD *)v76 + 1)) )
            {
              v76 += 8;
              break;
            }
            if ( (unsigned __int8)sub_31A6C30(*(_QWORD *)(a1 + 440), *((_QWORD *)v76 + 2)) )
            {
              v76 += 16;
              break;
            }
            if ( (unsigned __int8)sub_31A6C30(*(_QWORD *)(a1 + 440), *((_QWORD *)v76 + 3)) )
            {
              v76 += 24;
              break;
            }
            v76 += 32;
            if ( v76 == v78 )
              goto LABEL_182;
          }
LABEL_137:
          if ( !**(_BYTE **)(*(_QWORD *)(*(_QWORD *)(a1 + 440) + 56LL) + 8LL) )
          {
            v52 = sub_DFAB90(*(__int64 **)(a1 + 448), v97);
            if ( v75 != v76 )
              goto LABEL_125;
            goto LABEL_80;
          }
          goto LABEL_187;
        }
        v78 = *(char **)(v74 + 32);
LABEL_182:
        v88 = v75 - v78;
        if ( v75 - v78 != 16 )
        {
          if ( v88 != 24 )
          {
            v76 = v75;
            if ( v88 != 8 )
              goto LABEL_137;
            goto LABEL_185;
          }
          if ( (unsigned __int8)sub_31A6C30(*(_QWORD *)(a1 + 440), *(_QWORD *)v78) )
          {
LABEL_199:
            v76 = v78;
            goto LABEL_137;
          }
          v78 += 8;
        }
        if ( !(unsigned __int8)sub_31A6C30(*(_QWORD *)(a1 + 440), *(_QWORD *)v78) )
        {
          v78 += 8;
LABEL_185:
          if ( !(unsigned __int8)sub_31A6C30(*(_QWORD *)(a1 + 440), *(_QWORD *)v78) )
          {
            if ( !**(_BYTE **)(*(_QWORD *)(*(_QWORD *)(a1 + 440) + 56LL) + 8LL) )
              goto LABEL_79;
LABEL_187:
            v52 = sub_DFAB90(*(__int64 **)(a1 + 448), v97);
            goto LABEL_125;
          }
          goto LABEL_199;
        }
        goto LABEL_199;
      }
    }
  }
  return 1;
}
