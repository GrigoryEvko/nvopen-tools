// Function: sub_199EAC0
// Address: 0x199eac0
//
__int64 **__fastcall sub_199EAC0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __m128i a5, __m128i a6)
{
  __int64 ***v7; // r15
  __int64 **v8; // r13
  __int64 v9; // rax
  __int64 **v10; // rdi
  __int64 **result; // rax
  __int64 **v12; // rbx
  __int64 v13; // rsi
  __int64 v14; // r14
  __int64 ***v15; // rdx
  __int64 *v16; // rbx
  __int64 v17; // rax
  __int64 ***v18; // r8
  __int64 v19; // r14
  __int64 **v20; // r15
  __int64 v21; // rax
  unsigned __int8 *v22; // rsi
  __int64 v23; // rax
  __int64 v24; // rsi
  unsigned int v25; // eax
  _QWORD *v26; // rdi
  __int64 v27; // rax
  unsigned __int64 v28; // r12
  __int64 v29; // rax
  __int64 v30; // rdi
  __int64 v31; // rax
  void *v32; // rdi
  unsigned int v33; // eax
  __int64 v34; // rdx
  __int64 v35; // rax
  __int64 v36; // r14
  __int64 v37; // r15
  __int64 v38; // rax
  __int64 *v39; // r15
  __int64 v40; // rax
  __int64 v41; // rcx
  __int64 *v42; // r15
  __int64 v43; // rsi
  __int64 v44; // rax
  unsigned int v45; // edi
  unsigned __int64 v46; // rcx
  char v47; // al
  __int64 v48; // rax
  __int64 *v49; // r12
  __int64 v50; // rax
  __int64 v51; // rcx
  __int64 v52; // rsi
  unsigned __int8 *v53; // rsi
  __int64 v54; // r15
  __int64 v55; // rdx
  __int64 v56; // r13
  __int64 v57; // rax
  __int64 v58; // rdx
  __int64 v59; // r9
  char v60; // di
  unsigned int v61; // esi
  __int64 v62; // rdx
  __int64 v63; // rax
  __int64 v64; // rcx
  __int64 v65; // rax
  __int64 v66; // rdx
  __int64 v67; // r15
  __int64 v68; // rbx
  __int64 v69; // rbx
  __int64 v70; // r10
  __int64 v71; // rax
  __int64 v72; // rax
  unsigned __int8 *v73; // rsi
  unsigned __int8 *v74; // rsi
  __int64 v75; // rsi
  __int64 v76; // rax
  __int64 v77; // rsi
  unsigned int v78; // eax
  _QWORD *v79; // rdi
  __int64 v80; // r10
  __int64 *v81; // rbx
  __int64 v82; // rcx
  __int64 v83; // rax
  __int64 v84; // rsi
  __int64 v85; // rdx
  unsigned __int8 *v86; // rsi
  __int64 *v87; // [rsp+8h] [rbp-138h]
  __int64 *v88; // [rsp+10h] [rbp-130h]
  __int64 v91; // [rsp+28h] [rbp-118h]
  __int64 *v93; // [rsp+40h] [rbp-100h]
  __int64 v94; // [rsp+40h] [rbp-100h]
  __int64 *v95; // [rsp+48h] [rbp-F8h]
  unsigned __int64 v96; // [rsp+48h] [rbp-F8h]
  __int64 v97; // [rsp+48h] [rbp-F8h]
  __int64 v98; // [rsp+48h] [rbp-F8h]
  __int64 v99; // [rsp+48h] [rbp-F8h]
  __int64 v100; // [rsp+48h] [rbp-F8h]
  __int64 v101; // [rsp+48h] [rbp-F8h]
  __int64 v102; // [rsp+58h] [rbp-E8h]
  __int64 v103; // [rsp+60h] [rbp-E0h]
  __int64 *v104; // [rsp+68h] [rbp-D8h]
  __int64 v105; // [rsp+68h] [rbp-D8h]
  __int64 v106; // [rsp+78h] [rbp-C8h] BYREF
  __int64 v107[2]; // [rsp+80h] [rbp-C0h] BYREF
  char v108; // [rsp+90h] [rbp-B0h]
  char v109; // [rsp+91h] [rbp-AFh]
  unsigned __int8 *v110[2]; // [rsp+A0h] [rbp-A0h] BYREF
  __int16 v111; // [rsp+B0h] [rbp-90h]
  unsigned __int8 *v112; // [rsp+C0h] [rbp-80h] BYREF
  __int64 v113; // [rsp+C8h] [rbp-78h]
  __int64 *v114; // [rsp+D0h] [rbp-70h] BYREF
  __int64 v115; // [rsp+D8h] [rbp-68h]
  __int64 v116; // [rsp+E0h] [rbp-60h]
  int v117; // [rsp+E8h] [rbp-58h]
  __int64 v118; // [rsp+F0h] [rbp-50h]
  __int64 v119; // [rsp+F8h] [rbp-48h]

  v7 = *(__int64 ****)a2;
  v8 = **(__int64 ****)a2;
  v9 = 24LL * (*((_DWORD *)v8 + 5) & 0xFFFFFFF);
  if ( (*((_BYTE *)v8 + 23) & 0x40) != 0 )
  {
    v10 = (__int64 **)*(v8 - 1);
    v8 = &v10[(unsigned __int64)v9 / 8];
  }
  else
  {
    v10 = &v8[v9 / 0xFFFFFFFFFFFFFFF8LL];
  }
  while ( 1 )
  {
    result = sub_1992730(v10, v8, a1[5], a1[1]);
    v12 = result;
    if ( result == v8 )
      return result;
    v13 = (__int64)*result;
    v14 = (__int64)*result;
    if ( *((_BYTE *)*result + 16) == 60 )
      break;
    if ( v7[2] == (__int64 **)sub_146F1B0(a1[1], v13) )
      goto LABEL_9;
LABEL_4:
    if ( v7[2] == (__int64 **)sub_146F1B0(a1[1], v14) )
      goto LABEL_9;
    v10 = v12 + 3;
  }
  v14 = *(_QWORD *)(v13 - 24);
  if ( v7[2] != (__int64 **)sub_146F1B0(a1[1], v13) )
    goto LABEL_4;
LABEL_9:
  v102 = *(_QWORD *)v14;
  v91 = sub_1456E10(a1[1], *(_QWORD *)v14);
  v15 = *(__int64 ****)a2;
  v16 = (__int64 *)(*(_QWORD *)a2 + 24LL);
  v17 = 3LL * *(unsigned int *)(a2 + 8);
  v93 = (__int64 *)(*(_QWORD *)a2 + v17 * 8);
  if ( v93 != v16 )
  {
    v103 = v14;
    v104 = 0;
    v95 = a1;
    while ( 1 )
    {
      v28 = *v16;
      if ( *(_BYTE *)(*v16 + 16) == 77 )
      {
        v31 = sub_13FCB50(v95[5]);
        v28 = sub_157EBA0(v31);
      }
      if ( !sub_14560B0(v16[2]) )
      {
        v29 = sub_147BE00(v95[1], v16[2], v91);
        if ( v104 )
        {
          v30 = v95[1];
          v114 = v104;
          v115 = v29;
          v112 = (unsigned __int8 *)&v114;
          v113 = 0x200000002LL;
          v104 = sub_147DD40(v30, (__int64 *)&v112, 0, 0, a5, a6);
          if ( v112 != (unsigned __int8 *)&v114 )
            _libc_free((unsigned __int64)v112);
        }
        else
        {
          v104 = (__int64 *)v29;
        }
      }
      if ( !v104 || sub_14560B0((__int64)v104) )
      {
        v18 = (__int64 ***)v16[1];
        v19 = v103;
        goto LABEL_14;
      }
      ++*(_QWORD *)(a3 + 152);
      v32 = *(void **)(a3 + 168);
      if ( v32 == *(void **)(a3 + 160) )
        goto LABEL_48;
      v33 = 4 * (*(_DWORD *)(a3 + 180) - *(_DWORD *)(a3 + 184));
      v34 = *(unsigned int *)(a3 + 176);
      if ( v33 < 0x20 )
        v33 = 32;
      if ( (unsigned int)v34 <= v33 )
        break;
      sub_16CC920(a3 + 152);
LABEL_49:
      sub_1940B30(a3 + 88);
      v35 = sub_38767A0(a3, v104, v91, v28);
      v36 = v95[1];
      v37 = sub_145DC80(v36, v35);
      v38 = sub_145DC80(v95[1], v103);
      v115 = v37;
      v114 = (__int64 *)v38;
      v113 = 0x200000002LL;
      v112 = (unsigned __int8 *)&v114;
      v39 = sub_147DD40(v36, (__int64 *)&v112, 0, 0, a5, a6);
      if ( v112 != (unsigned __int8 *)&v114 )
        _libc_free((unsigned __int64)v112);
      v40 = sub_38767A0(a3, v39, v102, v28);
      v18 = (__int64 ***)v16[1];
      v19 = v40;
      if ( *((_WORD *)v104 + 12) )
        goto LABEL_58;
      v87 = (__int64 *)v16[1];
      v42 = (__int64 *)v95[4];
      v88 = (__int64 *)*v16;
      if ( !(unsigned __int8)sub_1994130((__int64)v42, *v16, (__int64)v87, v41, (unsigned int)v18)
        || (unsigned int)sub_1997E70(v104[4] + 24) > 0x40 )
      {
        v18 = (__int64 ***)v16[1];
LABEL_58:
        v103 = v19;
        v104 = 0;
        goto LABEL_14;
      }
      v43 = sub_19927B0((__int64)v42, v88, v87);
      v44 = v104[4];
      v45 = *(_DWORD *)(v44 + 32);
      if ( v45 <= 0x40 )
        v46 = (__int64)(*(_QWORD *)(v44 + 24) << (64 - (unsigned __int8)v45)) >> (64 - (unsigned __int8)v45);
      else
        v46 = **(_QWORD **)(v44 + 24);
      if ( v46 )
      {
        v47 = sub_14A2A90(v42, v43, 0, v46, 1u, 0);
        v18 = (__int64 ***)v16[1];
        if ( v47 )
          goto LABEL_14;
        goto LABEL_58;
      }
      v18 = (__int64 ***)v16[1];
LABEL_14:
      v20 = *v18;
      if ( (__int64 **)v102 != *v18 )
      {
        v21 = sub_16498A0(v28);
        v112 = 0;
        v115 = v21;
        v116 = 0;
        v117 = 0;
        v118 = 0;
        v119 = 0;
        v113 = *(_QWORD *)(v28 + 40);
        v114 = (__int64 *)(v28 + 24);
        v22 = *(unsigned __int8 **)(v28 + 48);
        v110[0] = v22;
        if ( v22 )
        {
          sub_1623A60((__int64)v110, (__int64)v22, 2);
          if ( v112 )
            sub_161E7C0((__int64)&v112, (__int64)v112);
          v112 = v110[0];
          if ( v110[0] )
            sub_1623210((__int64)v110, v110[0], (__int64)&v112);
        }
        v109 = 1;
        v107[0] = (__int64)"lsr.chain";
        v108 = 3;
        if ( v20 != *(__int64 ***)v19 )
        {
          if ( *(_BYTE *)(v19 + 16) <= 0x10u )
          {
            v23 = sub_15A4670((__int64 ***)v19, v20);
            v24 = (__int64)v112;
            v19 = v23;
LABEL_23:
            if ( v24 )
              sub_161E7C0((__int64)&v112, v24);
LABEL_25:
            v18 = (__int64 ***)v16[1];
            goto LABEL_26;
          }
          v111 = 257;
          v48 = sub_15FDF30((_QWORD *)v19, (__int64)v20, (__int64)v110, 0);
          v19 = v48;
          if ( v113 )
          {
            v49 = v114;
            sub_157E9D0(v113 + 40, v48);
            v50 = *(_QWORD *)(v19 + 24);
            v51 = *v49;
            *(_QWORD *)(v19 + 32) = v49;
            v51 &= 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)(v19 + 24) = v51 | v50 & 7;
            *(_QWORD *)(v51 + 8) = v19 + 24;
            *v49 = *v49 & 7 | (v19 + 24);
          }
          sub_164B780(v19, v107);
          if ( !v112 )
            goto LABEL_25;
          v106 = (__int64)v112;
          sub_1623A60((__int64)&v106, (__int64)v112, 2);
          v52 = *(_QWORD *)(v19 + 48);
          if ( v52 )
            sub_161E7C0(v19 + 48, v52);
          v53 = (unsigned __int8 *)v106;
          *(_QWORD *)(v19 + 48) = v106;
          if ( v53 )
            sub_1623210((__int64)&v106, v53, v19 + 48);
        }
        v24 = (__int64)v112;
        goto LABEL_23;
      }
LABEL_26:
      sub_1648780(*v16, (__int64)v18, v19);
      v25 = *(_DWORD *)(a4 + 8);
      if ( v25 >= *(_DWORD *)(a4 + 12) )
      {
        sub_170B450(a4, 0);
        v25 = *(_DWORD *)(a4 + 8);
      }
      v26 = (_QWORD *)(*(_QWORD *)a4 + 24LL * v25);
      if ( v26 )
      {
        v27 = v16[1];
        *v26 = 6;
        v26[1] = 0;
        v26[2] = v27;
        if ( v27 != -8 && v27 != 0 && v27 != -16 )
          sub_164C220((__int64)v26);
        v25 = *(_DWORD *)(a4 + 8);
      }
      v16 += 3;
      *(_DWORD *)(a4 + 8) = v25 + 1;
      if ( v93 == v16 )
      {
        v14 = v103;
        a1 = v95;
        v15 = *(__int64 ****)a2;
        v17 = 3LL * *(unsigned int *)(a2 + 8);
        goto LABEL_41;
      }
    }
    memset(v32, -1, 8 * v34);
LABEL_48:
    *(_QWORD *)(a3 + 180) = 0;
    goto LABEL_49;
  }
LABEL_41:
  result = v15[v17 - 3];
  if ( *((_BYTE *)result + 16) == 77 )
  {
    v54 = sub_157F280(**(_QWORD **)(a1[5] + 32));
    v105 = v55;
    result = (__int64 **)v110;
    if ( v54 != v55 )
    {
      v56 = v54;
      while ( 1 )
      {
        v57 = *(_QWORD *)v56;
        v58 = *(_QWORD *)v14;
        if ( *(_QWORD *)v56 == *(_QWORD *)v14
          || *(_BYTE *)(v57 + 8) == 15
          && *(_BYTE *)(v58 + 8) == 15
          && *(_DWORD *)(v57 + 8) >> 8 == *(_DWORD *)(v58 + 8) >> 8 )
        {
          break;
        }
LABEL_73:
        result = *(__int64 ***)(v56 + 32);
        if ( !result )
          BUG();
        v56 = 0;
        if ( *((_BYTE *)result - 8) == 77 )
          v56 = (__int64)(result - 3);
        if ( v105 == v56 )
          return result;
      }
      v59 = sub_13FCB50(a1[5]);
      v60 = *(_BYTE *)(v56 + 23) & 0x40;
      v61 = *(_DWORD *)(v56 + 20) & 0xFFFFFFF;
      if ( v61 )
      {
        v62 = 24LL * *(unsigned int *)(v56 + 56) + 8;
        v63 = 0;
        while ( 1 )
        {
          v64 = v56 - 24LL * v61;
          if ( v60 )
            v64 = *(_QWORD *)(v56 - 8);
          if ( v59 == *(_QWORD *)(v64 + v62) )
            break;
          ++v63;
          v62 += 8;
          if ( v61 == (_DWORD)v63 )
            goto LABEL_114;
        }
        v65 = 24 * v63;
        if ( v60 )
        {
LABEL_85:
          v66 = *(_QWORD *)(v56 - 8);
          goto LABEL_86;
        }
      }
      else
      {
LABEL_114:
        v65 = 0x17FFFFFFE8LL;
        if ( v60 )
          goto LABEL_85;
      }
      v66 = v56 - 24LL * v61;
LABEL_86:
      v67 = *(_QWORD *)(v66 + v65);
      if ( !v67 )
        BUG();
      if ( *(_BYTE *)(v67 + 16) <= 0x17u )
        goto LABEL_73;
      v68 = sub_146F1B0(a1[1], *(_QWORD *)(v66 + v65));
      if ( v68 != sub_146F1B0(a1[1], v14) )
        goto LABEL_73;
      v69 = *(_QWORD *)v67;
      v70 = v14;
      if ( v102 != *(_QWORD *)v67 )
      {
        v71 = sub_13FCB50(a1[5]);
        v96 = sub_157EBA0(v71);
        v72 = sub_16498A0(v96);
        v112 = 0;
        v115 = v72;
        v116 = 0;
        v117 = 0;
        v118 = 0;
        v119 = 0;
        v113 = *(_QWORD *)(v96 + 40);
        v114 = (__int64 *)(v96 + 24);
        v73 = *(unsigned __int8 **)(v96 + 48);
        v110[0] = v73;
        if ( v73 )
        {
          sub_1623A60((__int64)v110, (__int64)v73, 2);
          if ( v112 )
            sub_161E7C0((__int64)&v112, (__int64)v112);
          v112 = v110[0];
          if ( v110[0] )
            sub_1623210((__int64)v110, v110[0], (__int64)&v112);
        }
        v74 = *(unsigned __int8 **)(v67 + 48);
        v110[0] = v74;
        if ( v74 )
        {
          sub_1623A60((__int64)v110, (__int64)v74, 2);
          v75 = (__int64)v112;
          if ( v112 )
            goto LABEL_97;
LABEL_98:
          v112 = v110[0];
          if ( v110[0] )
            sub_1623210((__int64)v110, v110[0], (__int64)&v112);
        }
        else
        {
          v75 = (__int64)v112;
          if ( v112 )
          {
LABEL_97:
            sub_161E7C0((__int64)&v112, v75);
            goto LABEL_98;
          }
        }
        v109 = 1;
        v107[0] = (__int64)"lsr.chain";
        v108 = 3;
        if ( v69 == *(_QWORD *)v14 )
        {
          v77 = (__int64)v112;
          v70 = v14;
          goto LABEL_103;
        }
        if ( *(_BYTE *)(v14 + 16) <= 0x10u )
        {
          v76 = sub_15A4A70((__int64 ***)v14, v69);
          v77 = (__int64)v112;
          v70 = v76;
          goto LABEL_103;
        }
        v111 = 257;
        v80 = sub_15FDFF0(v14, v69, (__int64)v110, 0);
        if ( v113 )
        {
          v81 = v114;
          v98 = v80;
          sub_157E9D0(v113 + 40, v80);
          v80 = v98;
          v82 = *v81;
          v83 = *(_QWORD *)(v98 + 24);
          *(_QWORD *)(v98 + 32) = v81;
          v82 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v98 + 24) = v82 | v83 & 7;
          *(_QWORD *)(v82 + 8) = v98 + 24;
          *v81 = *v81 & 7 | (v98 + 24);
        }
        v99 = v80;
        sub_164B780(v80, v107);
        v70 = v99;
        if ( v112 )
        {
          v106 = (__int64)v112;
          sub_1623A60((__int64)&v106, (__int64)v112, 2);
          v70 = v99;
          v84 = *(_QWORD *)(v99 + 48);
          v85 = v99 + 48;
          if ( v84 )
          {
            v94 = v99;
            v100 = v99 + 48;
            sub_161E7C0(v100, v84);
            v70 = v94;
            v85 = v100;
          }
          v86 = (unsigned __int8 *)v106;
          *(_QWORD *)(v70 + 48) = v106;
          if ( v86 )
          {
            v101 = v70;
            sub_1623210((__int64)&v106, v86, v85);
            v70 = v101;
          }
          v77 = (__int64)v112;
LABEL_103:
          if ( v77 )
          {
            v97 = v70;
            sub_161E7C0((__int64)&v112, v77);
            v70 = v97;
          }
        }
      }
      sub_1648780(v56, v67, v70);
      v78 = *(_DWORD *)(a4 + 8);
      if ( v78 >= *(_DWORD *)(a4 + 12) )
      {
        sub_170B450(a4, 0);
        v78 = *(_DWORD *)(a4 + 8);
      }
      v79 = (_QWORD *)(*(_QWORD *)a4 + 24LL * v78);
      if ( v79 )
      {
        *v79 = 6;
        v79[1] = 0;
        v79[2] = v67;
        if ( v67 != -16 && v67 != -8 )
          sub_164C220((__int64)v79);
        v78 = *(_DWORD *)(a4 + 8);
      }
      *(_DWORD *)(a4 + 8) = v78 + 1;
      goto LABEL_73;
    }
  }
  return result;
}
