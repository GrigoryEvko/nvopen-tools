// Function: sub_104B550
// Address: 0x104b550
//
unsigned __int8 *__fastcall sub_104B550(
        __int64 a1,
        unsigned __int8 *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6)
{
  unsigned __int8 *v11; // rax
  unsigned __int8 *v12; // rdx
  bool v13; // cc
  unsigned __int8 *v14; // r11
  unsigned __int8 v16; // al
  __int64 v17; // r10
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // r13
  unsigned __int64 v21; // rax
  int v22; // edx
  unsigned __int64 v23; // rax
  const char *v24; // rax
  int v25; // edi
  __int64 v26; // rdx
  __int64 v27; // rdx
  __int64 v28; // rax
  __int64 v29; // r8
  __int64 v30; // r9
  __int64 v31; // r12
  __int64 v32; // rax
  unsigned __int64 v33; // rdx
  __int64 v34; // rax
  unsigned __int8 *v35; // r11
  __int64 v36; // r12
  __int64 v37; // r14
  __int64 *v38; // r15
  __int64 v39; // rax
  __int64 v40; // r9
  __int64 v41; // rdx
  unsigned __int64 v42; // r8
  unsigned __int64 v43; // rax
  unsigned __int64 v44; // r8
  __int64 v45; // rbx
  __int64 v46; // rdx
  __int64 *v47; // r13
  _QWORD *v48; // rax
  __int64 v49; // r8
  __int64 v50; // r11
  unsigned int v51; // ecx
  __int64 v52; // r10
  __int64 v53; // rsi
  __int64 *v54; // r13
  unsigned int v55; // eax
  __int64 v56; // r8
  __int64 v57; // r9
  __int64 v58; // rax
  unsigned __int8 *v59; // rdi
  __int64 v60; // rax
  __int64 v61; // r13
  unsigned __int64 v62; // rax
  unsigned __int64 v63; // rbx
  __int64 v64; // rbx
  __int64 v65; // rdx
  unsigned __int8 *v66; // rdx
  char v67; // al
  unsigned __int8 v68; // al
  __int64 v69; // rsi
  unsigned __int8 *v70; // rsi
  __int64 *v71; // rdi
  __int64 *v72; // rax
  __int64 v73; // rsi
  int v74; // edx
  char v75; // dl
  __int64 v76; // rax
  __int64 v77; // [rsp+8h] [rbp-148h]
  unsigned __int8 *v78; // [rsp+10h] [rbp-140h]
  int v79; // [rsp+10h] [rbp-140h]
  __int64 v80; // [rsp+18h] [rbp-138h]
  __int64 v81; // [rsp+18h] [rbp-138h]
  int v82; // [rsp+20h] [rbp-130h]
  __int64 v83; // [rsp+20h] [rbp-130h]
  __int64 v84; // [rsp+20h] [rbp-130h]
  __int64 v86; // [rsp+30h] [rbp-120h]
  __int64 v87; // [rsp+30h] [rbp-120h]
  __int64 v88; // [rsp+30h] [rbp-120h]
  __int64 v89; // [rsp+38h] [rbp-118h]
  unsigned __int8 *v90; // [rsp+38h] [rbp-118h]
  __int64 v91; // [rsp+38h] [rbp-118h]
  __int64 v92; // [rsp+38h] [rbp-118h]
  unsigned __int8 *v93; // [rsp+38h] [rbp-118h]
  __int64 v94; // [rsp+38h] [rbp-118h]
  __int64 v95; // [rsp+38h] [rbp-118h]
  unsigned __int8 *v96; // [rsp+38h] [rbp-118h]
  unsigned __int8 *v97; // [rsp+38h] [rbp-118h]
  unsigned __int8 *v98; // [rsp+38h] [rbp-118h]
  unsigned __int8 *v99; // [rsp+38h] [rbp-118h]
  __int64 v100; // [rsp+38h] [rbp-118h]
  __int64 v101; // [rsp+38h] [rbp-118h]
  __int64 v102; // [rsp+48h] [rbp-108h]
  __int64 v103[4]; // [rsp+50h] [rbp-100h] BYREF
  __int16 v104; // [rsp+70h] [rbp-E0h]
  unsigned __int8 *v105[4]; // [rsp+80h] [rbp-D0h] BYREF
  _QWORD *v106; // [rsp+A0h] [rbp-B0h]
  __int64 v107; // [rsp+A8h] [rbp-A8h]
  _QWORD v108[4]; // [rsp+B0h] [rbp-A0h] BYREF
  const char *v109; // [rsp+D0h] [rbp-80h] BYREF
  __int64 v110; // [rsp+D8h] [rbp-78h]
  _QWORD v111[2]; // [rsp+E0h] [rbp-70h] BYREF
  __int16 v112; // [rsp+F0h] [rbp-60h]

  v11 = *(unsigned __int8 **)(a1 + 24);
  v12 = *(unsigned __int8 **)(a1 + 8);
  v105[0] = a2;
  v13 = *a2 <= 0x1Cu;
  v106 = v108;
  v105[3] = v11;
  v105[1] = v12;
  v105[2] = 0;
  v107 = 0x400000000LL;
  if ( !v13 )
  {
    v108[0] = a2;
    LODWORD(v107) = 1;
  }
  v89 = a3;
  v14 = sub_104B4A0(v105, a3, a4, a5, 1);
  if ( !v14 )
  {
    v16 = *a2;
    if ( *a2 > 0x1Cu )
    {
      v17 = a5;
      if ( (unsigned int)v16 - 67 <= 0xC )
      {
        v18 = a3;
        a3 = *((_QWORD *)a2 - 4);
        v19 = sub_104B550(a1, a3, v18, a4, a5, a6);
        v14 = 0;
        v20 = v19;
        if ( !v19 )
          goto LABEL_4;
        v21 = *(_QWORD *)(a4 + 48) & 0xFFFFFFFFFFFFFFF8LL;
        if ( v21 == a4 + 48 )
        {
          v23 = 0;
        }
        else
        {
          if ( !v21 )
            BUG();
          v22 = *(unsigned __int8 *)(v21 - 24);
          v23 = v21 - 24;
          if ( (unsigned int)(v22 - 30) >= 0xB )
            v23 = 0;
        }
        v91 = v23 + 24;
        v24 = sub_BD5D20((__int64)a2);
        v112 = 773;
        v25 = *a2;
        v110 = v26;
        v27 = *((_QWORD *)a2 + 1);
        v109 = v24;
        v111[0] = ".phi.trans.insert";
        v28 = sub_B51D30(v25 - 29, v20, v27, (__int64)&v109, v91, 0);
        a3 = *((_QWORD *)a2 + 6);
        v14 = (unsigned __int8 *)v28;
        v109 = (const char *)a3;
        if ( a3 )
        {
          v92 = v28;
          sub_B96E90((__int64)&v109, a3, 1);
          v14 = (unsigned __int8 *)v92;
          v31 = v92 + 48;
          if ( (const char **)(v92 + 48) == &v109 )
          {
            a3 = (__int64)v109;
            if ( v109 )
            {
              sub_B91220(v92 + 48, (__int64)v109);
              v14 = (unsigned __int8 *)v92;
            }
            goto LABEL_18;
          }
          a3 = *(_QWORD *)(v92 + 48);
          if ( !a3 )
          {
LABEL_49:
            a3 = (__int64)v109;
            *((_QWORD *)v14 + 6) = v109;
            if ( a3 )
            {
              v98 = v14;
              sub_B976B0((__int64)&v109, (unsigned __int8 *)a3, v31);
              v14 = v98;
            }
            goto LABEL_18;
          }
        }
        else
        {
          v31 = v28 + 48;
          if ( (const char **)(v28 + 48) == &v109 || (a3 = *(_QWORD *)(v28 + 48)) == 0 )
          {
LABEL_18:
            v32 = *(unsigned int *)(a6 + 8);
            v33 = v32 + 1;
            if ( v32 + 1 <= (unsigned __int64)*(unsigned int *)(a6 + 12) )
            {
LABEL_19:
              *(_QWORD *)(*(_QWORD *)a6 + 8 * v32) = v14;
              ++*(_DWORD *)(a6 + 8);
              goto LABEL_4;
            }
            v99 = v14;
LABEL_65:
            a3 = a6 + 16;
            sub_C8D5F0(a6, (const void *)(a6 + 16), v33, 8u, v29, v30);
            v32 = *(unsigned int *)(a6 + 8);
            v14 = v99;
            goto LABEL_19;
          }
        }
        v97 = v14;
        sub_B91220(v31, a3);
        v14 = v97;
        goto LABEL_49;
      }
      if ( v16 != 63 )
      {
        if ( !(_BYTE)qword_4F8FB88 || v16 != 42 )
          goto LABEL_4;
        v59 = (a2[7] & 0x40) != 0
            ? (unsigned __int8 *)*((_QWORD *)a2 - 1)
            : &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
        if ( **((_BYTE **)v59 + 4) != 17 )
          goto LABEL_4;
        a3 = *(_QWORD *)v59;
        v60 = sub_104B550(a1, *(_QWORD *)v59, v89, a4, a5, a6);
        v14 = 0;
        v61 = v60;
        if ( !v60 )
          goto LABEL_4;
        v62 = *(_QWORD *)(a4 + 48) & 0xFFFFFFFFFFFFFFF8LL;
        if ( v62 == a4 + 48 )
        {
          v63 = 0;
        }
        else
        {
          if ( !v62 )
            BUG();
          v63 = v62 - 24;
          if ( (unsigned int)*(unsigned __int8 *)(v62 - 24) - 30 >= 0xB )
            v63 = 0;
        }
        v64 = v63 + 24;
        v109 = sub_BD5D20((__int64)a2);
        v112 = 773;
        v110 = v65;
        v111[0] = ".phi.trans.insert";
        if ( (a2[7] & 0x40) != 0 )
          v66 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
        else
          v66 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
        v99 = (unsigned __int8 *)sub_B504D0(13, v61, *((_QWORD *)v66 + 4), (__int64)&v109, v64, 0);
        v67 = sub_B44900((__int64)a2);
        sub_B44850(v99, v67);
        v68 = sub_B448F0((__int64)a2);
        a3 = v68;
        sub_B447F0(v99, v68);
        v32 = *(unsigned int *)(a6 + 8);
        v14 = v99;
        v33 = v32 + 1;
        if ( v32 + 1 <= (unsigned __int64)*(unsigned int *)(a6 + 12) )
          goto LABEL_19;
        goto LABEL_65;
      }
      v109 = (const char *)v111;
      v110 = 0x800000000LL;
      v86 = *((_QWORD *)a2 + 5);
      v34 = 32LL * (*((_DWORD *)a2 + 1) & 0x7FFFFFF);
      if ( (a2[7] & 0x40) != 0 )
      {
        v35 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
        v93 = &v35[v34];
      }
      else
      {
        v93 = a2;
        v35 = &a2[-v34];
      }
      if ( v35 != v93 )
      {
        v78 = a2;
        v36 = a6;
        v37 = v17;
        v38 = (__int64 *)v35;
        while ( 1 )
        {
          a3 = *v38;
          v39 = sub_104B550(a1, *v38, v86, a4, v37, v36);
          if ( !v39 )
            break;
          v41 = (unsigned int)v110;
          v42 = (unsigned int)v110 + 1LL;
          if ( v42 > HIDWORD(v110) )
          {
            v77 = v39;
            sub_C8D5F0((__int64)&v109, v111, (unsigned int)v110 + 1LL, 8u, v42, v40);
            v41 = (unsigned int)v110;
            v39 = v77;
          }
          v38 += 4;
          *(_QWORD *)&v109[8 * v41] = v39;
          LODWORD(v110) = v110 + 1;
          if ( v93 == (unsigned __int8 *)v38 )
          {
            a6 = v36;
            a2 = v78;
            goto LABEL_30;
          }
        }
        v14 = 0;
        goto LABEL_44;
      }
LABEL_30:
      v43 = *(_QWORD *)(a4 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v43 == a4 + 48 )
      {
        v44 = 0;
      }
      else
      {
        if ( !v43 )
          BUG();
        v44 = v43 - 24;
        if ( (unsigned int)*(unsigned __int8 *)(v43 - 24) - 30 >= 0xB )
          v44 = 0;
      }
      v80 = v44 + 24;
      v103[0] = (__int64)sub_BD5D20((__int64)a2);
      v103[2] = (__int64)".phi.trans.insert";
      v45 = (unsigned int)v110 - 1LL;
      v104 = 773;
      v103[1] = v46;
      v47 = (__int64 *)(v109 + 8);
      v82 = v110;
      v94 = *(_QWORD *)v109;
      v87 = *((_QWORD *)a2 + 9);
      v48 = sub_BD2C40(88, v110);
      v49 = v80;
      v50 = (__int64)v48;
      if ( v48 )
      {
        v51 = v82 & 0x7FFFFFF;
        v52 = *(_QWORD *)(v94 + 8);
        if ( (unsigned int)*(unsigned __int8 *)(v52 + 8) - 17 > 1 )
        {
          v71 = &v47[v45];
          if ( v47 != v71 )
          {
            v72 = v47;
            while ( 1 )
            {
              v73 = *(_QWORD *)(*v72 + 8);
              v74 = *(unsigned __int8 *)(v73 + 8);
              if ( v74 == 17 )
              {
                v75 = 0;
                goto LABEL_84;
              }
              if ( v74 == 18 )
                break;
              if ( v71 == ++v72 )
                goto LABEL_36;
            }
            v75 = 1;
LABEL_84:
            BYTE4(v102) = v75;
            v79 = v82 & 0x7FFFFFF;
            LODWORD(v102) = *(_DWORD *)(v73 + 32);
            v81 = v50;
            v84 = v49;
            v76 = sub_BCE1B0((__int64 *)v52, v102);
            v49 = v84;
            v50 = v81;
            v51 = v79;
            v52 = v76;
          }
        }
LABEL_36:
        v83 = v50;
        sub_B44260(v50, v52, 34, v51, v49, 0);
        *(_QWORD *)(v83 + 72) = v87;
        *(_QWORD *)(v83 + 80) = sub_B4DC50(v87, (__int64)v47, v45);
        sub_B4D9A0(v83, v94, v47, v45, (__int64)v103);
        v50 = v83;
      }
      v53 = *((_QWORD *)a2 + 6);
      v54 = (__int64 *)(v50 + 48);
      v103[0] = v53;
      if ( v53 )
      {
        v88 = v50;
        sub_B96E90((__int64)v103, v53, 1);
        v50 = v88;
        if ( v54 == v103 )
        {
          if ( v103[0] )
          {
            sub_B91220((__int64)v103, v103[0]);
            v50 = v88;
          }
          goto LABEL_41;
        }
        v69 = *(_QWORD *)(v88 + 48);
        if ( !v69 )
        {
LABEL_74:
          v70 = (unsigned __int8 *)v103[0];
          *(_QWORD *)(v50 + 48) = v103[0];
          if ( v70 )
          {
            v101 = v50;
            sub_B976B0((__int64)v103, v70, (__int64)v54);
            v50 = v101;
          }
          goto LABEL_41;
        }
      }
      else if ( v54 == v103 || (v69 = *(_QWORD *)(v50 + 48)) == 0 )
      {
LABEL_41:
        v95 = v50;
        v55 = sub_B4DE20((__int64)a2);
        a3 = v55;
        sub_B4DDE0(v95, v55);
        v58 = *(unsigned int *)(a6 + 8);
        v14 = (unsigned __int8 *)v95;
        if ( v58 + 1 > (unsigned __int64)*(unsigned int *)(a6 + 12) )
        {
          a3 = a6 + 16;
          sub_C8D5F0(a6, (const void *)(a6 + 16), v58 + 1, 8u, v56, v57);
          v58 = *(unsigned int *)(a6 + 8);
          v14 = (unsigned __int8 *)v95;
        }
        *(_QWORD *)(*(_QWORD *)a6 + 8 * v58) = v14;
        ++*(_DWORD *)(a6 + 8);
LABEL_44:
        if ( v109 != (const char *)v111 )
        {
          v96 = v14;
          _libc_free(v109, a3);
          v14 = v96;
        }
        goto LABEL_4;
      }
      v100 = v50;
      sub_B91220((__int64)v54, v69);
      v50 = v100;
      goto LABEL_74;
    }
  }
LABEL_4:
  if ( v106 != v108 )
  {
    v90 = v14;
    _libc_free(v106, a3);
    return v90;
  }
  return v14;
}
