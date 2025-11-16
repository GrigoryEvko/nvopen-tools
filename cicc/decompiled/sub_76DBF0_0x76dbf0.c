// Function: sub_76DBF0
// Address: 0x76dbf0
//
void __fastcall sub_76DBF0(__int64 a1, __int64 a2, unsigned __int64 *a3, _DWORD *a4, _DWORD *a5, int *a6)
{
  unsigned __int64 v10; // rdx
  int *v11; // r14
  _BYTE *v12; // r10
  _QWORD *v13; // rax
  __int64 v14; // rdi
  __int64 v15; // rax
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // rbx
  __m128i *v19; // rax
  __int64 v20; // rdx
  _DWORD *v21; // rcx
  __int64 v22; // r12
  _BYTE *v23; // rdi
  __int64 v24; // r12
  _QWORD *v25; // rax
  __int64 v26; // rax
  _QWORD *v27; // r13
  _BYTE *v28; // r10
  _QWORD *v29; // r12
  _BYTE *v30; // rax
  _BYTE *v31; // rdi
  __int64 v32; // r12
  __int64 v33; // r13
  char v34; // al
  _QWORD *v35; // r12
  _BYTE *v36; // rax
  __int64 j; // rsi
  void *v38; // rax
  _QWORD *v39; // rax
  __int64 v40; // rax
  __int64 v41; // rdi
  __int64 v42; // rcx
  _BYTE *v43; // r10
  _QWORD *v44; // rdi
  __int64 v45; // rax
  _BYTE *v46; // rax
  _QWORD *v47; // r13
  __int64 v48; // rax
  _BYTE *v49; // rax
  __int64 v50; // r10
  _QWORD *v51; // rax
  _QWORD *v52; // rdi
  __int64 i; // rax
  _QWORD *v54; // rax
  __int64 v55; // rdi
  __m128i *v56; // rax
  __int64 v57; // r10
  _DWORD *v58; // rcx
  _QWORD *v59; // rdx
  _BYTE *v60; // r11
  _QWORD *v61; // rax
  _BOOL4 v62; // eax
  __int64 v63; // rdx
  __int64 v64; // r8
  int v65; // eax
  bool v66; // zf
  int v67; // eax
  __int64 v68; // r13
  _QWORD *v69; // rax
  _BYTE *v70; // rax
  __int64 v71; // rax
  _BYTE *v72; // rax
  __m128i *v73; // rax
  __int64 v74; // rax
  _QWORD *v75; // r12
  __int64 v76; // rax
  _BYTE *v77; // [rsp-80h] [rbp-80h]
  __int64 v78; // [rsp-78h] [rbp-78h]
  _DWORD *v79; // [rsp-78h] [rbp-78h]
  _QWORD *v80; // [rsp-70h] [rbp-70h]
  _BYTE *v81; // [rsp-70h] [rbp-70h]
  _BYTE *v82; // [rsp-70h] [rbp-70h]
  __int64 v83; // [rsp-70h] [rbp-70h]
  _DWORD *v84; // [rsp-68h] [rbp-68h]
  _DWORD *v85; // [rsp-68h] [rbp-68h]
  _DWORD *v86; // [rsp-68h] [rbp-68h]
  _DWORD *v87; // [rsp-68h] [rbp-68h]
  int v88; // [rsp-68h] [rbp-68h]
  _DWORD *v89; // [rsp-68h] [rbp-68h]
  _BYTE *v90; // [rsp-68h] [rbp-68h]
  __int64 v91; // [rsp-68h] [rbp-68h]
  _QWORD *v92; // [rsp-68h] [rbp-68h]
  __int64 v93; // [rsp-68h] [rbp-68h]
  _DWORD *v94; // [rsp-60h] [rbp-60h]
  __int64 v95; // [rsp-60h] [rbp-60h]
  _BYTE *v96; // [rsp-60h] [rbp-60h]
  _BYTE *v97; // [rsp-60h] [rbp-60h]
  _BYTE *v98; // [rsp-60h] [rbp-60h]
  __int64 v99; // [rsp-60h] [rbp-60h]
  _DWORD *v100; // [rsp-60h] [rbp-60h]
  __int64 v101; // [rsp-60h] [rbp-60h]
  _DWORD *v102; // [rsp-60h] [rbp-60h]
  __int64 v103; // [rsp-60h] [rbp-60h]
  _BYTE *v104; // [rsp-60h] [rbp-60h]
  __int64 v105; // [rsp-60h] [rbp-60h]
  __int64 v106; // [rsp-60h] [rbp-60h]
  __int64 v107; // [rsp-60h] [rbp-60h]
  __int64 v108; // [rsp-60h] [rbp-60h]
  __int64 v109; // [rsp-58h] [rbp-58h] BYREF
  _QWORD *v110; // [rsp-50h] [rbp-50h]

  if ( !a1 )
    return;
  if ( (*(_BYTE *)(a1 + 41) & 1) != 0 )
  {
LABEL_11:
    *a5 = 0;
LABEL_12:
    *a6 = 1;
    return;
  }
  v10 = *a3;
  v11 = (int *)a2;
  *a3 = v10 + 1;
  if ( unk_4D04378 && v10 > unk_4D04378 )
  {
    *a4 = 1;
    *a6 = 1;
    *a5 = 0;
    return;
  }
  v12 = *(_BYTE **)(a1 + 48);
  if ( v12 )
  {
    a2 = (__int64)a6;
    v94 = a4;
    v13 = sub_73F620(*(const __m128i **)(a1 + 48), a6);
    a4 = v94;
    v12 = v13;
  }
  switch ( *(_BYTE *)(a1 + 40) )
  {
    case 0:
      if ( (unsigned int)(*v11 - 3) <= 2 )
        goto LABEL_76;
      v99 = (__int64)v12;
      sub_76DB10((const __m128i *)a1, (__int64)v11)[3].m128i_i64[0] = (__int64)v12;
      sub_7304E0(v99);
      return;
    case 1:
      if ( v12[24] != 2
        || (v91 = (__int64)a4, v104 = v12, v62 = sub_70FCE0(*((_QWORD *)v12 + 7)), v12 = v104, a4 = (_DWORD *)v91, !v62) )
      {
        if ( (unsigned int)(*v11 - 3) <= 2 )
        {
          v81 = v12;
          v100 = a4;
          sub_7E1790(&v109);
          sub_76DBF0(*(_QWORD *)(a1 + 72), &v109, a3, v100, a5, a6);
          v42 = (__int64)v100;
          v43 = v81;
          v88 = *a6;
          if ( *a6 )
          {
            v88 = 0;
            v77 = 0;
          }
          else
          {
            v44 = v110;
            if ( !v110 )
              v44 = sub_73A830(0, 5u);
            v45 = sub_72CBE0();
            v46 = sub_73E130(v44, v45);
            v43 = v81;
            v42 = (__int64)v100;
            v77 = v46;
          }
          goto LABEL_69;
        }
        v82 = v12;
        v89 = a4;
        sub_7E1760(&v109);
        sub_76DBF0(*(_QWORD *)(a1 + 72), &v109, a3, v89, a5, a6);
        v57 = (__int64)v82;
        v58 = v89;
        if ( *a6 )
        {
          if ( !*(_QWORD *)(a1 + 80) )
            return;
          v60 = 0;
        }
        else
        {
          v59 = *(_QWORD **)(a1 + 80);
          v60 = v110;
          if ( !v59 )
            goto LABEL_96;
        }
        v90 = v60;
        v79 = v58;
        sub_7E1760(&v109);
        sub_76DBF0(*(_QWORD *)(a1 + 80), &v109, a3, v79, a5, a6);
        v60 = v90;
        v57 = (__int64)v82;
        if ( *a6 )
          return;
        v59 = v110;
LABEL_96:
        if ( v59 )
        {
          if ( !v60 )
          {
            v92 = v59;
            v106 = v57;
            v72 = sub_726B30(24);
            v59 = v92;
            v57 = v106;
            v60 = v72;
          }
        }
        else if ( !v60 )
        {
          goto LABEL_98;
        }
        v83 = (__int64)v59;
        v93 = v57;
        v107 = (__int64)v60;
        v73 = sub_76DB10((const __m128i *)a1, (__int64)v11);
        v73[3].m128i_i64[0] = v93;
        v73[4].m128i_i64[1] = v107;
        v73[5].m128i_i64[0] = v83;
        *(_QWORD *)(v107 + 24) = v73;
        if ( v83 )
          *(_QWORD *)(v83 + 24) = v73;
        return;
      }
      v65 = sub_711520(*((_QWORD *)v104 + 7), a2, v63, v91, v64);
      v43 = v104;
      v42 = v91;
      v66 = v65 == 0;
      v67 = *v11;
      if ( v66 )
      {
        if ( (unsigned int)(v67 - 3) <= 2 )
        {
          sub_7E1790(&v109);
          sub_76DBF0(*(_QWORD *)(a1 + 72), &v109, a3, v91, a5, a6);
          if ( !*a6 )
          {
            v75 = v110;
            if ( !v110 )
              v75 = sub_73A830(0, 5u);
            v76 = sub_72CBE0();
            v23 = sub_73E130(v75, v76);
            if ( !*a6 )
              goto LABEL_75;
          }
          return;
        }
        sub_7E1760(&v109);
        sub_76DBF0(*(_QWORD *)(a1 + 72), &v109, a3, v91, a5, a6);
        if ( *a6 )
          return;
        v31 = v110;
        v57 = (__int64)v104;
        if ( v110 )
        {
LABEL_39:
          sub_7E6810(v31, v11, 0);
          return;
        }
LABEL_98:
        v103 = v57;
        sub_7304E0(v57);
        v61 = sub_726B30(0);
        v61[6] = v103;
        *v61 = unk_4D03F38;
        v61[1] = unk_4D03F38;
        sub_7E6810(v61, v11, 0);
        return;
      }
      if ( (unsigned int)(v67 - 3) > 2 )
      {
        if ( !*(_QWORD *)(a1 + 80) )
          return;
        sub_7E1760(&v109);
        sub_76DBF0(*(_QWORD *)(a1 + 80), &v109, a3, v91, a5, a6);
        if ( *a6 )
          return;
        v31 = v110;
        if ( !v110 )
          return;
        goto LABEL_39;
      }
      v88 = 1;
      v77 = 0;
LABEL_69:
      v101 = v42;
      if ( *(_QWORD *)(a1 + 80) )
      {
        v78 = (__int64)v43;
        sub_7E1790(&v109);
        sub_76DBF0(*(_QWORD *)(a1 + 80), &v109, a3, v101, a5, a6);
        if ( *a6 )
          return;
        v47 = v110;
        v48 = sub_72CBE0();
        v49 = sub_73E130(v47, v48);
        v50 = v78;
        v23 = v49;
      }
      else
      {
        v105 = (__int64)v43;
        v68 = sub_72CBE0();
        v69 = sub_73A830(0, 5u);
        v70 = sub_73E110((__int64)v69, v68);
        v50 = v105;
        v23 = v70;
      }
      if ( !*a6 )
      {
        if ( !v88 )
        {
          v108 = v50;
          *(_QWORD *)(v50 + 16) = v77;
          *((_QWORD *)v77 + 2) = v23;
          v74 = sub_72CBE0();
          v23 = sub_73DBF0(0x67u, v74, v108);
          v20 = (4 * *(_BYTE *)(a1 + 41)) & 8;
          v23[25] = v20 | v23[25] & 0xF7;
        }
LABEL_75:
        sub_7E25D0(v23, v11, v20, v21);
      }
      return;
    case 5:
    case 0xC:
      v84 = a4;
      v96 = v12;
      if ( (unsigned int)(*v11 - 3) <= 2 )
        goto LABEL_12;
      sub_7E1760(&v109);
      sub_76DBF0(*(_QWORD *)(a1 + 72), &v109, a3, v84, a5, a6);
      if ( !*a6 )
      {
        v18 = (__int64)v110;
        v19 = sub_76DB10((const __m128i *)a1, (__int64)v11);
        v19[4].m128i_i64[1] = v18;
        v19[3].m128i_i64[0] = (__int64)v96;
        *(_QWORD *)(v18 + 24) = v19;
      }
      return;
    case 6:
      v40 = *(_QWORD *)(*(_QWORD *)(qword_4F08048 + 80) + 72LL);
      if ( !v40 )
        goto LABEL_11;
      v41 = 0;
      while ( 2 )
      {
        if ( *(_QWORD *)(*(_QWORD *)(a1 + 72) + 128LL) != v40 )
        {
          v41 = v40;
          if ( *(_QWORD *)(v40 + 16) )
          {
            v40 = *(_QWORD *)(v40 + 16);
            continue;
          }
          goto LABEL_11;
        }
        break;
      }
      if ( !v41 )
        goto LABEL_11;
      if ( *(_BYTE *)(v41 + 40) == 11 )
        v41 = ((__int64 (*)(void))sub_7E2C20)();
      if ( v41 != a1 )
        goto LABEL_11;
      return;
    case 7:
      if ( (*(_BYTE *)(*(_QWORD *)(a1 + 72) + 120LL) & 0x10) != 0 )
        goto LABEL_11;
      return;
    case 8:
      v14 = *(_QWORD *)(qword_4F08048 + 80);
      while ( 1 )
      {
        v95 = (__int64)v12;
        v15 = sub_7E2C20(v14);
        v14 = v15;
        if ( !v15 )
          goto LABEL_11;
        v12 = (_BYTE *)v95;
        if ( *(_BYTE *)(v15 + 40) != 11 )
        {
          if ( a1 != v15 )
            goto LABEL_11;
          if ( (unsigned int)(*v11 - 3) > 2 )
          {
            if ( !v95 )
              return;
            if ( (*(_BYTE *)(v95 + 25) & 4) != 0 )
            {
              if ( !(unsigned int)sub_731770(v95, 0, v10, (__int64)a4, v16, v17) )
                return;
              v12 = (_BYTE *)v95;
              if ( (*(_BYTE *)(v95 + 25) & 4) != 0 )
              {
                v71 = sub_72CBE0();
                v12 = sub_73E110(v95, v71);
              }
            }
            v54 = (_QWORD *)sub_7E69E0(v12, v11);
            if ( v54 )
            {
              *v54 = unk_4D03F38;
              v54[1] = unk_4D03F38;
            }
            return;
          }
          if ( v95 )
          {
LABEL_76:
            sub_7E25D0(v12, v11, v10, a4);
            return;
          }
          for ( i = *(_QWORD *)(*(_QWORD *)(qword_4F08048 + 32) + 152LL);
                *(_BYTE *)(i + 140) == 12;
                i = *(_QWORD *)(i + 160) )
          {
            ;
          }
          if ( (*(_BYTE *)(*(_QWORD *)(i + 168) + 16LL) & 0x40) != 0 || (unsigned int)sub_8D2600(*(_QWORD *)(i + 160)) )
            return;
          goto LABEL_11;
        }
      }
    case 0xB:
      if ( (unsigned int)(*v11 - 3) > 2 )
      {
        v102 = a4;
        v51 = sub_726B30(11);
        v52 = v51;
        if ( v51 )
        {
          *v51 = unk_4D03F38;
          v51[1] = unk_4D03F38;
        }
        sub_7E6810(v51, v11, 0);
        sub_7E1740(v52);
        v21 = v102;
      }
      else
      {
        v85 = a4;
        sub_7E1790(&v109);
        v21 = v85;
      }
      v22 = *(_QWORD *)(a1 + 72);
      if ( v22 )
      {
        do
        {
          v86 = v21;
          sub_76DBF0(v22, &v109, a3, v21, a5, a6);
          if ( *a6 )
            return;
          v22 = *(_QWORD *)(v22 + 16);
          v21 = v86;
        }
        while ( v22 );
      }
      else if ( *a6 )
      {
        return;
      }
      if ( (unsigned int)(*v11 - 3) > 2 )
        return;
      v23 = v110;
      if ( !v110 )
      {
        v24 = sub_72CBE0();
        v25 = sub_73A830(0, 5u);
        v23 = sub_73E110((__int64)v25, v24);
      }
      goto LABEL_75;
    case 0xD:
      v87 = a4;
      v97 = v12;
      if ( (unsigned int)(*v11 - 3) <= 2 )
        goto LABEL_12;
      sub_7E1760(&v109);
      sub_76DBF0(**(_QWORD **)(a1 + 80), &v109, a3, v87, a5, a6);
      if ( *a6 )
        return;
      v80 = v110;
      sub_7E1760(&v109);
      sub_76DBF0(*(_QWORD *)(a1 + 72), &v109, a3, v87, a5, a6);
      if ( *a6 )
        return;
      v26 = *(_QWORD *)(a1 + 80);
      v27 = v110;
      v28 = v97;
      v29 = *(_QWORD **)(v26 + 8);
      if ( v29 )
      {
        v29 = sub_73F620(*(const __m128i **)(v26 + 8), a6);
        sub_7304E0((__int64)v29);
        v28 = v97;
      }
      v98 = v28;
      v30 = sub_726B30(13);
      v31 = v30;
      if ( v30 )
      {
        *(_QWORD *)v30 = unk_4D03F38;
        *((_QWORD *)v30 + 1) = unk_4D03F38;
      }
      *((_QWORD *)v30 + 6) = v98;
      *((_QWORD *)v30 + 9) = v27;
      v27[3] = v30;
      **((_QWORD **)v30 + 10) = v80;
      if ( v80 )
        v80[3] = v30;
      *(_QWORD *)(*((_QWORD *)v30 + 10) + 8LL) = v29;
      goto LABEL_39;
    case 0x11:
      v32 = *(_QWORD *)(a1 + 72);
      v33 = sub_76DBC0(*(_QWORD *)(v32 + 8));
      if ( (unsigned int)sub_8D3410(*(_QWORD *)(v33 + 120)) )
        goto LABEL_11;
      v34 = *(_BYTE *)(v32 + 48);
      if ( v34 == 2 )
      {
        v55 = *(_QWORD *)(v32 + 56);
        if ( *(_BYTE *)(v55 + 173) == 10 )
          goto LABEL_11;
        v56 = sub_740190(v55, 0, 1u);
        v35 = sub_73A720(v56, 0);
      }
      else
      {
        if ( v34 != 3 )
          goto LABEL_11;
        v35 = sub_73F620(*(const __m128i **)(v32 + 56), a6);
      }
      v36 = sub_731250(v33);
      *((_QWORD *)v36 + 2) = v35;
      for ( j = *(_QWORD *)(v33 + 120); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
        ;
      v38 = sub_73DBF0(0x49u, j, (__int64)v36);
      v39 = (_QWORD *)sub_7E69E0(v38, v11);
      if ( v39 )
      {
        *v39 = unk_4D03F38;
        v39[1] = unk_4D03F38;
      }
      *(_BYTE *)(v33 + 173) |= 8u;
      return;
    case 0x14:
      return;
    case 0x18:
      if ( (unsigned int)(*v11 - 3) > 2 )
        sub_76DB10((const __m128i *)a1, (__int64)v11);
      return;
    default:
      goto LABEL_11;
  }
}
