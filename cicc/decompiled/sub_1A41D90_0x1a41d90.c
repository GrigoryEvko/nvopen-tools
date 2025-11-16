// Function: sub_1A41D90
// Address: 0x1a41d90
//
__int64 __fastcall sub_1A41D90(
        __int64 a1,
        _QWORD *a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  unsigned __int64 v10; // r15
  __int64 v11; // rbx
  __int64 result; // rax
  __int64 v13; // rax
  int v14; // r9d
  unsigned __int8 *v15; // rsi
  __int64 v16; // rax
  __int64 v17; // rax
  _QWORD *v18; // r8
  __int64 v19; // r8
  int v20; // r9d
  double v21; // xmm4_8
  double v22; // xmm5_8
  unsigned __int8 *v23; // rsi
  unsigned __int8 *v24; // rdx
  unsigned __int8 *i; // rsi
  __int64 v26; // r13
  __int64 v27; // r15
  __int64 v28; // rbx
  __int64 v29; // rdx
  __int64 v30; // r9
  _QWORD *v31; // rdx
  unsigned __int8 *v32; // rbx
  __int64 v33; // rdx
  __int64 v34; // rcx
  int v35; // r8d
  int v36; // r9d
  _BYTE *v37; // rdx
  _QWORD *v38; // rax
  _QWORD *j; // rdx
  int v40; // r9d
  __int64 v41; // r12
  _QWORD *v42; // rbx
  _QWORD *v43; // rax
  unsigned __int64 v44; // r13
  unsigned __int64 v45; // rdi
  _QWORD *v46; // rbx
  __int64 v47; // rdx
  char v48; // al
  _QWORD *v49; // rcx
  unsigned __int64 v50; // r15
  __int64 **v51; // r13
  int v52; // ebx
  __int64 v53; // r14
  __int64 v54; // rax
  _QWORD *v55; // rax
  unsigned int v56; // r8d
  _QWORD *v57; // rbx
  __int64 v58; // rax
  __int64 *v59; // rax
  __int64 *v60; // rax
  int v61; // r8d
  __int64 *v62; // r10
  __int64 **v63; // rcx
  __int64 **v64; // rax
  __int64 v65; // rdx
  __int64 *v66; // rax
  unsigned __int64 *v67; // r13
  __int64 v68; // rax
  unsigned __int64 v69; // rcx
  __int64 v70; // rsi
  unsigned __int8 *v71; // rsi
  __int64 v72; // rdi
  unsigned __int8 *v73; // rbx
  unsigned __int8 *v74; // r12
  unsigned __int64 v75; // rdi
  __int64 v76; // rax
  __int64 *v77; // rax
  __int64 v78; // [rsp+20h] [rbp-700h]
  int v80; // [rsp+5Ch] [rbp-6C4h]
  __int64 v81; // [rsp+60h] [rbp-6C0h]
  __int64 v82; // [rsp+68h] [rbp-6B8h]
  int v83; // [rsp+70h] [rbp-6B0h]
  unsigned int v84; // [rsp+74h] [rbp-6ACh]
  int v85; // [rsp+78h] [rbp-6A8h]
  _QWORD *v86; // [rsp+78h] [rbp-6A8h]
  __int64 v87; // [rsp+80h] [rbp-6A0h]
  __int64 v88; // [rsp+88h] [rbp-698h]
  _BYTE *v89; // [rsp+90h] [rbp-690h]
  __int64 v90; // [rsp+98h] [rbp-688h]
  unsigned __int8 *v91; // [rsp+A8h] [rbp-678h] BYREF
  _QWORD v92[2]; // [rsp+B0h] [rbp-670h] BYREF
  __m128i v93; // [rsp+C0h] [rbp-660h] BYREF
  __int64 v94; // [rsp+D0h] [rbp-650h]
  _QWORD *v95; // [rsp+E0h] [rbp-640h] BYREF
  __int16 v96; // [rsp+F0h] [rbp-630h]
  __m128 v97; // [rsp+100h] [rbp-620h] BYREF
  __int64 v98; // [rsp+110h] [rbp-610h]
  _BYTE v99[16]; // [rsp+120h] [rbp-600h] BYREF
  __int16 v100; // [rsp+130h] [rbp-5F0h]
  unsigned __int8 *v101; // [rsp+140h] [rbp-5E0h] BYREF
  __int64 v102; // [rsp+148h] [rbp-5D8h]
  unsigned __int64 *v103; // [rsp+150h] [rbp-5D0h]
  __int64 v104; // [rsp+158h] [rbp-5C8h]
  __int64 v105; // [rsp+160h] [rbp-5C0h]
  int v106; // [rsp+168h] [rbp-5B8h]
  __int64 v107; // [rsp+170h] [rbp-5B0h]
  __int64 v108; // [rsp+178h] [rbp-5A8h]
  _BYTE *v109; // [rsp+190h] [rbp-590h] BYREF
  __int64 v110; // [rsp+198h] [rbp-588h]
  _BYTE v111[64]; // [rsp+1A0h] [rbp-580h] BYREF
  __int64 v112[5]; // [rsp+1E0h] [rbp-540h] BYREF
  char *v113; // [rsp+208h] [rbp-518h]
  char v114; // [rsp+218h] [rbp-508h] BYREF
  _QWORD *v115; // [rsp+260h] [rbp-4C0h] BYREF
  __int64 v116; // [rsp+268h] [rbp-4B8h]
  _QWORD v117[3]; // [rsp+270h] [rbp-4B0h] BYREF
  char *v118; // [rsp+288h] [rbp-498h] BYREF
  char v119; // [rsp+298h] [rbp-488h] BYREF
  int v120; // [rsp+2D8h] [rbp-448h]
  unsigned __int8 *v121; // [rsp+2E0h] [rbp-440h] BYREF
  __int64 v122; // [rsp+2E8h] [rbp-438h]
  _WORD v123[536]; // [rsp+2F0h] [rbp-430h] BYREF

  v10 = (unsigned __int64)a2;
  if ( !*(_DWORD *)(a1 + 496) || (result = sub_1A3F5B0(a1, (__int64)a2), (_BYTE)result) )
  {
    v11 = *a2;
    result = 0;
    if ( *(_BYTE *)(*a2 + 8LL) == 16 )
    {
      v13 = sub_16498A0((__int64)a2);
      v15 = (unsigned __int8 *)a2[6];
      v101 = 0;
      v104 = v13;
      v16 = *(_QWORD *)(v10 + 40);
      v105 = 0;
      v102 = v16;
      v106 = 0;
      v107 = 0;
      v108 = 0;
      v103 = (unsigned __int64 *)(v10 + 24);
      v121 = v15;
      if ( v15 )
      {
        sub_1623A60((__int64)&v121, (__int64)v15, 2);
        v101 = v121;
        if ( v121 )
          sub_1623210((__int64)&v121, v121, (__int64)&v101);
      }
      v87 = *(_QWORD *)(v11 + 32);
      v17 = *(_DWORD *)(v10 + 20) & 0xFFFFFFF;
      v85 = *(_DWORD *)(v10 + 20) & 0xFFFFFFF;
      v84 = v17 - 1;
      v18 = *(_QWORD **)(v10 - 24 * v17);
      if ( *(_BYTE *)(*v18 + 8LL) != 16 )
      {
        v123[0] = 257;
        v18 = (_QWORD *)sub_156DA60((__int64 *)&v101, v87, v18, (__int64 *)&v121);
      }
      sub_1A41500((__int64)v112, (_QWORD *)a1, v10, (unsigned __int64)v18, (__int64)v18, v14);
      v121 = (unsigned __int8 *)v123;
      v122 = 0x800000000LL;
      if ( v84 )
      {
        v23 = (unsigned __int8 *)v123;
        v24 = (unsigned __int8 *)v123;
        if ( v84 > 8uLL )
        {
          sub_1A3EFF0((__int64)&v121, v84);
          v23 = v121;
          v24 = &v121[128 * (unsigned __int64)(unsigned int)v122];
        }
        for ( i = &v23[128 * (unsigned __int64)v84]; i != v24; v24 += 128 )
        {
          if ( v24 )
          {
            memset(v24, 0, 0x80u);
            *((_DWORD *)v24 + 13) = 8;
            *((_QWORD *)v24 + 5) = v24 + 56;
          }
        }
        LODWORD(v122) = v84;
        v26 = v10;
        v27 = 0;
        do
        {
          v28 = v27++;
          v29 = v27 - (*(_DWORD *)(v26 + 20) & 0xFFFFFFF);
          v30 = *(_QWORD *)(v26 + 24 * v29);
          if ( *(_BYTE *)(*(_QWORD *)v30 + 8LL) != 16 )
          {
            v31 = *(_QWORD **)(v26 + 24 * v29);
            LOWORD(v117[0]) = 257;
            v30 = sub_156DA60((__int64 *)&v101, v87, v31, (__int64 *)&v115);
          }
          sub_1A41500((__int64)&v115, (_QWORD *)a1, v26, v30, v19, v30);
          v32 = &v121[128 * v28];
          *(_QWORD *)v32 = v115;
          *((_QWORD *)v32 + 1) = v116;
          *((_QWORD *)v32 + 2) = v117[0];
          *((_QWORD *)v32 + 3) = v117[1];
          *((_QWORD *)v32 + 4) = v117[2];
          sub_1A3ED60((__int64)(v32 + 40), &v118, v33, v34, v35, v36);
          *((_DWORD *)v32 + 30) = v120;
          if ( v118 != &v119 )
            _libc_free((unsigned __int64)v118);
        }
        while ( v84 > (unsigned int)v27 );
        v10 = v26;
      }
      v109 = v111;
      v110 = 0x800000000LL;
      if ( (_DWORD)v87 )
      {
        v37 = v111;
        v38 = v111;
        if ( (unsigned int)v87 > 8uLL )
        {
          sub_16CD150((__int64)&v109, v111, (unsigned int)v87, 8, v19, v20);
          v37 = v109;
          v38 = &v109[8 * (unsigned int)v110];
        }
        for ( j = &v37[8 * (unsigned int)v87]; j != v38; ++v38 )
        {
          if ( v38 )
            *v38 = 0;
        }
        LODWORD(v110) = v87;
      }
      v40 = v87;
      if ( (_DWORD)v87 )
      {
        v88 = v10;
        v90 = 0;
        v78 = v84;
        v41 = 8LL * (unsigned int)(v85 - 2) + 8;
        while ( 1 )
        {
          v42 = v117;
          v116 = 0x800000000LL;
          v115 = v117;
          v43 = v117;
          if ( v84 )
          {
            if ( v84 > 8uLL )
            {
              sub_16CD150((__int64)&v115, v117, v84, 8, v19, v40);
              v42 = v115;
              v43 = &v115[(unsigned int)v116];
            }
            if ( &v42[v78] != v43 )
            {
              do
              {
                if ( v43 )
                  *v43 = 0;
                ++v43;
              }
              while ( &v42[v78] != v43 );
              v42 = v115;
            }
            v44 = 0;
            LODWORD(v116) = v84;
            while ( 1 )
            {
              v45 = v44;
              v46 = &v42[v44 / 8];
              v44 += 8LL;
              *v46 = sub_1A3F820((__int64 *)&v121[16 * v45], v90);
              if ( v41 == v44 )
                break;
              v42 = v115;
            }
          }
          LODWORD(v95) = v90;
          v96 = 265;
          v92[0] = sub_1649960(v88);
          v92[1] = v47;
          v93.m128i_i64[0] = (__int64)v92;
          v93.m128i_i64[1] = (__int64)".i";
          v48 = v96;
          LOWORD(v94) = 773;
          if ( (_BYTE)v96 )
          {
            if ( (_BYTE)v96 == 1 )
            {
              a3 = (__m128)_mm_loadu_si128(&v93);
              v97 = a3;
              v98 = v94;
            }
            else
            {
              v49 = v95;
              if ( HIBYTE(v96) != 1 )
              {
                v49 = &v95;
                v48 = 2;
              }
              v97.m128_u64[1] = (unsigned __int64)v49;
              v97.m128_u64[0] = (unsigned __int64)&v93;
              LOBYTE(v98) = 2;
              BYTE1(v98) = v48;
            }
          }
          else
          {
            LOWORD(v98) = 256;
          }
          v50 = (unsigned int)v116;
          v51 = (__int64 **)v115;
          v52 = v116;
          v89 = sub_1A3F820(v112, v90);
          v53 = *(_QWORD *)(v88 + 56);
          v86 = &v109[8 * v90];
          if ( v89[16] > 0x10u )
            goto LABEL_53;
          if ( v50 )
            break;
LABEL_99:
          v99[4] = 0;
          v57 = (_QWORD *)sub_15A2E80(v53, (__int64)v89, v51, v50, 0, (__int64)v99, 0);
LABEL_71:
          *v86 = v57;
          if ( sub_15FA300(v88) )
          {
            v72 = *(_QWORD *)&v109[8 * v90];
            if ( *(_BYTE *)(v72 + 16) == 56 )
              sub_15FA2E0(v72, 1);
          }
          if ( v115 != v117 )
            _libc_free((unsigned __int64)v115);
          if ( (unsigned int)v87 == ++v90 )
          {
            v10 = v88;
            goto LABEL_78;
          }
        }
        v54 = 0;
        while ( *((_BYTE *)v51[v54] + 16) <= 0x10u )
        {
          if ( v50 == ++v54 )
            goto LABEL_99;
        }
LABEL_53:
        v100 = 257;
        if ( !v53 )
        {
          v76 = *(_QWORD *)v89;
          if ( *(_BYTE *)(*(_QWORD *)v89 + 8LL) == 16 )
            v76 = **(_QWORD **)(v76 + 16);
          v53 = *(_QWORD *)(v76 + 24);
        }
        v55 = sub_1648A60(72, (int)v50 + 1);
        v56 = v52 + 1;
        v57 = v55;
        if ( v55 )
        {
          v82 = (__int64)v55;
          v81 = (__int64)&v55[-3 * v56];
          v58 = *(_QWORD *)v89;
          if ( *(_BYTE *)(*(_QWORD *)v89 + 8LL) == 16 )
            v58 = **(_QWORD **)(v58 + 16);
          v80 = v56;
          v83 = *(_DWORD *)(v58 + 8) >> 8;
          v59 = (__int64 *)sub_15F9F50(v53, (__int64)v51, v50);
          v60 = (__int64 *)sub_1646BA0(v59, v83);
          v61 = v80;
          v62 = v60;
          if ( *(_BYTE *)(*(_QWORD *)v89 + 8LL) == 16 )
          {
            v77 = sub_16463B0(v60, *(_QWORD *)(*(_QWORD *)v89 + 32LL));
            v61 = v80;
            v62 = v77;
          }
          else
          {
            v63 = &v51[v50];
            if ( v51 != v63 )
            {
              v64 = v51;
              while ( 1 )
              {
                v65 = **v64;
                if ( *(_BYTE *)(v65 + 8) == 16 )
                  break;
                if ( v63 == ++v64 )
                  goto LABEL_63;
              }
              v66 = sub_16463B0(v62, *(_QWORD *)(v65 + 32));
              v61 = v80;
              v62 = v66;
            }
          }
LABEL_63:
          sub_15F1EA0((__int64)v57, (__int64)v62, 32, v81, v61, 0);
          v57[7] = v53;
          v57[8] = sub_15F9F50(v53, (__int64)v51, v50);
          sub_15F9CE0((__int64)v57, (__int64)v89, (__int64 *)v51, v50, (__int64)v99);
        }
        else
        {
          v82 = 0;
        }
        if ( v102 )
        {
          v67 = v103;
          sub_157E9D0(v102 + 40, (__int64)v57);
          v68 = v57[3];
          v69 = *v67;
          v57[4] = v67;
          v69 &= 0xFFFFFFFFFFFFFFF8LL;
          v57[3] = v69 | v68 & 7;
          *(_QWORD *)(v69 + 8) = v57 + 3;
          *v67 = *v67 & 7 | (unsigned __int64)(v57 + 3);
        }
        sub_164B780(v82, (__int64 *)&v97);
        if ( v101 )
        {
          v91 = v101;
          sub_1623A60((__int64)&v91, (__int64)v101, 2);
          v70 = v57[6];
          if ( v70 )
            sub_161E7C0((__int64)(v57 + 6), v70);
          v71 = v91;
          v57[6] = v91;
          if ( v71 )
            sub_1623210((__int64)&v91, v71, (__int64)(v57 + 6));
        }
        goto LABEL_71;
      }
LABEL_78:
      sub_1A41120(a1, v10, &v109, a3, a4, a5, a6, v21, v22, a9, a10);
      if ( v109 != v111 )
        _libc_free((unsigned __int64)v109);
      v73 = v121;
      v74 = &v121[128 * (unsigned __int64)(unsigned int)v122];
      if ( v121 != v74 )
      {
        do
        {
          v74 -= 128;
          v75 = *((_QWORD *)v74 + 5);
          if ( (unsigned __int8 *)v75 != v74 + 56 )
            _libc_free(v75);
        }
        while ( v73 != v74 );
        v74 = v121;
      }
      if ( v74 != (unsigned __int8 *)v123 )
        _libc_free((unsigned __int64)v74);
      if ( v113 != &v114 )
        _libc_free((unsigned __int64)v113);
      result = 1;
      if ( v101 )
      {
        sub_161E7C0((__int64)&v101, (__int64)v101);
        return 1;
      }
    }
  }
  return result;
}
