// Function: sub_1A3F820
// Address: 0x1a3f820
//
_BYTE *__fastcall sub_1A3F820(__int64 *a1, unsigned int a2)
{
  __int64 v2; // r12
  __int64 **v3; // rbx
  _BYTE *result; // rax
  __int64 v5; // r15
  __int64 v7; // r13
  _QWORD *v8; // rax
  unsigned int v9; // r9d
  __int64 v10; // rsi
  unsigned __int8 *v11; // rsi
  __int64 v12; // rax
  __int64 i; // rdi
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 *v17; // r13
  __int64 v18; // r15
  _QWORD *v19; // r14
  __int64 v20; // rax
  _QWORD *v21; // rdx
  __int64 v22; // rcx
  __int64 *v23; // rax
  __int64 v24; // rdi
  const char *v25; // rax
  char v26; // al
  __int64 v27; // rdx
  _QWORD *v28; // rcx
  _BYTE *v29; // r15
  __int64 v30; // rax
  __int64 *v31; // rax
  bool v32; // cc
  _QWORD *v33; // r13
  _QWORD *v34; // rax
  unsigned __int64 *v35; // r15
  __int64 v36; // rax
  unsigned __int64 v37; // rcx
  __int64 v38; // rsi
  __int64 v39; // rdx
  unsigned __int8 *v40; // rsi
  const char *v41; // rax
  __int64 v42; // r10
  __int64 v43; // rdx
  __int64 *v44; // r15
  __int64 v45; // rax
  __int64 v46; // rax
  _QWORD *v47; // rax
  _QWORD *v48; // rcx
  __int64 v49; // rax
  __int64 *v50; // rax
  __int64 v51; // rax
  __int64 v52; // rcx
  __int64 *v53; // r10
  __int64 v54; // rax
  unsigned __int64 *v55; // r14
  __int64 v56; // rax
  unsigned __int64 v57; // rcx
  __int64 v58; // rsi
  unsigned __int8 *v59; // rsi
  __int64 *v60; // rax
  __int64 v61; // rax
  unsigned int v62; // r9d
  __int64 v63; // r10
  __int64 v64; // rsi
  __int64 v65; // rax
  __int64 v66; // rsi
  __int64 v67; // rdx
  unsigned __int8 *v68; // rsi
  __int64 v69; // [rsp+8h] [rbp-148h]
  int v70; // [rsp+10h] [rbp-140h]
  __int64 v71; // [rsp+18h] [rbp-138h]
  unsigned int v72; // [rsp+18h] [rbp-138h]
  unsigned int v74; // [rsp+20h] [rbp-130h]
  __int64 v75; // [rsp+20h] [rbp-130h]
  __int64 *v76; // [rsp+20h] [rbp-130h]
  unsigned int v77; // [rsp+20h] [rbp-130h]
  unsigned int v78; // [rsp+20h] [rbp-130h]
  __int64 v79; // [rsp+28h] [rbp-128h]
  unsigned int v80; // [rsp+28h] [rbp-128h]
  _BYTE *v81; // [rsp+28h] [rbp-128h]
  __int64 *v82; // [rsp+28h] [rbp-128h]
  _QWORD *v83; // [rsp+28h] [rbp-128h]
  __int64 **v84; // [rsp+28h] [rbp-128h]
  __int64 v85; // [rsp+28h] [rbp-128h]
  __int64 v86; // [rsp+28h] [rbp-128h]
  __int64 v87; // [rsp+28h] [rbp-128h]
  __int64 *v88; // [rsp+30h] [rbp-120h] BYREF
  __int64 v89; // [rsp+38h] [rbp-118h] BYREF
  const char *v90; // [rsp+40h] [rbp-110h] BYREF
  __int64 v91; // [rsp+48h] [rbp-108h]
  __m128i v92; // [rsp+50h] [rbp-100h] BYREF
  __int64 v93; // [rsp+60h] [rbp-F0h]
  _QWORD v94[2]; // [rsp+70h] [rbp-E0h] BYREF
  __int16 v95; // [rsp+80h] [rbp-D0h]
  __m128i v96; // [rsp+90h] [rbp-C0h] BYREF
  __int64 v97; // [rsp+A0h] [rbp-B0h]
  _QWORD v98[2]; // [rsp+B0h] [rbp-A0h] BYREF
  __int16 v99; // [rsp+C0h] [rbp-90h]
  unsigned __int8 *v100; // [rsp+D0h] [rbp-80h] BYREF
  __int64 v101; // [rsp+D8h] [rbp-78h]
  __int64 *v102; // [rsp+E0h] [rbp-70h]
  _QWORD *v103; // [rsp+E8h] [rbp-68h]
  __int64 v104; // [rsp+F0h] [rbp-60h]
  int v105; // [rsp+F8h] [rbp-58h]
  __int64 v106; // [rsp+100h] [rbp-50h]
  __int64 v107; // [rsp+108h] [rbp-48h]

  v2 = a2;
  v3 = (__int64 **)a1[3];
  if ( !v3 )
    v3 = (__int64 **)(a1 + 5);
  result = (_BYTE *)(*v3)[a2];
  if ( !result )
  {
    v5 = *a1;
    v7 = a2;
    v79 = a1[1];
    v8 = (_QWORD *)sub_157E9C0(*a1);
    v101 = v5;
    v100 = 0;
    v9 = a2;
    v103 = v8;
    v104 = 0;
    v105 = 0;
    v106 = 0;
    v107 = 0;
    v102 = (__int64 *)v79;
    if ( v79 != v5 + 40 )
    {
      if ( !v79 )
        BUG();
      v10 = *(_QWORD *)(v79 + 24);
      v98[0] = v10;
      if ( v10 )
      {
        sub_1623A60((__int64)v98, v10, 2);
        v9 = a2;
        if ( v100 )
        {
          sub_161E7C0((__int64)&v100, (__int64)v100);
          v11 = (unsigned __int8 *)v98[0];
          v9 = a2;
        }
        else
        {
          v11 = (unsigned __int8 *)v98[0];
        }
        v100 = v11;
        if ( v11 )
        {
          v80 = v9;
          sub_1623210((__int64)v98, v11, (__int64)&v100);
          v9 = v80;
        }
      }
    }
    v12 = a1[4];
    if ( v12 )
    {
      if ( !**v3 )
      {
        v74 = v9;
        v84 = (__int64 **)sub_1646BA0(**(__int64 ***)(*(_QWORD *)(v12 + 24) + 16LL), *(_DWORD *)(v12 + 8) >> 8);
        v41 = sub_1649960(a1[2]);
        v42 = a1[2];
        v94[0] = v41;
        v9 = v74;
        LOWORD(v97) = 773;
        v96.m128i_i64[0] = (__int64)v94;
        v94[1] = v43;
        v96.m128i_i64[1] = (__int64)".i0";
        v44 = *v3;
        if ( v84 != *(__int64 ***)v42 )
        {
          if ( *(_BYTE *)(v42 + 16) > 0x10u )
          {
            v99 = 257;
            v61 = sub_15FDBD0(47, v42, (__int64)v84, (__int64)v98, 0);
            v62 = v74;
            v63 = v61;
            if ( v101 )
            {
              v72 = v74;
              v85 = v61;
              v76 = v102;
              sub_157E9D0(v101 + 40, v61);
              v63 = v85;
              v62 = v72;
              v64 = *v76;
              v65 = *(_QWORD *)(v85 + 24);
              *(_QWORD *)(v85 + 32) = v76;
              v64 &= 0xFFFFFFFFFFFFFFF8LL;
              *(_QWORD *)(v85 + 24) = v64 | v65 & 7;
              *(_QWORD *)(v64 + 8) = v85 + 24;
              *v76 = *v76 & 7 | (v85 + 24);
            }
            v77 = v62;
            v86 = v63;
            sub_164B780(v63, v96.m128i_i64);
            v42 = v86;
            v9 = v77;
            if ( v100 )
            {
              v92.m128i_i64[0] = (__int64)v100;
              sub_1623A60((__int64)&v92, (__int64)v100, 2);
              v42 = v86;
              v9 = v77;
              v66 = *(_QWORD *)(v86 + 48);
              v67 = v86 + 48;
              if ( v66 )
              {
                sub_161E7C0(v86 + 48, v66);
                v9 = v77;
                v42 = v86;
                v67 = v86 + 48;
              }
              v68 = (unsigned __int8 *)v92.m128i_i64[0];
              *(_QWORD *)(v42 + 48) = v92.m128i_i64[0];
              if ( v68 )
              {
                v78 = v9;
                v87 = v42;
                sub_1623210((__int64)&v92, v68, v67);
                v9 = v78;
                v42 = v87;
              }
            }
          }
          else
          {
            v45 = sub_15A46C0(47, (__int64 ***)v42, v84, 0);
            v9 = v74;
            v42 = v45;
          }
        }
        *v44 = v42;
      }
      if ( v9 )
      {
        v24 = a1[2];
        LODWORD(v94[0]) = v9;
        v95 = 265;
        v25 = sub_1649960(v24);
        LOWORD(v93) = 773;
        v90 = v25;
        v92.m128i_i64[0] = (__int64)&v90;
        v92.m128i_i64[1] = (__int64)".i";
        v26 = v95;
        v91 = v27;
        if ( (_BYTE)v95 )
        {
          if ( (_BYTE)v95 == 1 )
          {
            v96 = _mm_loadu_si128(&v92);
            v97 = v93;
          }
          else
          {
            v28 = (_QWORD *)v94[0];
            if ( HIBYTE(v95) != 1 )
            {
              v28 = v94;
              v26 = 2;
            }
            v96.m128i_i64[1] = (__int64)v28;
            v96.m128i_i64[0] = (__int64)&v92;
            LOBYTE(v97) = 2;
            BYTE1(v97) = v26;
          }
        }
        else
        {
          LOWORD(v97) = 256;
        }
        v29 = (_BYTE *)**v3;
        v82 = &(*v3)[v7];
        v30 = sub_1643350(v103);
        v31 = (__int64 *)sub_159C470(v30, v2, 0);
        v32 = v29[16] <= 0x10u;
        v88 = v31;
        if ( v32 )
        {
          BYTE4(v98[0]) = 0;
          v33 = (_QWORD *)sub_15A2E80(0, (__int64)v29, &v88, 1u, 0, (__int64)v98, 0);
        }
        else
        {
          v99 = 257;
          v46 = *(_QWORD *)v29;
          if ( *(_BYTE *)(*(_QWORD *)v29 + 8LL) == 16 )
            v46 = **(_QWORD **)(v46 + 16);
          v75 = *(_QWORD *)(v46 + 24);
          v47 = sub_1648A60(72, 2u);
          v33 = v47;
          if ( v47 )
          {
            v71 = (__int64)v47;
            v48 = v47 - 6;
            v49 = *(_QWORD *)v29;
            if ( *(_BYTE *)(*(_QWORD *)v29 + 8LL) == 16 )
              v49 = **(_QWORD **)(v49 + 16);
            v69 = (__int64)v48;
            v70 = *(_DWORD *)(v49 + 8) >> 8;
            v50 = (__int64 *)sub_15F9F50(v75, (__int64)&v88, 1);
            v51 = sub_1646BA0(v50, v70);
            v52 = v69;
            v53 = (__int64 *)v51;
            v54 = *(_QWORD *)v29;
            if ( *(_BYTE *)(*(_QWORD *)v29 + 8LL) == 16 || (v54 = *v88, *(_BYTE *)(*v88 + 8) == 16) )
            {
              v60 = sub_16463B0(v53, *(_QWORD *)(v54 + 32));
              v52 = v69;
              v53 = v60;
            }
            sub_15F1EA0((__int64)v33, (__int64)v53, 32, v52, 2, 0);
            v33[7] = v75;
            v33[8] = sub_15F9F50(v75, (__int64)&v88, 1);
            sub_15F9CE0((__int64)v33, (__int64)v29, (__int64 *)&v88, 1, (__int64)v98);
          }
          else
          {
            v71 = 0;
          }
          if ( v101 )
          {
            v55 = (unsigned __int64 *)v102;
            sub_157E9D0(v101 + 40, (__int64)v33);
            v56 = v33[3];
            v57 = *v55;
            v33[4] = v55;
            v57 &= 0xFFFFFFFFFFFFFFF8LL;
            v33[3] = v57 | v56 & 7;
            *(_QWORD *)(v57 + 8) = v33 + 3;
            *v55 = *v55 & 7 | (unsigned __int64)(v33 + 3);
          }
          sub_164B780(v71, v96.m128i_i64);
          if ( v100 )
          {
            v89 = (__int64)v100;
            sub_1623A60((__int64)&v89, (__int64)v100, 2);
            v58 = v33[6];
            if ( v58 )
              sub_161E7C0((__int64)(v33 + 6), v58);
            v59 = (unsigned __int8 *)v89;
            v33[6] = v89;
            if ( v59 )
              sub_1623210((__int64)&v89, v59, (__int64)(v33 + 6));
          }
        }
        *v82 = (__int64)v33;
      }
    }
    else
    {
LABEL_19:
      for ( i = a1[2]; *(_BYTE *)(i + 16) == 84; i = v22 )
      {
        v20 = *(_QWORD *)(i - 24);
        if ( *(_BYTE *)(v20 + 16) != 13 )
          break;
        v21 = *(_QWORD **)(v20 + 24);
        if ( *(_DWORD *)(v20 + 32) > 0x40u )
          v21 = (_QWORD *)*v21;
        v22 = *(_QWORD *)(i - 72);
        a1[2] = v22;
        if ( v9 == (_DWORD)v21 )
        {
          (*v3)[(unsigned int)v21] = *(_QWORD *)(i - 48);
          result = (_BYTE *)(*v3)[(unsigned int)v21];
          goto LABEL_16;
        }
        v23 = &(*v3)[(unsigned int)v21];
        if ( !*v23 )
        {
          *v23 = *(_QWORD *)(i - 48);
          goto LABEL_19;
        }
      }
      LODWORD(v94[0]) = v9;
      v95 = 265;
      v90 = sub_1649960(i);
      v92.m128i_i64[0] = (__int64)&v90;
      v92.m128i_i64[1] = (__int64)".i";
      v91 = v14;
      LOWORD(v93) = 773;
      v96.m128i_i64[1] = v94[0];
      v96.m128i_i64[0] = (__int64)&v92;
      LOWORD(v97) = 2306;
      v15 = sub_1643350(v103);
      v16 = sub_159C470(v15, v2, 0);
      v17 = &(*v3)[v7];
      v18 = v16;
      if ( *(_BYTE *)(a1[2] + 16) > 0x10u || *(_BYTE *)(v16 + 16) > 0x10u )
      {
        v83 = (_QWORD *)a1[2];
        v99 = 257;
        v34 = sub_1648A60(56, 2u);
        v19 = v34;
        if ( v34 )
          sub_15FA320((__int64)v34, v83, v18, (__int64)v98, 0);
        if ( v101 )
        {
          v35 = (unsigned __int64 *)v102;
          sub_157E9D0(v101 + 40, (__int64)v19);
          v36 = v19[3];
          v37 = *v35;
          v19[4] = v35;
          v37 &= 0xFFFFFFFFFFFFFFF8LL;
          v19[3] = v37 | v36 & 7;
          *(_QWORD *)(v37 + 8) = v19 + 3;
          *v35 = *v35 & 7 | (unsigned __int64)(v19 + 3);
        }
        sub_164B780((__int64)v19, v96.m128i_i64);
        if ( v100 )
        {
          v89 = (__int64)v100;
          sub_1623A60((__int64)&v89, (__int64)v100, 2);
          v38 = v19[6];
          v39 = (__int64)(v19 + 6);
          if ( v38 )
          {
            sub_161E7C0((__int64)(v19 + 6), v38);
            v39 = (__int64)(v19 + 6);
          }
          v40 = (unsigned __int8 *)v89;
          v19[6] = v89;
          if ( v40 )
            sub_1623210((__int64)&v89, v40, v39);
        }
      }
      else
      {
        v19 = (_QWORD *)sub_15A37D0((_BYTE *)a1[2], v16, 0);
      }
      *v17 = (__int64)v19;
    }
    result = (_BYTE *)(*v3)[v2];
LABEL_16:
    if ( v100 )
    {
      v81 = result;
      sub_161E7C0((__int64)&v100, (__int64)v100);
      return v81;
    }
  }
  return result;
}
