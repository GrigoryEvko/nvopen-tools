// Function: sub_1CB14E0
// Address: 0x1cb14e0
//
__int64 ***__fastcall sub_1CB14E0(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, char a5)
{
  _QWORD *v5; // r15
  unsigned __int8 *v10; // rsi
  __int64 v11; // r12
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 ***v14; // rax
  __int64 ***v15; // r14
  unsigned int v16; // esi
  int v17; // edx
  __int64 v18; // rax
  _QWORD *v19; // r13
  int v20; // ecx
  __int64 v21; // rdx
  unsigned __int64 *v22; // rbx
  __int64 v23; // rdx
  __int64 v24; // rsi
  unsigned int v25; // ecx
  __int64 v26; // rdx
  __int64 v27; // r8
  unsigned int v28; // r13d
  __int64 *v29; // rax
  __int64 **v30; // rax
  __int64 *v31; // rax
  __int64 **v32; // rax
  __int64 **v34; // rax
  __int64 v35; // rax
  unsigned __int64 *v36; // r13
  __int64 **v37; // rax
  unsigned __int64 v38; // rcx
  __int64 v39; // rsi
  unsigned __int8 *v40; // rsi
  __int64 v41; // rax
  unsigned __int64 *v42; // r13
  __int64 **v43; // rax
  unsigned __int64 v44; // rcx
  __int64 v45; // rsi
  __int64 v46; // rdx
  unsigned __int8 *v47; // rsi
  __int64 v48; // rax
  unsigned __int64 *v49; // rbx
  __int64 **v50; // rax
  unsigned __int64 v51; // rcx
  __int64 v52; // rsi
  unsigned __int8 *v53; // rsi
  __int64 v54; // rdi
  unsigned int v55; // edx
  __int64 v56; // rcx
  int v57; // ecx
  __int64 v58; // rsi
  int v59; // r9d
  _QWORD *v60; // r8
  unsigned int v61; // edx
  __int64 v62; // rdi
  int v63; // edx
  int v64; // r9d
  int v65; // r11d
  _QWORD *v66; // r9
  int v67; // ecx
  int v68; // edx
  int v69; // edx
  __int64 v70; // rsi
  int v71; // r9d
  unsigned int v72; // ecx
  __int64 v73; // rdi
  __int64 v74; // [rsp+8h] [rbp-108h]
  char v75; // [rsp+14h] [rbp-FCh]
  __int64 v76; // [rsp+18h] [rbp-F8h]
  __int64 v77; // [rsp+20h] [rbp-F0h]
  __int16 v78; // [rsp+20h] [rbp-F0h]
  char v79; // [rsp+28h] [rbp-E8h]
  unsigned __int8 *v81; // [rsp+38h] [rbp-D8h] BYREF
  __int64 v82[2]; // [rsp+40h] [rbp-D0h] BYREF
  __int16 v83; // [rsp+50h] [rbp-C0h]
  unsigned __int8 *v84; // [rsp+60h] [rbp-B0h] BYREF
  __int64 v85; // [rsp+68h] [rbp-A8h] BYREF
  __int64 v86; // [rsp+70h] [rbp-A0h]
  __int64 v87; // [rsp+78h] [rbp-98h]
  __int64 v88; // [rsp+80h] [rbp-90h]
  unsigned __int8 *v89; // [rsp+90h] [rbp-80h] BYREF
  __int64 v90; // [rsp+98h] [rbp-78h]
  unsigned __int64 *v91; // [rsp+A0h] [rbp-70h]
  __int64 v92; // [rsp+A8h] [rbp-68h]
  __int64 v93; // [rsp+B0h] [rbp-60h]
  int v94; // [rsp+B8h] [rbp-58h]
  __int64 v95; // [rsp+C0h] [rbp-50h]
  __int64 v96; // [rsp+C8h] [rbp-48h]

  v5 = (_QWORD *)*a2;
  if ( !a4 )
    BUG();
  v77 = *(_QWORD *)(a4 + 16);
  v91 = (unsigned __int64 *)a4;
  v89 = 0;
  v90 = v77;
  v92 = sub_157E9C0(v77);
  v93 = 0;
  v94 = 0;
  v95 = 0;
  v96 = 0;
  if ( v91 != (unsigned __int64 *)(v77 + 40) )
  {
    v10 = *(unsigned __int8 **)(a4 + 24);
    v84 = v10;
    if ( v10 )
    {
      sub_1623A60((__int64)&v84, (__int64)v10, 2);
      if ( v89 )
        sub_161E7C0((__int64)&v89, (__int64)v89);
      v89 = v84;
      if ( v84 )
        sub_1623210((__int64)&v84, v84, (__int64)&v89);
    }
  }
  v11 = *(_QWORD *)(a3 - 24LL * (*(_DWORD *)(a3 + 20) & 0xFFFFFFF));
  v12 = *(unsigned int *)(a1 + 24);
  if ( (_DWORD)v12 )
  {
    v24 = *(_QWORD *)(a1 + 8);
    v25 = (v12 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
    v26 = v24 + 48LL * v25;
    v27 = *(_QWORD *)(v26 + 24);
    if ( v11 == v27 )
    {
LABEL_30:
      if ( v26 != v24 + 48 * v12 )
      {
        v15 = *(__int64 ****)(v26 + 40);
        goto LABEL_32;
      }
    }
    else
    {
      v63 = 1;
      while ( v27 != -8 )
      {
        v64 = v63 + 1;
        v25 = (v12 - 1) & (v63 + v25);
        v26 = v24 + 48LL * v25;
        v27 = *(_QWORD *)(v26 + 24);
        if ( v11 == v27 )
          goto LABEL_30;
        v63 = v64;
      }
    }
  }
  v74 = *(_QWORD *)(v11 + 24);
  v75 = *(_BYTE *)(v11 + 80) & 1;
  v79 = *(_BYTE *)(v11 + 32) & 0xF;
  v76 = *(_QWORD *)(v11 - 24);
  v82[0] = (__int64)sub_1649960(v11);
  v82[1] = v13;
  LOWORD(v86) = 261;
  v84 = (unsigned __int8 *)v82;
  v78 = (*(_BYTE *)(v11 + 33) >> 2) & 7;
  v14 = (__int64 ***)sub_1648A60(88, 1u);
  v15 = v14;
  if ( v14 )
    sub_15E51E0((__int64)v14, (__int64)a2, v74, v75, v79, v76, (__int64)&v84, v11, v78, 1u, 0);
  v85 = 2;
  v86 = 0;
  v87 = v11;
  if ( v11 != -16 && v11 != -8 )
    sub_164C220((__int64)&v85);
  v16 = *(_DWORD *)(a1 + 24);
  v88 = a1;
  v84 = (unsigned __int8 *)&unk_49F8530;
  if ( !v16 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_16;
  }
  v18 = v87;
  v54 = *(_QWORD *)(a1 + 8);
  v55 = (v16 - 1) & (((unsigned int)v87 >> 9) ^ ((unsigned int)v87 >> 4));
  v19 = (_QWORD *)(v54 + 48LL * v55);
  v56 = v19[3];
  if ( v87 != v56 )
  {
    v65 = 1;
    v66 = 0;
    while ( v56 != -8 )
    {
      if ( v56 == -16 && !v66 )
        v66 = v19;
      v55 = (v16 - 1) & (v65 + v55);
      v19 = (_QWORD *)(v54 + 48LL * v55);
      v56 = v19[3];
      if ( v87 == v56 )
        goto LABEL_67;
      ++v65;
    }
    v67 = *(_DWORD *)(a1 + 16);
    if ( v66 )
      v19 = v66;
    ++*(_QWORD *)a1;
    v20 = v67 + 1;
    if ( 4 * v20 < 3 * v16 )
    {
      if ( v16 - *(_DWORD *)(a1 + 20) - v20 > v16 >> 3 )
      {
LABEL_19:
        *(_DWORD *)(a1 + 16) = v20;
        if ( v19[3] == -8 )
        {
          v22 = v19 + 1;
          if ( v18 != -8 )
          {
LABEL_24:
            v19[3] = v18;
            if ( v18 != 0 && v18 != -8 && v18 != -16 )
              sub_1649AC0(v22, v85 & 0xFFFFFFFFFFFFFFF8LL);
            v18 = v87;
          }
        }
        else
        {
          --*(_DWORD *)(a1 + 20);
          v21 = v19[3];
          if ( v18 != v21 )
          {
            v22 = v19 + 1;
            if ( v21 != -8 && v21 != 0 && v21 != -16 )
            {
              sub_1649B30(v19 + 1);
              v18 = v87;
            }
            goto LABEL_24;
          }
        }
        v23 = v88;
        v19[5] = 0;
        v19[4] = v23;
        goto LABEL_67;
      }
      sub_1CB10E0(a1, v16);
      v68 = *(_DWORD *)(a1 + 24);
      if ( !v68 )
        goto LABEL_17;
      v69 = v68 - 1;
      v70 = *(_QWORD *)(a1 + 8);
      v71 = 1;
      v60 = 0;
      v18 = v87;
      v72 = v69 & (((unsigned int)v87 >> 9) ^ ((unsigned int)v87 >> 4));
      v19 = (_QWORD *)(v70 + 48LL * v72);
      v73 = v19[3];
      if ( v87 == v73 )
        goto LABEL_18;
      while ( v73 != -8 )
      {
        if ( !v60 && v73 == -16 )
          v60 = v19;
        v72 = v69 & (v71 + v72);
        v19 = (_QWORD *)(v70 + 48LL * v72);
        v73 = v19[3];
        if ( v87 == v73 )
          goto LABEL_18;
        ++v71;
      }
      goto LABEL_73;
    }
LABEL_16:
    sub_1CB10E0(a1, 2 * v16);
    v17 = *(_DWORD *)(a1 + 24);
    if ( !v17 )
    {
LABEL_17:
      v18 = v87;
      v19 = 0;
LABEL_18:
      v20 = *(_DWORD *)(a1 + 16) + 1;
      goto LABEL_19;
    }
    v57 = v17 - 1;
    v58 = *(_QWORD *)(a1 + 8);
    v59 = 1;
    v60 = 0;
    v18 = v87;
    v61 = (v17 - 1) & (((unsigned int)v87 >> 9) ^ ((unsigned int)v87 >> 4));
    v19 = (_QWORD *)(v58 + 48LL * v61);
    v62 = v19[3];
    if ( v62 == v87 )
      goto LABEL_18;
    while ( v62 != -8 )
    {
      if ( !v60 && v62 == -16 )
        v60 = v19;
      v61 = v57 & (v59 + v61);
      v19 = (_QWORD *)(v58 + 48LL * v61);
      v62 = v19[3];
      if ( v87 == v62 )
        goto LABEL_18;
      ++v59;
    }
LABEL_73:
    if ( v60 )
      v19 = v60;
    goto LABEL_18;
  }
LABEL_67:
  v84 = (unsigned __int8 *)&unk_49EE2B0;
  if ( v18 != 0 && v18 != -8 && v18 != -16 )
    sub_1649B30(&v85);
  v19[5] = v15;
LABEL_32:
  v28 = *((_DWORD *)*v15 + 2);
  v29 = (__int64 *)sub_1643330(v5);
  v30 = (__int64 **)sub_1646BA0(v29, v28 >> 8);
  v82[0] = (__int64)"bcast";
  v83 = 259;
  if ( v30 != *v15 )
  {
    if ( *((_BYTE *)v15 + 16) > 0x10u )
    {
      LOWORD(v86) = 257;
      v41 = sub_15FDBD0(47, (__int64)v15, (__int64)v30, (__int64)&v84, 0);
      v15 = (__int64 ***)v41;
      if ( v90 )
      {
        v42 = v91;
        sub_157E9D0(v90 + 40, v41);
        v43 = v15[3];
        v44 = *v42;
        v15[4] = (__int64 **)v42;
        v44 &= 0xFFFFFFFFFFFFFFF8LL;
        v15[3] = (__int64 **)(v44 | (unsigned __int8)v43 & 7);
        *(_QWORD *)(v44 + 8) = v15 + 3;
        *v42 = *v42 & 7 | (unsigned __int64)(v15 + 3);
      }
      sub_164B780((__int64)v15, v82);
      if ( v89 )
      {
        v81 = v89;
        sub_1623A60((__int64)&v81, (__int64)v89, 2);
        v45 = (__int64)v15[6];
        v46 = (__int64)(v15 + 6);
        if ( v45 )
        {
          sub_161E7C0((__int64)(v15 + 6), v45);
          v46 = (__int64)(v15 + 6);
        }
        v47 = v81;
        v15[6] = (__int64 **)v81;
        if ( v47 )
          sub_1623210((__int64)&v81, v47, v46);
      }
    }
    else
    {
      v15 = (__int64 ***)sub_15A46C0(47, v15, v30, 0);
    }
  }
  v31 = (__int64 *)sub_1643330(v5);
  v32 = (__int64 **)sub_1646BA0(v31, 0);
  v83 = 257;
  if ( v32 != *v15 )
  {
    if ( *((_BYTE *)v15 + 16) > 0x10u )
    {
      LOWORD(v86) = 257;
      v35 = sub_15FDBD0(48, (__int64)v15, (__int64)v32, (__int64)&v84, 0);
      v15 = (__int64 ***)v35;
      if ( v90 )
      {
        v36 = v91;
        sub_157E9D0(v90 + 40, v35);
        v37 = v15[3];
        v38 = *v36;
        v15[4] = (__int64 **)v36;
        v38 &= 0xFFFFFFFFFFFFFFF8LL;
        v15[3] = (__int64 **)(v38 | (unsigned __int8)v37 & 7);
        *(_QWORD *)(v38 + 8) = v15 + 3;
        *v36 = *v36 & 7 | (unsigned __int64)(v15 + 3);
      }
      sub_164B780((__int64)v15, v82);
      if ( v89 )
      {
        v81 = v89;
        sub_1623A60((__int64)&v81, (__int64)v89, 2);
        v39 = (__int64)v15[6];
        if ( v39 )
          sub_161E7C0((__int64)(v15 + 6), v39);
        v40 = v81;
        v15[6] = (__int64 **)v81;
        if ( v40 )
          sub_1623210((__int64)&v81, v40, (__int64)(v15 + 6));
      }
    }
    else
    {
      v15 = (__int64 ***)sub_15A46C0(48, v15, v32, 0);
    }
  }
  if ( a5 )
    goto LABEL_39;
  v34 = (__int64 **)sub_1646BA0(*(__int64 **)(v11 + 24), 0);
  v82[0] = (__int64)"bcast";
  v83 = 259;
  if ( v34 == *v15 )
    goto LABEL_39;
  if ( *((_BYTE *)v15 + 16) <= 0x10u )
  {
    v15 = (__int64 ***)sub_15A46C0(47, v15, v34, 0);
LABEL_39:
    if ( v89 )
      sub_161E7C0((__int64)&v89, (__int64)v89);
    return v15;
  }
  LOWORD(v86) = 257;
  v48 = sub_15FDBD0(47, (__int64)v15, (__int64)v34, (__int64)&v84, 0);
  v15 = (__int64 ***)v48;
  if ( v90 )
  {
    v49 = v91;
    sub_157E9D0(v90 + 40, v48);
    v50 = v15[3];
    v51 = *v49;
    v15[4] = (__int64 **)v49;
    v51 &= 0xFFFFFFFFFFFFFFF8LL;
    v15[3] = (__int64 **)(v51 | (unsigned __int8)v50 & 7);
    *(_QWORD *)(v51 + 8) = v15 + 3;
    *v49 = *v49 & 7 | (unsigned __int64)(v15 + 3);
  }
  sub_164B780((__int64)v15, v82);
  if ( v89 )
  {
    v81 = v89;
    sub_1623A60((__int64)&v81, (__int64)v89, 2);
    v52 = (__int64)v15[6];
    if ( v52 )
      sub_161E7C0((__int64)(v15 + 6), v52);
    v53 = v81;
    v15[6] = (__int64 **)v81;
    if ( v53 )
      sub_1623210((__int64)&v81, v53, (__int64)(v15 + 6));
    goto LABEL_39;
  }
  return v15;
}
