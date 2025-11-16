// Function: sub_1483CF0
// Address: 0x1483cf0
//
__int64 __fastcall sub_1483CF0(_QWORD *a1, __int64 a2, __int64 a3, __m128i a4, __m128i a5)
{
  __int64 v5; // r13
  __int64 v8; // rax
  unsigned int v9; // r14d
  __int64 v10; // r15
  __int64 v11; // r15
  __int64 *v12; // rdi
  __int64 v14; // r15
  int v15; // r14d
  int v16; // eax
  __int64 v17; // rdi
  int v18; // edx
  int v19; // r14d
  __int64 v20; // rax
  unsigned int v21; // r14d
  __int64 v22; // rax
  __int16 v23; // ax
  __int64 v24; // r15
  __int64 *v25; // rax
  __int64 *v26; // r8
  __int64 *v27; // r13
  __int64 v28; // rsi
  __int64 v29; // r14
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rax
  int v33; // eax
  __int64 *v34; // r8
  __int64 *v35; // r13
  __int64 v36; // rsi
  __int64 *v37; // r15
  __int64 v38; // rax
  __int64 v39; // r14
  unsigned __int64 v40; // r15
  __int64 v41; // r13
  __int64 v42; // rax
  __int64 v43; // r14
  __int64 v44; // rax
  __int64 *v45; // rcx
  __int64 *v46; // r13
  __int64 v47; // rsi
  __int64 v48; // rax
  __int64 v49; // rax
  __int64 v50; // rax
  __int64 v51; // rax
  __int64 v52; // r15
  __int64 *v53; // rax
  _BYTE *v54; // rsi
  __int64 v55; // rax
  __int64 v56; // r9
  __int64 v57; // rax
  __int64 v58; // r13
  int v59; // eax
  __int64 v60; // rax
  __int64 v61; // [rsp+0h] [rbp-130h]
  __int64 v62; // [rsp+0h] [rbp-130h]
  __int64 v63; // [rsp+8h] [rbp-128h]
  __int64 v64; // [rsp+8h] [rbp-128h]
  __int64 v65; // [rsp+10h] [rbp-120h]
  __int64 v66; // [rsp+10h] [rbp-120h]
  __int64 v67; // [rsp+10h] [rbp-120h]
  __int64 v68; // [rsp+10h] [rbp-120h]
  __int64 v69; // [rsp+10h] [rbp-120h]
  __int64 v70; // [rsp+10h] [rbp-120h]
  __int64 v71; // [rsp+10h] [rbp-120h]
  __int64 v72; // [rsp+18h] [rbp-118h]
  __int64 v73; // [rsp+18h] [rbp-118h]
  __int64 *v74; // [rsp+18h] [rbp-118h]
  __int64 v76; // [rsp+18h] [rbp-118h]
  __int64 v77; // [rsp+20h] [rbp-110h]
  __int64 v78; // [rsp+20h] [rbp-110h]
  __int64 *v79; // [rsp+20h] [rbp-110h]
  __int64 v80; // [rsp+20h] [rbp-110h]
  __int64 v81; // [rsp+28h] [rbp-108h]
  __int64 *v82; // [rsp+28h] [rbp-108h]
  int v83; // [rsp+28h] [rbp-108h]
  __int64 v84; // [rsp+28h] [rbp-108h]
  __int64 v85[2]; // [rsp+30h] [rbp-100h] BYREF
  __int64 *v86; // [rsp+40h] [rbp-F0h] BYREF
  __int64 v87; // [rsp+48h] [rbp-E8h]
  _BYTE v88[32]; // [rsp+50h] [rbp-E0h] BYREF
  __int64 *v89; // [rsp+70h] [rbp-C0h] BYREF
  __int64 v90; // [rsp+78h] [rbp-B8h]
  _BYTE v91[176]; // [rsp+80h] [rbp-B0h] BYREF

  v5 = a2;
  if ( *(_WORD *)(a3 + 24) )
    goto LABEL_5;
  v8 = *(_QWORD *)(a3 + 32);
  v9 = *(_DWORD *)(v8 + 32);
  v10 = v8 + 24;
  if ( v9 > 0x40 )
  {
    if ( (unsigned int)sub_16A57B0(v8 + 24) != v9 - 1 )
    {
      if ( v9 == (unsigned int)sub_16A57B0(v10) )
        goto LABEL_5;
      goto LABEL_11;
    }
    return a2;
  }
  if ( *(_QWORD *)(v8 + 24) == 1 )
    return a2;
  if ( !*(_QWORD *)(v8 + 24) )
    goto LABEL_5;
LABEL_11:
  v14 = sub_1456040(a2);
  v15 = sub_1455840(*(_QWORD *)(a3 + 32) + 24LL);
  v16 = sub_1456C90((__int64)a1, v14);
  v17 = *(_QWORD *)(a3 + 32);
  v18 = v16 - v15;
  v19 = v16 - v15 - 1;
  if ( *(_DWORD *)(v17 + 32) > 0x40u )
  {
    v83 = v18;
    v33 = sub_16A5940(v17 + 24);
    v18 = v83;
    if ( v33 == 1 )
      v18 = v19;
  }
  else
  {
    v20 = *(_QWORD *)(v17 + 24);
    if ( v20 && (v20 & (v20 - 1)) == 0 )
      v18 = v19;
  }
  v21 = v18 + sub_1456C90((__int64)a1, v14);
  v22 = sub_15E0530(a1[3]);
  v77 = sub_1644900(v22, v21);
  v23 = *(_WORD *)(a2 + 24);
  if ( v23 == 7 )
  {
    v81 = sub_13A5BC0((_QWORD *)a2, (__int64)a1);
    if ( !*(_WORD *)(v81 + 24) )
    {
      v24 = *(_QWORD *)(v81 + 32) + 24LL;
      v72 = *(_QWORD *)(a3 + 32) + 24LL;
      sub_16AB0A0(&v89, v24, v72);
      if ( sub_13D01C0((__int64)&v89) )
      {
        v69 = sub_14747F0((__int64)a1, a2, v77, 0);
        v61 = *(_QWORD *)(a2 + 48);
        v63 = sub_14747F0((__int64)a1, v81, v77, 0);
        v44 = sub_14747F0((__int64)a1, **(_QWORD **)(a2 + 32), v77, 0);
        if ( v69 == sub_14799E0((__int64)a1, v44, v63, v61, 0) )
        {
          sub_135E100((__int64 *)&v89);
          v45 = *(__int64 **)(a2 + 32);
          v89 = (__int64 *)v91;
          v90 = 0x400000000LL;
          v79 = &v45[*(_QWORD *)(a2 + 40)];
          if ( v79 != v45 )
          {
            v46 = v45;
            do
            {
              v47 = *v46++;
              v86 = (__int64 *)sub_1483CF0(a1, v47, a3);
              sub_1458920((__int64)&v89, &v86);
            }
            while ( v79 != v46 );
            v5 = a2;
          }
          v48 = sub_14785F0((__int64)a1, &v89, *(_QWORD *)(v5 + 48), 1u);
          v12 = v89;
          v11 = v48;
          if ( v89 == (__int64 *)v91 )
            return v11;
          goto LABEL_7;
        }
      }
      sub_135E100((__int64 *)&v89);
      v25 = *(__int64 **)(a2 + 32);
      if ( !*(_WORD *)(*v25 + 24) )
      {
        v65 = *v25;
        sub_16AB0A0(&v89, v72, v24);
        if ( !sub_13D01C0((__int64)&v89)
          || (v62 = v65,
              v76 = sub_14747F0((__int64)a1, a2, v77, 0),
              v64 = *(_QWORD *)(a2 + 48),
              v70 = sub_14747F0((__int64)a1, v81, v77, 0),
              v49 = sub_14747F0((__int64)a1, **(_QWORD **)(a2 + 32), v77, 0),
              v76 != sub_14799E0((__int64)a1, v49, v70, v64, 0)) )
        {
          sub_135E100((__int64 *)&v89);
          v23 = *(_WORD *)(a2 + 24);
          goto LABEL_19;
        }
        sub_135E100((__int64 *)&v89);
        v71 = *(_QWORD *)(v62 + 32) + 24LL;
        sub_16AB0A0(v85, v71, v24);
        if ( !sub_13A38F0((__int64)v85, 0) )
        {
          v58 = *(_QWORD *)(a2 + 48);
          sub_13A38D0((__int64)&v86, v71);
          sub_16A7590(&v86, v85);
          v59 = v87;
          LODWORD(v87) = 0;
          LODWORD(v90) = v59;
          v89 = v86;
          v60 = sub_145CF40((__int64)a1, (__int64)&v89);
          v5 = sub_14799E0((__int64)a1, v60, v81, v58, 1u);
          sub_135E100((__int64 *)&v89);
          sub_135E100((__int64 *)&v86);
        }
        sub_135E100(v85);
      }
    }
    v23 = *(_WORD *)(v5 + 24);
  }
LABEL_19:
  if ( v23 == 5 )
  {
    v86 = (__int64 *)v88;
    v87 = 0x400000000LL;
    v26 = *(__int64 **)(v5 + 32);
    v82 = &v26[*(_QWORD *)(v5 + 40)];
    if ( v26 != v82 )
    {
      v66 = v5;
      v27 = *(__int64 **)(v5 + 32);
      do
      {
        v28 = *v27++;
        v89 = (__int64 *)sub_14747F0((__int64)a1, v28, v77, 0);
        sub_1458920((__int64)&v86, &v89);
      }
      while ( v82 != v27 );
      v5 = v66;
    }
    v29 = sub_14747F0((__int64)a1, v5, v77, 0);
    if ( v29 == sub_147EE30(a1, &v86, 0, 0, a4, a5) )
    {
      v39 = *(_QWORD *)(v5 + 40);
      if ( (_DWORD)v39 )
      {
        v84 = v5;
        v68 = 8LL * (unsigned int)v39;
        v40 = 0;
        while ( 1 )
        {
          v41 = *(_QWORD *)(*(_QWORD *)(v84 + 32) + v40);
          v42 = sub_1483CF0(a1, v41, a3);
          v43 = v42;
          if ( *(_WORD *)(v42 + 24) != 6 && v41 == sub_13A5B60((__int64)a1, v42, a3, 0, 0) )
            break;
          v40 += 8LL;
          if ( v68 == v40 )
          {
            v5 = v84;
            goto LABEL_25;
          }
        }
        v54 = *(_BYTE **)(v84 + 32);
        v55 = *(_QWORD *)(v84 + 40);
        v89 = (__int64 *)v91;
        v90 = 0x400000000LL;
        sub_145C5B0((__int64)&v89, v54, &v54[8 * v55]);
        sub_1453D20((__int64)&v86, (char **)&v89);
        v56 = v43;
        if ( v89 != (__int64 *)v91 )
        {
          _libc_free((unsigned __int64)v89);
          v56 = v43;
        }
        v86[v40 / 8] = v56;
        v57 = sub_147EE30(a1, &v86, 0, 0, a4, a5);
        v12 = v86;
        v11 = v57;
        if ( v86 == (__int64 *)v88 )
          return v11;
        goto LABEL_7;
      }
    }
LABEL_25:
    if ( v86 != (__int64 *)v88 )
      _libc_free((unsigned __int64)v86);
    v23 = *(_WORD *)(v5 + 24);
  }
  if ( v23 == 6 )
  {
    v30 = *(_QWORD *)(v5 + 40);
    if ( !*(_WORD *)(v30 + 24) )
    {
      LOBYTE(v86) = 0;
      sub_16AA580(&v89, *(_QWORD *)(v30 + 32) + 24LL, *(_QWORD *)(a3 + 32) + 24LL, &v86);
      if ( (_BYTE)v86 )
      {
        v11 = sub_145CF80((__int64)a1, **(_QWORD **)(a3 + 32), 0, 0);
      }
      else
      {
        v50 = sub_145CF40((__int64)a1, (__int64)&v89);
        v11 = sub_1483CF0(a1, *(_QWORD *)(v5 + 32), v50);
      }
      sub_135E100((__int64 *)&v89);
      return v11;
    }
LABEL_5:
    v89 = (__int64 *)v91;
    v90 = 0x2000000000LL;
    sub_16BD3E0(&v89, 6);
    sub_16BD4C0(&v89, v5);
    sub_16BD4C0(&v89, a3);
    v86 = 0;
    v11 = sub_16BDDE0(a1 + 102, &v89, &v86);
    if ( v11 )
    {
      v12 = v89;
      if ( v89 == (__int64 *)v91 )
        return v11;
    }
    else
    {
      v73 = sub_16BD760(&v89, a1 + 108);
      v78 = v31;
      v32 = sub_145CDC0(0x30u, a1 + 108);
      v11 = v32;
      if ( v32 )
      {
        *(_QWORD *)(v32 + 32) = v5;
        *(_QWORD *)v32 = 0;
        *(_QWORD *)(v32 + 8) = v73;
        *(_QWORD *)(v32 + 16) = v78;
        *(_DWORD *)(v32 + 24) = 6;
        *(_QWORD *)(v32 + 40) = a3;
      }
      sub_16BDA20(a1 + 102, v32, v86);
      sub_146DBF0((__int64)a1, v11);
      v12 = v89;
      if ( v89 == (__int64 *)v91 )
        return v11;
    }
LABEL_7:
    _libc_free((unsigned __int64)v12);
    return v11;
  }
  if ( v23 == 4 )
  {
    v89 = (__int64 *)v91;
    v90 = 0x400000000LL;
    v34 = *(__int64 **)(v5 + 32);
    v74 = &v34[*(_QWORD *)(v5 + 40)];
    if ( v34 != v74 )
    {
      v67 = v5;
      v35 = *(__int64 **)(v5 + 32);
      do
      {
        v36 = *v35++;
        v86 = (__int64 *)sub_14747F0((__int64)a1, v36, v77, 0);
        sub_1458920((__int64)&v89, &v86);
      }
      while ( v74 != v35 );
      v5 = v67;
    }
    v37 = (__int64 *)sub_14747F0((__int64)a1, v5, v77, 0);
    if ( v37 == sub_147DD40((__int64)a1, (__int64 *)&v89, 0, 0, a4, a5) )
    {
      LODWORD(v90) = 0;
      v51 = *(_QWORD *)(v5 + 40);
      v52 = 0;
      v80 = 8LL * (unsigned int)v51;
      if ( (_DWORD)v51 )
      {
        do
        {
          v86 = (__int64 *)sub_1483CF0(a1, *(_QWORD *)(*(_QWORD *)(v5 + 32) + v52), a3);
          if ( *((_WORD *)v86 + 12) == 6 )
            break;
          if ( *(_QWORD *)(*(_QWORD *)(v5 + 32) + v52) != sub_13A5B60((__int64)a1, (__int64)v86, a3, 0, 0) )
            break;
          v52 += 8;
          sub_1458920((__int64)&v89, &v86);
        }
        while ( v80 != v52 );
      }
      if ( *(_QWORD *)(v5 + 40) == (unsigned int)v90 )
      {
        v53 = sub_147DD40((__int64)a1, (__int64 *)&v89, 0, 0, a4, a5);
        v12 = v89;
        v11 = (__int64)v53;
        if ( v89 == (__int64 *)v91 )
          return v11;
        goto LABEL_7;
      }
    }
    if ( v89 != (__int64 *)v91 )
      _libc_free((unsigned __int64)v89);
    v23 = *(_WORD *)(v5 + 24);
  }
  if ( v23 )
    goto LABEL_5;
  v38 = sub_15A2C70(*(_QWORD *)(v5 + 32), *(_QWORD *)(a3 + 32), 0);
  return sub_145CE20((__int64)a1, v38);
}
