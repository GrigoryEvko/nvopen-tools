// Function: sub_1E3BF90
// Address: 0x1e3bf90
//
__int64 __fastcall sub_1E3BF90(_QWORD *a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v6; // r13
  __int64 v7; // rax
  __int64 *v8; // r15
  __int64 v9; // rax
  __int64 v10; // r15
  _BYTE *v11; // rsi
  __int64 v12; // r15
  _QWORD *v13; // rax
  __int64 v14; // rbx
  __int64 v15; // rax
  __int64 v16; // r13
  _QWORD *v17; // rax
  _QWORD *v18; // rbx
  unsigned __int64 *v19; // r13
  __int64 v20; // rax
  unsigned __int64 v21; // rcx
  __int64 v22; // rsi
  unsigned __int8 *v23; // rsi
  __int64 *v24; // rdx
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rax
  __int64 v28; // r15
  __int64 (*v29)(void); // rax
  __int64 *v30; // rbx
  __int64 v31; // rax
  __int64 v32; // rdx
  _DWORD *v33; // rbx
  unsigned int v34; // esi
  int *v35; // rax
  int v36; // r10d
  _QWORD *v37; // rax
  __int64 v38; // r13
  __int64 v39; // rsi
  unsigned __int8 *v40; // rsi
  __int64 v41; // rdx
  __int64 v42; // rdx
  __int64 v43; // rdi
  __int64 v44; // r14
  __int64 v45; // r13
  __int64 *v46; // rax
  __int64 v47; // rax
  __int64 v48; // rdx
  __int64 v49; // rsi
  int v51; // eax
  int v52; // r9d
  __int64 v53; // rbx
  __int64 v54; // rsi
  __int64 *v55; // r13
  __int64 *v56; // rax
  __int64 v57; // r13
  __int64 v58; // rax
  __int64 v59; // r14
  const char *v60; // rax
  __int64 v61; // rdx
  __int64 v62; // r14
  __int64 v63; // [rsp+28h] [rbp-498h]
  __int64 *v67; // [rsp+40h] [rbp-480h]
  __int64 v68; // [rsp+48h] [rbp-478h]
  __int64 v69; // [rsp+48h] [rbp-478h]
  __int64 v70; // [rsp+50h] [rbp-470h]
  __int64 v71; // [rsp+58h] [rbp-468h]
  __int64 v72; // [rsp+60h] [rbp-460h]
  __int64 v73; // [rsp+60h] [rbp-460h]
  _DWORD *i; // [rsp+68h] [rbp-458h]
  __int64 *v75; // [rsp+68h] [rbp-458h]
  __int64 v76; // [rsp+78h] [rbp-448h] BYREF
  __int64 v77; // [rsp+80h] [rbp-440h] BYREF
  __int64 v78; // [rsp+88h] [rbp-438h]
  __int64 v79; // [rsp+90h] [rbp-430h]
  int v80; // [rsp+98h] [rbp-428h]
  _QWORD v81[2]; // [rsp+A0h] [rbp-420h] BYREF
  _QWORD v82[2]; // [rsp+B0h] [rbp-410h] BYREF
  unsigned __int8 *v83; // [rsp+C0h] [rbp-400h] BYREF
  __int64 v84; // [rsp+C8h] [rbp-3F8h]
  __int64 v85; // [rsp+D0h] [rbp-3F0h]
  __int64 v86; // [rsp+D8h] [rbp-3E8h]
  int v87; // [rsp+E0h] [rbp-3E0h]
  __int64 *v88; // [rsp+E8h] [rbp-3D8h]
  unsigned __int8 *v89; // [rsp+F0h] [rbp-3D0h] BYREF
  __int64 v90; // [rsp+F8h] [rbp-3C8h]
  _QWORD v91[3]; // [rsp+100h] [rbp-3C0h] BYREF
  int v92; // [rsp+118h] [rbp-3A8h]
  __int64 v93; // [rsp+120h] [rbp-3A0h]
  __int64 v94; // [rsp+128h] [rbp-398h]
  __int64 (__fastcall **v95)(); // [rsp+140h] [rbp-380h] BYREF
  _QWORD v96[3]; // [rsp+148h] [rbp-378h] BYREF
  unsigned __int64 v97; // [rsp+160h] [rbp-360h]
  __int64 v98; // [rsp+168h] [rbp-358h]
  unsigned __int64 v99; // [rsp+170h] [rbp-350h]
  __int64 v100; // [rsp+178h] [rbp-348h]
  char v101[8]; // [rsp+180h] [rbp-340h] BYREF
  int v102; // [rsp+188h] [rbp-338h]
  _QWORD v103[2]; // [rsp+190h] [rbp-330h] BYREF
  _QWORD v104[2]; // [rsp+1A0h] [rbp-320h] BYREF
  _QWORD v105[28]; // [rsp+1B0h] [rbp-310h] BYREF
  __int16 v106; // [rsp+290h] [rbp-230h]
  __int64 v107; // [rsp+298h] [rbp-228h]
  __int64 v108; // [rsp+2A0h] [rbp-220h]
  __int64 v109; // [rsp+2A8h] [rbp-218h]
  __int64 v110; // [rsp+2B0h] [rbp-210h]
  char *v111; // [rsp+2C0h] [rbp-200h] BYREF
  __int64 v112; // [rsp+2C8h] [rbp-1F8h]
  _QWORD v113[62]; // [rsp+2D0h] [rbp-1F0h] BYREF

  sub_222DF20(v105);
  v106 = 0;
  v105[27] = 0;
  v105[0] = off_4A06798;
  v107 = 0;
  v108 = 0;
  v109 = 0;
  v95 = (__int64 (__fastcall **)())qword_4A071C8;
  v110 = 0;
  *(_QWORD *)((char *)&v96[-1] + qword_4A071C8[-3]) = &unk_4A071F0;
  sub_222DD70((char *)&v96[-1] + (_QWORD)*(v95 - 3), 0);
  v96[1] = 0;
  v96[2] = 0;
  v97 = 0;
  v95 = off_4A07238;
  v98 = 0;
  v99 = 0;
  v105[0] = off_4A07260;
  v100 = 0;
  v96[0] = off_4A07480;
  sub_220A990(v101);
  v102 = 16;
  LOBYTE(v104[0]) = 0;
  v96[0] = off_4A07080;
  v103[0] = v104;
  v103[1] = 0;
  sub_222DD70(v105, v96);
  sub_223E0D0(&v95, "OUTLINED_FUNCTION_", 18);
  sub_223E760(&v95, *(unsigned int *)(a3 + 40));
  v6 = *a2;
  v7 = sub_1643270((_QWORD *)*a2);
  LOBYTE(v113[0]) = 0;
  v112 = 0;
  v8 = (__int64 *)v7;
  v111 = (char *)v113;
  if ( v99 )
  {
    if ( v99 > v97 )
      sub_2241130(&v111, 0, 0, v98, v99 - v98);
    else
      sub_2241130(&v111, 0, 0, v98, v97 - v98);
  }
  else
  {
    sub_2240AE0(&v111, v103);
  }
  v90 = 0;
  v71 = v112;
  v72 = (__int64)v111;
  v89 = (unsigned __int8 *)v91;
  v9 = sub_1644EA0(v8, v91, 0, 0);
  v10 = sub_1632080((__int64)a2, v72, v71, v9, 0);
  if ( *(_BYTE *)(v10 + 16) )
    v10 = 0;
  v76 = v10;
  if ( v111 != (char *)v113 )
  {
    j_j___libc_free_0(v111, v113[0] + 1LL);
    v10 = v76;
  }
  *(_WORD *)(v10 + 32) = *(_WORD *)(v10 + 32) & 0xBF00 | 0x4087;
  sub_15E0D50(v10, -1, 34);
  sub_15E0D50(v76, -1, 17);
  v11 = (_BYTE *)a1[21];
  if ( v11 == (_BYTE *)a1[22] )
  {
    sub_14F2380((__int64)(a1 + 20), v11, &v76);
    v12 = v76;
  }
  else
  {
    v12 = v76;
    if ( v11 )
    {
      *(_QWORD *)v11 = v76;
      v11 = (_BYTE *)a1[21];
    }
    a1[21] = v11 + 8;
  }
  v111 = "entry";
  LOWORD(v113[0]) = 259;
  v13 = (_QWORD *)sub_22077B0(64);
  v14 = (__int64)v13;
  if ( v13 )
    sub_157FB60(v13, v6, (__int64)&v111, v12, 0);
  v15 = sub_157E9C0(v14);
  v90 = v14;
  v91[1] = v15;
  v16 = v15;
  v91[0] = v14 + 40;
  v89 = 0;
  v91[2] = 0;
  v92 = 0;
  v93 = 0;
  v94 = 0;
  LOWORD(v113[0]) = 257;
  v17 = sub_1648A60(56, 0);
  v18 = v17;
  if ( v17 )
    sub_15F6F90((__int64)v17, v16, 0, 0);
  if ( v90 )
  {
    v19 = (unsigned __int64 *)v91[0];
    sub_157E9D0(v90 + 40, (__int64)v18);
    v20 = v18[3];
    v21 = *v19;
    v18[4] = v19;
    v21 &= 0xFFFFFFFFFFFFFFF8LL;
    v18[3] = v21 | v20 & 7;
    *(_QWORD *)(v21 + 8) = v18 + 3;
    *v19 = *v19 & 7 | (unsigned __int64)(v18 + 3);
  }
  sub_164B780((__int64)v18, (__int64 *)&v111);
  if ( v89 )
  {
    v83 = v89;
    sub_1623A60((__int64)&v83, (__int64)v89, 2);
    v22 = v18[6];
    if ( v22 )
      sub_161E7C0((__int64)(v18 + 6), v22);
    v23 = v83;
    v18[6] = v83;
    if ( v23 )
      sub_1623210((__int64)&v83, v23, (__int64)(v18 + 6));
  }
  v24 = (__int64 *)a1[1];
  v25 = *v24;
  v26 = v24[1];
  if ( v25 == v26 )
LABEL_66:
    BUG();
  while ( *(_UNKNOWN **)v25 != &unk_4FC6A0E )
  {
    v25 += 16;
    if ( v26 == v25 )
      goto LABEL_66;
  }
  v27 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v25 + 8) + 104LL))(*(_QWORD *)(v25 + 8), &unk_4FC6A0E);
  v73 = sub_1E305C0(v27, v76);
  v63 = 0;
  v28 = (__int64)sub_1E0B6F0(v73, 0);
  v29 = *(__int64 (**)(void))(**(_QWORD **)(v73 + 16) + 40LL);
  if ( v29 != sub_1D00B00 )
    v63 = v29();
  v30 = *(__int64 **)(v73 + 328);
  sub_1DD8DC0(v73 + 320, v28);
  v31 = *(_QWORD *)v28;
  v32 = *v30;
  *(_QWORD *)(v28 + 8) = v30;
  v32 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v28 = v32 | v31 & 7;
  *(_QWORD *)(v32 + 8) = v28;
  *v30 = v28 | *v30 & 7;
  v33 = *(_DWORD **)(a3 + 48);
  for ( i = *(_DWORD **)(a3 + 56); i != v33; *(_QWORD *)(v28 + 24) = *(_QWORD *)(v28 + 24) & 7LL | v38 )
  {
    v42 = *(unsigned int *)(a4 + 64);
    v43 = *(_QWORD *)(a4 + 48);
    if ( (_DWORD)v42 )
    {
      v34 = (v42 - 1) & (37 * *v33);
      v35 = (int *)(v43 + 16LL * v34);
      v36 = *v35;
      if ( *v33 == *v35 )
        goto LABEL_32;
      v51 = 1;
      while ( v36 != -1 )
      {
        v52 = v51 + 1;
        v34 = (v42 - 1) & (v51 + v34);
        v35 = (int *)(v43 + 16LL * v34);
        v36 = *v35;
        if ( *v33 == *v35 )
          goto LABEL_32;
        v51 = v52;
      }
    }
    v35 = (int *)(v43 + 16 * v42);
LABEL_32:
    v37 = sub_1E0B7C0(v73, *((_QWORD *)v35 + 1));
    v37[7] = 0;
    v38 = (__int64)v37;
    *((_BYTE *)v37 + 49) = 0;
    v111 = 0;
    if ( v37 + 8 != &v111 )
    {
      v39 = v37[8];
      if ( v39 )
      {
        v68 = (__int64)(v37 + 8);
        sub_161E7C0((__int64)(v37 + 8), v39);
        v40 = (unsigned __int8 *)v111;
        *(_QWORD *)(v38 + 64) = v111;
        if ( v40 )
          sub_1623210((__int64)&v111, v40, v68);
      }
    }
    ++v33;
    sub_1DD5BA0((__int64 *)(v28 + 16), v38);
    v41 = *(_QWORD *)(v28 + 24);
    *(_QWORD *)(v38 + 8) = v28 + 24;
    v41 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)v38 = v41 | *(_QWORD *)v38 & 7LL;
    *(_QWORD *)(v41 + 8) = v38;
  }
  (*(void (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v63 + 1048LL))(v63, v28, v73, a3);
  v44 = *(_QWORD *)(a3 + 16);
  v45 = *(_QWORD *)(a3 + 8);
  if ( v45 != v44 )
  {
    while ( 1 )
    {
      if ( *(_QWORD *)v45 )
      {
        v46 = *(__int64 **)(*(_QWORD *)(*(_QWORD *)v45 + 24LL) + 56LL);
        if ( v46 )
        {
          v47 = sub_1626D20(*v46);
          if ( v47 )
            break;
        }
      }
      v45 += 16;
      if ( v44 == v45 )
        goto LABEL_45;
    }
    v53 = v47;
    v54 = (__int64)a2;
    sub_15A5590((__int64)&v111, a2, 1, *(_QWORD *)(v47 + 8 * (5LL - *(unsigned int *)(v47 + 8))));
    if ( *(_BYTE *)v53 != 15 )
      v53 = *(_QWORD *)(v53 - 8LL * *(unsigned int *)(v53 + 8));
    v77 = 0;
    v78 = 0;
    v55 = (__int64 *)a1[20];
    v56 = (__int64 *)a1[21];
    v79 = 0;
    v80 = 0;
    v67 = v56;
    if ( v56 != v55 )
    {
      v75 = v55;
      do
      {
        v57 = *v75;
        v81[1] = 0;
        LOBYTE(v82[0]) = 0;
        v81[0] = v82;
        v87 = 1;
        v83 = (unsigned __int8 *)&unk_49EFBE0;
        v86 = 0;
        v88 = v81;
        v85 = 0;
        v84 = 0;
        sub_38B9BB0(&v77, &v83, v57, 0);
        v58 = sub_15A66D0((__int64)&v111, 0, 0);
        v59 = sub_15A5D90((__int64)&v111, v58, 0, 0);
        if ( v86 != v84 )
          sub_16E7BA0((__int64 *)&v83);
        v69 = *v88;
        v70 = v88[1];
        v60 = sub_1649960(v57);
        v62 = sub_15A7010(
                (__int64)&v111,
                (_BYTE *)v53,
                (__int64)v60,
                v61,
                v69,
                v70,
                v53,
                0,
                v59,
                0,
                1,
                0,
                64,
                1u,
                0,
                0,
                0);
        sub_15A5DE0((__int64)&v111, v62);
        v54 = v62;
        sub_1627150(v57, v62);
        sub_16E7BC0((__int64 *)&v83);
        if ( (_QWORD *)v81[0] != v82 )
        {
          v54 = v82[0] + 1LL;
          j_j___libc_free_0(v81[0], v82[0] + 1LL);
        }
        ++v75;
      }
      while ( v67 != v75 );
    }
    sub_15A6130((__int64)&v111);
    j___libc_free_0(v78);
    sub_129E320((__int64)&v111, v54);
  }
LABEL_45:
  **(_QWORD **)(v73 + 352) &= ~4uLL;
  sub_1E69F60(*(_QWORD *)(v73 + 40));
  v49 = (__int64)v89;
  if ( v89 )
    sub_161E7C0((__int64)&v89, (__int64)v89);
  v95 = off_4A07238;
  v105[0] = off_4A07260;
  v96[0] = off_4A07080;
  if ( (_QWORD *)v103[0] != v104 )
  {
    v49 = v104[0] + 1LL;
    j_j___libc_free_0(v103[0], v104[0] + 1LL);
  }
  v96[0] = off_4A07480;
  sub_2209150(v101, v49, v48);
  v95 = (__int64 (__fastcall **)())qword_4A071C8;
  *(_QWORD *)((char *)&v96[-1] + qword_4A071C8[-3]) = &unk_4A071F0;
  v105[0] = off_4A06798;
  sub_222E050(v105);
  return v73;
}
