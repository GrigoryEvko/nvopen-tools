// Function: sub_F0DE90
// Address: 0xf0de90
//
unsigned __int8 *__fastcall sub_F0DE90(const __m128i *a1, unsigned __int8 *a2)
{
  _BYTE *v4; // rbx
  unsigned __int8 *v5; // rdx
  _BYTE *v6; // r15
  int v7; // r13d
  unsigned __int8 *v8; // r9
  unsigned int v10; // r13d
  __int64 v11; // r10
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // rax
  __int64 v15; // rdi
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r9
  __int64 v19; // r11
  __int8 v20; // r8
  __int64 v21; // rax
  __int64 v22; // rax
  unsigned __int8 *v23; // rax
  int v24; // eax
  unsigned int v25; // r11d
  __int64 v26; // rax
  __int64 v27; // r11
  __int64 v28; // rsi
  __int64 v29; // rbx
  __int64 v30; // rcx
  __int64 v31; // rdx
  __int64 v32; // r9
  __int8 v33; // r8
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // rbx
  unsigned __int8 *v37; // rax
  __int64 *v38; // rdi
  int v39; // r8d
  __int64 v40; // rcx
  unsigned __int8 *v41; // rax
  unsigned __int8 *v42; // rax
  unsigned __int8 *v43; // rax
  __int64 *v44; // rdi
  __int64 *v45; // rdi
  __int64 v46; // r13
  unsigned __int8 *v47; // r9
  __int64 v48; // r13
  char v49; // al
  __int64 v50; // r9
  __int64 v51; // rdx
  int v52; // r14d
  __int64 v53; // rbx
  __int64 v54; // r13
  unsigned __int8 *v55; // r14
  __int64 v56; // rdx
  unsigned int v57; // esi
  char v58; // al
  __int64 v59; // r9
  __int64 v60; // rdx
  int v61; // r14d
  __int64 v62; // rbx
  __int64 v63; // r13
  __int64 v64; // rdx
  unsigned int v65; // esi
  __int64 v66; // [rsp+8h] [rbp-108h]
  __int64 v67; // [rsp+8h] [rbp-108h]
  __int64 v68; // [rsp+10h] [rbp-100h]
  unsigned int v69; // [rsp+10h] [rbp-100h]
  __int64 v70; // [rsp+18h] [rbp-F8h]
  __int64 v71; // [rsp+18h] [rbp-F8h]
  __int64 v72; // [rsp+18h] [rbp-F8h]
  __int64 v73; // [rsp+20h] [rbp-F0h]
  unsigned int v74; // [rsp+20h] [rbp-F0h]
  __int64 v75; // [rsp+20h] [rbp-F0h]
  unsigned __int8 *v76; // [rsp+28h] [rbp-E8h]
  __int64 v77; // [rsp+28h] [rbp-E8h]
  __int64 v78; // [rsp+28h] [rbp-E8h]
  unsigned __int8 *v79; // [rsp+28h] [rbp-E8h]
  unsigned __int8 *v80; // [rsp+28h] [rbp-E8h]
  unsigned __int8 *v81; // [rsp+28h] [rbp-E8h]
  __int64 v82; // [rsp+28h] [rbp-E8h]
  __int64 v83; // [rsp+28h] [rbp-E8h]
  __int64 v84; // [rsp+28h] [rbp-E8h]
  __int64 v85; // [rsp+28h] [rbp-E8h]
  __int64 v86; // [rsp+28h] [rbp-E8h]
  __int64 v87; // [rsp+28h] [rbp-E8h]
  int v88[8]; // [rsp+30h] [rbp-E0h] BYREF
  __int16 v89; // [rsp+50h] [rbp-C0h]
  _BYTE v90[32]; // [rsp+60h] [rbp-B0h] BYREF
  __int16 v91; // [rsp+80h] [rbp-90h]
  __int64 v92; // [rsp+90h] [rbp-80h] BYREF
  __int64 v93; // [rsp+98h] [rbp-78h]
  __int64 v94; // [rsp+A0h] [rbp-70h]
  __int64 v95; // [rsp+A8h] [rbp-68h]
  __int64 v96; // [rsp+B0h] [rbp-60h]
  unsigned __int8 *v97; // [rsp+B8h] [rbp-58h]
  __int64 v98; // [rsp+C0h] [rbp-50h]
  __int64 v99; // [rsp+C8h] [rbp-48h]
  __int8 v100; // [rsp+D0h] [rbp-40h]
  char v101; // [rsp+D1h] [rbp-3Fh]

  v4 = 0;
  v5 = 0;
  v6 = (_BYTE *)*((_QWORD *)a2 - 8);
  v7 = *a2;
  v73 = *((_QWORD *)a2 - 4);
  if ( (unsigned __int8)(*v6 - 42) <= 0x11u )
    v5 = (unsigned __int8 *)*((_QWORD *)a2 - 8);
  v76 = v5;
  if ( (unsigned __int8)(**((_BYTE **)a2 - 4) - 42) <= 0x11u )
    v4 = (_BYTE *)*((_QWORD *)a2 - 4);
  v8 = sub_F0D210((__int64)a1, a2);
  if ( v8 )
    return v8;
  v10 = v7 - 29;
  v11 = v73;
  if ( v76 )
  {
    v74 = *v76 - 29;
    if ( sub_F075A0(v74, v10) )
    {
      v13 = *(_QWORD *)(v12 - 32);
      v14 = *(_QWORD *)(v12 - 64);
      v97 = a2;
      v15 = a1[6].m128i_i64[1];
      v66 = v11;
      v16 = a1[8].m128i_i64[0];
      v70 = v13;
      v17 = a1[7].m128i_i64[1];
      v18 = a1[6].m128i_i64[0];
      v94 = a1[7].m128i_i64[0];
      v19 = a1[9].m128i_i64[1];
      v20 = a1[10].m128i_i8[0];
      v68 = v14;
      v21 = a1[9].m128i_i64[0];
      v93 = v15;
      v95 = v17;
      v96 = v16;
      v99 = v19;
      v92 = v18;
      v100 = v20;
      v98 = v21;
      v101 = 0;
      v77 = sub_101E7C0(v10, v68, v11, &v92);
      v22 = sub_101E7C0(v10, v70, v66, &v92);
      v11 = v66;
      if ( v77 )
      {
        if ( v22 )
        {
          v48 = a1[2].m128i_i64[0];
          v72 = v22;
          v89 = 257;
          v47 = (unsigned __int8 *)(*(__int64 (__fastcall **)(_QWORD, _QWORD, __int64, __int64))(**(_QWORD **)(v48 + 80)
                                                                                               + 16LL))(
                                     *(_QWORD *)(v48 + 80),
                                     v74,
                                     v77,
                                     v22);
          if ( v47 )
            goto LABEL_36;
          v91 = 257;
          v82 = sub_B504D0(v74, v77, v72, (__int64)v90, 0, 0);
          v49 = sub_920620(v82);
          v50 = v82;
          if ( v49 )
          {
            v51 = *(_QWORD *)(v48 + 96);
            v52 = *(_DWORD *)(v48 + 104);
            if ( v51 )
            {
              sub_B99FD0(v82, 3u, v51);
              v50 = v82;
            }
            v83 = v50;
            sub_B45150(v50, v52);
            v50 = v83;
          }
          v84 = v50;
          (*(void (__fastcall **)(_QWORD, __int64, int *, _QWORD, _QWORD))(**(_QWORD **)(v48 + 88) + 16LL))(
            *(_QWORD *)(v48 + 88),
            v50,
            v88,
            *(_QWORD *)(v48 + 56),
            *(_QWORD *)(v48 + 64));
          v53 = *(_QWORD *)v48;
          v47 = (unsigned __int8 *)v84;
          v54 = *(_QWORD *)v48 + 16LL * *(unsigned int *)(v48 + 8);
          if ( v53 == v54 )
            goto LABEL_36;
          v55 = (unsigned __int8 *)v84;
          do
          {
            v56 = *(_QWORD *)(v53 + 8);
            v57 = *(_DWORD *)v53;
            v53 += 16;
            sub_B99FD0(v84, v57, v56);
          }
          while ( v54 != v53 );
          goto LABEL_45;
        }
        v23 = sub_AD93D0(v74, *(_QWORD *)(v77 + 8), 0, 0);
        v11 = v66;
        if ( (unsigned __int8 *)v77 == v23 )
        {
          v45 = (__int64 *)a1[2].m128i_i64[0];
          v91 = 257;
          v41 = (unsigned __int8 *)sub_F0A990(v45, v10, v70, v66, v88[0], 0, (__int64)v90, 0);
          goto LABEL_23;
        }
      }
      else if ( v22 )
      {
        v80 = (unsigned __int8 *)v22;
        v43 = sub_AD93D0(v74, *(_QWORD *)(v22 + 8), 0, 0);
        v11 = v66;
        if ( v80 == v43 )
        {
          v44 = (__int64 *)a1[2].m128i_i64[0];
          v91 = 257;
          v41 = (unsigned __int8 *)sub_F0A990(v44, v10, v68, v66, v88[0], 0, (__int64)v90, 0);
          goto LABEL_23;
        }
      }
    }
  }
  if ( !v4 )
    return sub_F0D870(a1, a2, (__int64)v6, v11);
  v24 = (unsigned __int8)*v4;
  v25 = v24 - 29;
  if ( v10 == 28 )
  {
    if ( (unsigned int)(v24 - 58) > 1 )
      return sub_F0D870(a1, a2, (__int64)v6, v11);
  }
  else if ( v10 == 29 )
  {
    if ( v24 != 57 )
      return sub_F0D870(a1, a2, (__int64)v6, v11);
  }
  else if ( v10 != 17 || ((*v4 - 42) & 0xFD) != 0 )
  {
    return sub_F0D870(a1, a2, (__int64)v6, v11);
  }
  v26 = *((_QWORD *)v4 - 4);
  v69 = v25;
  v27 = *((_QWORD *)v4 - 8);
  v28 = a1[7].m128i_i64[0];
  v71 = v11;
  v29 = a1[9].m128i_i64[1];
  v30 = a1[7].m128i_i64[1];
  v93 = a1[6].m128i_i64[1];
  v31 = a1[8].m128i_i64[0];
  v32 = a1[6].m128i_i64[0];
  v94 = v28;
  v33 = a1[10].m128i_i8[0];
  v75 = v26;
  v99 = v29;
  v34 = a1[9].m128i_i64[0];
  v95 = v30;
  v96 = v31;
  v67 = v27;
  v92 = v32;
  v100 = v33;
  v97 = a2;
  v98 = v34;
  v101 = 0;
  v78 = sub_101E7C0(v10, v6, v27, &v92);
  v35 = sub_101E7C0(v10, v6, v75, &v92);
  v11 = v71;
  v36 = v35;
  if ( v78 )
  {
    if ( !v35 )
    {
      v37 = sub_AD93D0(v69, *(_QWORD *)(v78 + 8), 0, 0);
      v11 = v71;
      if ( (unsigned __int8 *)v78 == v37 )
      {
        v38 = (__int64 *)a1[2].m128i_i64[0];
        v39 = v88[0];
        v91 = 257;
        v40 = v75;
LABEL_22:
        v41 = (unsigned __int8 *)sub_F0A990(v38, v10, (__int64)v6, v40, v39, 0, (__int64)v90, 0);
LABEL_23:
        v79 = v41;
        sub_BD6B90(v41, a2);
        return v79;
      }
      return sub_F0D870(a1, a2, (__int64)v6, v11);
    }
    v46 = a1[2].m128i_i64[0];
    v89 = 257;
    v47 = (unsigned __int8 *)(*(__int64 (__fastcall **)(_QWORD, _QWORD, __int64, __int64))(**(_QWORD **)(v46 + 80) + 16LL))(
                               *(_QWORD *)(v46 + 80),
                               v69,
                               v78,
                               v35);
    if ( v47 )
      goto LABEL_36;
    v91 = 257;
    v85 = sub_B504D0(v69, v78, v36, (__int64)v90, 0, 0);
    v58 = sub_920620(v85);
    v59 = v85;
    if ( v58 )
    {
      v60 = *(_QWORD *)(v46 + 96);
      v61 = *(_DWORD *)(v46 + 104);
      if ( v60 )
      {
        sub_B99FD0(v85, 3u, v60);
        v59 = v85;
      }
      v86 = v59;
      sub_B45150(v59, v61);
      v59 = v86;
    }
    v87 = v59;
    (*(void (__fastcall **)(_QWORD, __int64, int *, _QWORD, _QWORD))(**(_QWORD **)(v46 + 88) + 16LL))(
      *(_QWORD *)(v46 + 88),
      v59,
      v88,
      *(_QWORD *)(v46 + 56),
      *(_QWORD *)(v46 + 64));
    v62 = *(_QWORD *)v46;
    v47 = (unsigned __int8 *)v87;
    v63 = *(_QWORD *)v46 + 16LL * *(unsigned int *)(v46 + 8);
    if ( v62 == v63 )
    {
LABEL_36:
      v81 = v47;
      sub_BD6B90(v47, a2);
      return v81;
    }
    v55 = (unsigned __int8 *)v87;
    do
    {
      v64 = *(_QWORD *)(v62 + 8);
      v65 = *(_DWORD *)v62;
      v62 += 16;
      sub_B99FD0(v87, v65, v64);
    }
    while ( v63 != v62 );
LABEL_45:
    v47 = v55;
    goto LABEL_36;
  }
  if ( v35 )
  {
    v42 = sub_AD93D0(v69, *(_QWORD *)(v35 + 8), 0, 0);
    v11 = v71;
    if ( (unsigned __int8 *)v36 == v42 )
    {
      v38 = (__int64 *)a1[2].m128i_i64[0];
      v91 = 257;
      v39 = v88[0];
      v40 = v67;
      goto LABEL_22;
    }
  }
  return sub_F0D870(a1, a2, (__int64)v6, v11);
}
