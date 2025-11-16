// Function: sub_23F6380
// Address: 0x23f6380
//
__int64 __fastcall sub_23F6380(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v11; // rdx
  _BYTE *v12; // rax
  __int64 v13; // rdx
  __int64 v14; // r12
  _BYTE *v15; // rdi
  __int64 *v16; // rax
  __int64 v17; // r15
  __int64 *v18; // rax
  __int64 v19; // r15
  __int64 *v20; // rax
  __int64 v21; // r14
  __int64 v22; // rdi
  __int64 v23; // r14
  __int64 *v24; // rax
  __int64 *v25; // rax
  __int64 v26; // r12
  __int64 v27; // rdi
  __int64 v28; // r13
  unsigned int v29; // r12d
  __int64 *v30; // r14
  __int64 v32; // rdi
  __int64 v33; // rax
  bool v34; // r14
  __int64 v35; // r14
  _QWORD *v36; // r15
  __int64 v37; // rdi
  __int64 v38; // r14
  __int64 v39; // rdi
  _QWORD **v40; // rdx
  int v41; // ecx
  __int64 *v42; // rax
  __int64 v43; // rsi
  __int64 v44; // r14
  __int64 v45; // r13
  __int64 v46; // rdx
  unsigned int v47; // esi
  __int64 v48; // rdi
  _QWORD **v49; // rdx
  int v50; // ecx
  __int64 *v51; // rax
  __int64 v52; // rsi
  __int64 v53; // rcx
  __int64 v54; // rbx
  __int64 v55; // r12
  __int64 v56; // rdx
  unsigned int v57; // esi
  __int64 v58; // rcx
  __int64 v59; // rbx
  __int64 v60; // r15
  __int64 v61; // rdx
  unsigned int v62; // esi
  __int64 v63; // r14
  __int64 v64; // r12
  __int64 v65; // rdx
  unsigned int v66; // esi
  __int64 v67; // r12
  __int64 v68; // rbx
  __int64 v69; // r12
  __int64 v70; // rdx
  unsigned int v71; // esi
  _QWORD **v72; // rdx
  int v73; // ecx
  int v74; // eax
  __int64 *v75; // rax
  __int64 v76; // rsi
  __int64 v77; // rax
  __int64 v78; // rbx
  __int64 v79; // r14
  __int64 v80; // rdx
  unsigned int v81; // esi
  __int64 v82; // [rsp+8h] [rbp-1A8h]
  __int64 v83; // [rsp+18h] [rbp-198h]
  _BYTE *v84; // [rsp+28h] [rbp-188h]
  __int64 v85; // [rsp+30h] [rbp-180h]
  __int64 v86; // [rsp+38h] [rbp-178h]
  __int64 v87; // [rsp+48h] [rbp-168h]
  __int64 v88; // [rsp+58h] [rbp-158h]
  __int64 v89; // [rsp+58h] [rbp-158h]
  __int64 v90; // [rsp+60h] [rbp-150h]
  __int64 v91; // [rsp+68h] [rbp-148h]
  char v92; // [rsp+78h] [rbp-138h]
  unsigned __int64 v93; // [rsp+80h] [rbp-130h] BYREF
  unsigned int v94; // [rsp+88h] [rbp-128h]
  unsigned __int64 v95; // [rsp+90h] [rbp-120h] BYREF
  unsigned int v96; // [rsp+98h] [rbp-118h]
  unsigned __int64 v97; // [rsp+A0h] [rbp-110h] BYREF
  unsigned int v98; // [rsp+A8h] [rbp-108h]
  unsigned __int64 v99; // [rsp+B0h] [rbp-100h] BYREF
  unsigned int v100; // [rsp+B8h] [rbp-F8h]
  unsigned __int64 v101; // [rsp+C0h] [rbp-F0h] BYREF
  unsigned int v102; // [rsp+C8h] [rbp-E8h]
  unsigned __int64 v103; // [rsp+D0h] [rbp-E0h] BYREF
  unsigned int v104; // [rsp+D8h] [rbp-D8h]
  unsigned __int64 v105; // [rsp+E0h] [rbp-D0h] BYREF
  unsigned int v106; // [rsp+E8h] [rbp-C8h]
  unsigned __int64 v107; // [rsp+F0h] [rbp-C0h] BYREF
  unsigned int v108; // [rsp+F8h] [rbp-B8h]
  unsigned __int64 v109; // [rsp+100h] [rbp-B0h] BYREF
  unsigned int v110; // [rsp+108h] [rbp-A8h]
  unsigned __int64 v111; // [rsp+110h] [rbp-A0h]
  unsigned int v112; // [rsp+118h] [rbp-98h]
  _BYTE v113[32]; // [rsp+120h] [rbp-90h] BYREF
  __int16 v114; // [rsp+140h] [rbp-70h]
  unsigned __int64 v115; // [rsp+150h] [rbp-60h] BYREF
  __int64 v116; // [rsp+158h] [rbp-58h]
  __int16 v117; // [rsp+170h] [rbp-40h]

  v115 = sub_9208B0(a3, a2);
  v116 = v11;
  v92 = v11;
  v12 = (_BYTE *)sub_D63C20(a4, a1);
  v88 = v13;
  if ( !v12 )
    return 0;
  v14 = (__int64)v12;
  if ( !v13 )
    return 0;
  v15 = 0;
  if ( *v12 == 17 )
    v15 = v12;
  v84 = v15;
  v83 = sub_AE4570(a3, *(_QWORD *)(a1 + 8));
  v85 = sub_B33F60(a5, v83, (v115 + 7) >> 3, v92);
  v16 = sub_DD8400(a6, v14);
  v17 = sub_DBB9F0(a6, (__int64)v16, 0, 0);
  v98 = *(_DWORD *)(v17 + 8);
  if ( v98 > 0x40 )
    sub_C43780((__int64)&v97, (const void **)v17);
  else
    v97 = *(_QWORD *)v17;
  v100 = *(_DWORD *)(v17 + 24);
  if ( v100 > 0x40 )
    sub_C43780((__int64)&v99, (const void **)(v17 + 16));
  else
    v99 = *(_QWORD *)(v17 + 16);
  v18 = sub_DD8400(a6, v88);
  v19 = sub_DBB9F0(a6, (__int64)v18, 0, 0);
  v102 = *(_DWORD *)(v19 + 8);
  if ( v102 > 0x40 )
    sub_C43780((__int64)&v101, (const void **)v19);
  else
    v101 = *(_QWORD *)v19;
  v104 = *(_DWORD *)(v19 + 24);
  if ( v104 > 0x40 )
    sub_C43780((__int64)&v103, (const void **)(v19 + 16));
  else
    v103 = *(_QWORD *)(v19 + 16);
  v20 = sub_DD8400(a6, v85);
  v21 = sub_DBB9F0(a6, (__int64)v20, 0, 0);
  v106 = *(_DWORD *)(v21 + 8);
  if ( v106 > 0x40 )
    sub_C43780((__int64)&v105, (const void **)v21);
  else
    v105 = *(_QWORD *)v21;
  v108 = *(_DWORD *)(v21 + 24);
  if ( v108 > 0x40 )
    sub_C43780((__int64)&v107, (const void **)(v21 + 16));
  else
    v107 = *(_QWORD *)(v21 + 16);
  v22 = *(_QWORD *)(a5 + 80);
  v114 = 257;
  v23 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD, _QWORD))(*(_QWORD *)v22 + 32LL))(
          v22,
          15,
          v14,
          v88,
          0,
          0);
  if ( !v23 )
  {
    v117 = 257;
    v23 = sub_B504D0(15, v14, v88, (__int64)&v115, 0, 0);
    (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a5 + 88) + 16LL))(
      *(_QWORD *)(a5 + 88),
      v23,
      v113,
      *(_QWORD *)(a5 + 56),
      *(_QWORD *)(a5 + 64));
    v58 = *(_QWORD *)a5 + 16LL * *(unsigned int *)(a5 + 8);
    if ( *(_QWORD *)a5 != v58 )
    {
      v87 = a5;
      v59 = *(_QWORD *)a5;
      v60 = v58;
      do
      {
        v61 = *(_QWORD *)(v59 + 8);
        v62 = *(_DWORD *)v59;
        v59 += 16;
        sub_B99FD0(v23, v62, v61);
      }
      while ( v60 != v59 );
      a5 = v87;
    }
  }
  sub_AB0A00((__int64)&v95, (__int64)&v97);
  sub_AB0910((__int64)&v109, (__int64)&v101);
  if ( (int)sub_C49970((__int64)&v95, &v109) < 0 )
  {
    v48 = *(_QWORD *)(a5 + 80);
    v114 = 257;
    v86 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v48 + 56LL))(v48, 36, v14, v88);
    if ( !v86 )
    {
      v117 = 257;
      v86 = (__int64)sub_BD2C40(72, unk_3F10FD0);
      if ( v86 )
      {
        v49 = *(_QWORD ***)(v14 + 8);
        v50 = *((unsigned __int8 *)v49 + 8);
        if ( (unsigned int)(v50 - 17) > 1 )
        {
          v52 = sub_BCB2A0(*v49);
        }
        else
        {
          BYTE4(v90) = (_BYTE)v50 == 18;
          LODWORD(v90) = *((_DWORD *)v49 + 8);
          v51 = (__int64 *)sub_BCB2A0(*v49);
          v52 = sub_BCE1B0(v51, v90);
        }
        sub_B523C0(v86, v52, 53, 36, v14, v88, (__int64)&v115, 0, 0, 0);
      }
      (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a5 + 88) + 16LL))(
        *(_QWORD *)(a5 + 88),
        v86,
        v113,
        *(_QWORD *)(a5 + 56),
        *(_QWORD *)(a5 + 64));
      v53 = *(_QWORD *)a5 + 16LL * *(unsigned int *)(a5 + 8);
      if ( *(_QWORD *)a5 != v53 )
      {
        v82 = a5;
        v54 = *(_QWORD *)a5;
        v55 = v53;
        do
        {
          v56 = *(_QWORD *)(v54 + 8);
          v57 = *(_DWORD *)v54;
          v54 += 16;
          sub_B99FD0(v86, v57, v56);
        }
        while ( v55 != v54 );
        a5 = v82;
      }
    }
  }
  else
  {
    v24 = (__int64 *)sub_BD5C60(a1);
    v86 = sub_ACD720(v24);
  }
  if ( v110 > 0x40 && v109 )
    j_j___libc_free_0_0(v109);
  if ( v96 > 0x40 && v95 )
    j_j___libc_free_0_0(v95);
  sub_AB51C0((__int64)&v109, (__int64)&v97, (__int64)&v101);
  sub_AB0A00((__int64)&v93, (__int64)&v109);
  sub_AB0910((__int64)&v95, (__int64)&v105);
  if ( (int)sub_C49970((__int64)&v93, &v95) < 0 )
  {
    v39 = *(_QWORD *)(a5 + 80);
    v114 = 257;
    v26 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v39 + 56LL))(v39, 36, v23, v85);
    if ( !v26 )
    {
      v117 = 257;
      v26 = (__int64)sub_BD2C40(72, unk_3F10FD0);
      if ( v26 )
      {
        v40 = *(_QWORD ***)(v23 + 8);
        v41 = *((unsigned __int8 *)v40 + 8);
        if ( (unsigned int)(v41 - 17) > 1 )
        {
          v43 = sub_BCB2A0(*v40);
        }
        else
        {
          BYTE4(v91) = (_BYTE)v41 == 18;
          LODWORD(v91) = *((_DWORD *)v40 + 8);
          v42 = (__int64 *)sub_BCB2A0(*v40);
          v43 = sub_BCE1B0(v42, v91);
        }
        sub_B523C0(v26, v43, 53, 36, v23, v85, (__int64)&v115, 0, 0, 0);
      }
      (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a5 + 88) + 16LL))(
        *(_QWORD *)(a5 + 88),
        v26,
        v113,
        *(_QWORD *)(a5 + 56),
        *(_QWORD *)(a5 + 64));
      v44 = *(_QWORD *)a5;
      v45 = *(_QWORD *)a5 + 16LL * *(unsigned int *)(a5 + 8);
      if ( *(_QWORD *)a5 != v45 )
      {
        do
        {
          v46 = *(_QWORD *)(v44 + 8);
          v47 = *(_DWORD *)v44;
          v44 += 16;
          sub_B99FD0(v26, v47, v46);
        }
        while ( v45 != v44 );
      }
    }
  }
  else
  {
    v25 = (__int64 *)sub_BD5C60(a1);
    v26 = sub_ACD720(v25);
  }
  if ( v96 > 0x40 && v95 )
    j_j___libc_free_0_0(v95);
  if ( v94 > 0x40 && v93 )
    j_j___libc_free_0_0(v93);
  if ( v112 > 0x40 && v111 )
    j_j___libc_free_0_0(v111);
  if ( v110 > 0x40 && v109 )
    j_j___libc_free_0_0(v109);
  v27 = *(_QWORD *)(a5 + 80);
  v114 = 257;
  v28 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v27 + 16LL))(v27, 29, v86, v26);
  if ( !v28 )
  {
    v117 = 257;
    v28 = sub_B504D0(29, v86, v26, (__int64)&v115, 0, 0);
    (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a5 + 88) + 16LL))(
      *(_QWORD *)(a5 + 88),
      v28,
      v113,
      *(_QWORD *)(a5 + 56),
      *(_QWORD *)(a5 + 64));
    v63 = *(_QWORD *)a5;
    v64 = *(_QWORD *)a5 + 16LL * *(unsigned int *)(a5 + 8);
    if ( *(_QWORD *)a5 != v64 )
    {
      do
      {
        v65 = *(_QWORD *)(v63 + 8);
        v66 = *(_DWORD *)v63;
        v63 += 16;
        sub_B99FD0(v28, v66, v65);
      }
      while ( v64 != v63 );
    }
  }
  if ( !v84 )
    goto LABEL_68;
  v29 = *((_DWORD *)v84 + 8);
  v30 = (__int64 *)*((_QWORD *)v84 + 3);
  if ( v29 <= 0x40 )
  {
    if ( !v29 || (__int64)((_QWORD)v30 << (64 - (unsigned __int8)v29)) >= 0 )
      goto LABEL_45;
LABEL_68:
    sub_AB14C0((__int64)&v115, (__int64)&v97);
    v33 = 1LL << ((unsigned __int8)v116 - 1);
    if ( (unsigned int)v116 <= 0x40 )
    {
      v34 = (v115 & v33) != 0;
    }
    else
    {
      v34 = (*(_QWORD *)(v115 + 8LL * ((unsigned int)(v116 - 1) >> 6)) & v33) != 0;
      if ( v115 )
        j_j___libc_free_0_0(v115);
    }
    if ( v34 )
    {
      v114 = 257;
      v35 = sub_AD64C0(v83, 0, 0);
      v36 = (_QWORD *)(*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(a5 + 80) + 56LL))(
                        *(_QWORD *)(a5 + 80),
                        40,
                        v88,
                        v35);
      if ( !v36 )
      {
        v117 = 257;
        v36 = sub_BD2C40(72, unk_3F10FD0);
        if ( v36 )
        {
          v72 = *(_QWORD ***)(v88 + 8);
          v73 = *((unsigned __int8 *)v72 + 8);
          if ( (unsigned int)(v73 - 17) > 1 )
          {
            v76 = sub_BCB2A0(*v72);
          }
          else
          {
            v74 = *((_DWORD *)v72 + 8);
            BYTE4(v109) = (_BYTE)v73 == 18;
            LODWORD(v109) = v74;
            v75 = (__int64 *)sub_BCB2A0(*v72);
            v76 = sub_BCE1B0(v75, v109);
          }
          sub_B523C0((__int64)v36, v76, 53, 40, v88, v35, (__int64)&v115, 0, 0, 0);
        }
        (*(void (__fastcall **)(_QWORD, _QWORD *, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a5 + 88) + 16LL))(
          *(_QWORD *)(a5 + 88),
          v36,
          v113,
          *(_QWORD *)(a5 + 56),
          *(_QWORD *)(a5 + 64));
        v77 = *(_QWORD *)a5 + 16LL * *(unsigned int *)(a5 + 8);
        if ( *(_QWORD *)a5 != v77 )
        {
          v89 = a5;
          v78 = *(_QWORD *)a5;
          v79 = v77;
          do
          {
            v80 = *(_QWORD *)(v78 + 8);
            v81 = *(_DWORD *)v78;
            v78 += 16;
            sub_B99FD0((__int64)v36, v81, v80);
          }
          while ( v79 != v78 );
          a5 = v89;
        }
      }
      v37 = *(_QWORD *)(a5 + 80);
      v114 = 257;
      v38 = (*(__int64 (__fastcall **)(__int64, __int64, _QWORD *, __int64))(*(_QWORD *)v37 + 16LL))(v37, 29, v36, v28);
      if ( !v38 )
      {
        v117 = 257;
        v38 = sub_B504D0(29, (__int64)v36, v28, (__int64)&v115, 0, 0);
        (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a5 + 88) + 16LL))(
          *(_QWORD *)(a5 + 88),
          v38,
          v113,
          *(_QWORD *)(a5 + 56),
          *(_QWORD *)(a5 + 64));
        v67 = 16LL * *(unsigned int *)(a5 + 8);
        v68 = *(_QWORD *)a5;
        v69 = v68 + v67;
        while ( v69 != v68 )
        {
          v70 = *(_QWORD *)(v68 + 8);
          v71 = *(_DWORD *)v68;
          v68 += 16;
          sub_B99FD0(v38, v71, v70);
        }
      }
      v28 = v38;
    }
    goto LABEL_45;
  }
  v32 = (__int64)(v84 + 24);
  if ( (v30[(v29 - 1) >> 6] & (1LL << ((unsigned __int8)v29 - 1))) != 0 )
  {
    if ( v29 + 1 - (unsigned int)sub_C44500(v32) > 0x40 )
      goto LABEL_68;
  }
  else if ( v29 + 1 - (unsigned int)sub_C444A0(v32) > 0x40 )
  {
    goto LABEL_45;
  }
  if ( *v30 < 0 )
    goto LABEL_68;
LABEL_45:
  if ( v108 > 0x40 && v107 )
    j_j___libc_free_0_0(v107);
  if ( v106 > 0x40 && v105 )
    j_j___libc_free_0_0(v105);
  if ( v104 > 0x40 && v103 )
    j_j___libc_free_0_0(v103);
  if ( v102 > 0x40 && v101 )
    j_j___libc_free_0_0(v101);
  if ( v100 > 0x40 && v99 )
    j_j___libc_free_0_0(v99);
  if ( v98 > 0x40 && v97 )
    j_j___libc_free_0_0(v97);
  return v28;
}
