// Function: sub_3391DE0
// Address: 0x3391de0
//
__int64 __fastcall sub_3391DE0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r14
  __int128 v7; // rax
  __int64 v8; // r13
  __int128 v9; // rax
  int v10; // r9d
  __int64 v11; // rax
  __int64 v12; // r15
  __int64 v13; // r13
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rdi
  __int64 (__fastcall *v17)(__int64, __int64, unsigned int); // rax
  _DWORD *v18; // rax
  unsigned __int16 v19; // r8
  int v20; // eax
  __int64 v21; // r15
  unsigned int v22; // edx
  __int64 v23; // rax
  __int64 v24; // rdi
  __int64 (__fastcall *v25)(__int64, __int64, unsigned int); // rax
  __int64 v26; // rsi
  int v27; // eax
  unsigned int v28; // eax
  __int64 v29; // r15
  __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // r8
  __int64 v33; // r9
  __int64 v34; // rdx
  unsigned __int16 *v35; // rax
  __int128 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // r15
  __int64 v39; // rdx
  __int64 (__fastcall *v40)(__int64, __int64, __int64, __int64, __int64); // r12
  __int64 v41; // rax
  __int64 v42; // rax
  int v43; // eax
  int v44; // edx
  __int128 v45; // rax
  int v46; // r9d
  __int64 v47; // r12
  __int64 v48; // rdx
  __int64 v49; // r13
  __int64 v50; // r15
  __int128 v51; // rax
  int v52; // r9d
  __int64 v53; // r12
  __int64 v54; // rdx
  __int64 v55; // r15
  __int64 v56; // r15
  __int128 v57; // rax
  int v58; // r9d
  int v59; // edx
  __int64 v60; // r12
  unsigned int v61; // eax
  __int64 v63; // r12
  bool v64; // zf
  __int128 v65; // rax
  int v66; // r9d
  __int64 v67; // rax
  unsigned int v68; // edx
  __int64 v69; // rbx
  unsigned int v70; // r13d
  __int128 v71; // [rsp-40h] [rbp-180h]
  __int128 v72; // [rsp-20h] [rbp-160h]
  __int128 v73; // [rsp-10h] [rbp-150h]
  __int128 v74; // [rsp+0h] [rbp-140h]
  unsigned int v75; // [rsp+10h] [rbp-130h]
  __int128 v76; // [rsp+10h] [rbp-130h]
  int v77; // [rsp+20h] [rbp-120h]
  __int64 v78; // [rsp+28h] [rbp-118h]
  __int64 v79; // [rsp+28h] [rbp-118h]
  __int64 v81; // [rsp+38h] [rbp-108h]
  __int64 v82; // [rsp+38h] [rbp-108h]
  __int64 v83; // [rsp+40h] [rbp-100h]
  __int64 v84; // [rsp+48h] [rbp-F8h]
  unsigned int v85; // [rsp+48h] [rbp-F8h]
  __int64 (__fastcall *v86)(__int64, __int64, __int64, __int64); // [rsp+50h] [rbp-F0h]
  __int64 (__fastcall *v87)(__int64, __int64, __int64, __int64); // [rsp+50h] [rbp-F0h]
  unsigned int v88; // [rsp+50h] [rbp-F0h]
  __int64 v89; // [rsp+50h] [rbp-F0h]
  __int128 v91; // [rsp+60h] [rbp-E0h]
  __int128 v92; // [rsp+60h] [rbp-E0h]
  __int128 v93; // [rsp+60h] [rbp-E0h]
  unsigned __int64 v94; // [rsp+68h] [rbp-D8h]
  unsigned __int64 v95; // [rsp+F0h] [rbp-50h] BYREF
  unsigned int v96; // [rsp+F8h] [rbp-48h]
  unsigned __int64 v97; // [rsp+100h] [rbp-40h] BYREF
  unsigned int v98; // [rsp+108h] [rbp-38h]

  v4 = a2 + 24;
  *(_QWORD *)&v7 = sub_338B750(a1, *(_QWORD *)(a3 + 32));
  v8 = *(_QWORD *)(a1 + 864);
  v91 = v7;
  *(_QWORD *)&v7 = *(_QWORD *)(v7 + 48) + 16LL * DWORD2(v7);
  v81 = *(_QWORD *)(v7 + 8);
  v77 = *(unsigned __int16 *)v7;
  *(_QWORD *)&v9 = sub_34007B0(v8, a3, (int)a2 + 24, v77, v81, 0, 0);
  v11 = sub_3406EB0(v8, 57, (int)a2 + 24, v77, v81, v10, v91, v9);
  v12 = *(_QWORD *)(a1 + 864);
  v84 = v11;
  v13 = *(_QWORD *)(v12 + 16);
  v83 = v14;
  v86 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v13 + 1920LL);
  v15 = sub_2E79000(*(__int64 **)(v12 + 40));
  v16 = v15;
  if ( v86 == sub_3366DC0 )
  {
    v17 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v13 + 32LL);
    if ( v17 == sub_2D42F30 )
    {
      v18 = sub_AE2980(v16, 0);
      v19 = 2;
      v20 = v18[1];
      if ( v20 != 1 )
      {
        v19 = 3;
        if ( v20 != 2 )
        {
          v19 = 4;
          if ( v20 != 4 )
          {
            v19 = 5;
            if ( v20 != 8 )
            {
              v19 = 6;
              if ( v20 != 16 )
              {
                v19 = 7;
                if ( v20 != 32 )
                {
                  v19 = 8;
                  if ( v20 != 64 )
                    v19 = 9 * (v20 == 128);
                }
              }
            }
          }
        }
      }
    }
    else
    {
      v19 = v17(v13, v16, 0);
    }
  }
  else
  {
    v19 = ((__int64 (__fastcall *)(__int64, __int64))v86)(v13, v15);
  }
  v78 = sub_33FB310(v12, v84, v83, v4, v19, 0);
  v21 = *(_QWORD *)(a1 + 960);
  v75 = v22;
  v94 = v22 | *((_QWORD *)&v91 + 1) & 0xFFFFFFFF00000000LL;
  v87 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v13 + 1920LL);
  v23 = sub_2E79000(*(__int64 **)(*(_QWORD *)(a1 + 864) + 40LL));
  v24 = v23;
  if ( v87 == sub_3366DC0 )
  {
    v25 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v13 + 32LL);
    if ( v25 == sub_2D42F30 )
    {
      v26 = 2;
      v27 = sub_AE2980(v24, 0)[1];
      if ( v27 != 1 )
      {
        v26 = 3;
        if ( v27 != 2 )
        {
          v26 = 4;
          if ( v27 != 4 )
          {
            v26 = 5;
            if ( v27 != 8 )
            {
              v26 = 6;
              if ( v27 != 16 )
              {
                v26 = 7;
                if ( v27 != 32 )
                {
                  v26 = 8;
                  if ( v27 != 64 )
                    v26 = 9 * (unsigned int)(v27 == 128);
                }
              }
            }
          }
        }
      }
    }
    else
    {
      v26 = (unsigned int)v25(v13, v24, 0);
    }
  }
  else
  {
    v26 = ((unsigned int (__fastcall *)(__int64, __int64))v87)(v13, v23);
  }
  v28 = sub_374C900(v21, v26, 0);
  v29 = *(_QWORD *)(a1 + 864);
  v88 = v28;
  *(_QWORD *)&v74 = sub_3373A60(a1, v26, v30, v31, v32, v33);
  *((_QWORD *)&v74 + 1) = v34;
  v35 = (unsigned __int16 *)(*(_QWORD *)(v78 + 48) + 16LL * v75);
  *(_QWORD *)&v36 = sub_33F0B60(v29, v88, *v35, *((_QWORD *)v35 + 1));
  *((_QWORD *)&v73 + 1) = v94;
  *(_QWORD *)&v73 = v78;
  *(_QWORD *)&v92 = sub_340F900(v29, 49, v4, 1, 0, DWORD2(v74), v74, v36, v73);
  *((_QWORD *)&v92 + 1) = v37;
  *(_DWORD *)a2 = v88;
  if ( *(_BYTE *)(a3 + 49) )
  {
    v85 = v37;
    v63 = *(_QWORD *)(a2 + 8);
    v64 = v63 == sub_3374B60(a1, a4);
    v60 = *(_QWORD *)(a1 + 864);
    if ( v64 )
    {
      if ( (_QWORD)v92 )
      {
        nullsub_1875(v92, *(_QWORD *)(a1 + 864), 0);
        *(_QWORD *)(v60 + 384) = v92;
        v61 = v85;
        goto LABEL_34;
      }
      *(_QWORD *)(v60 + 384) = 0;
      *(_DWORD *)(v60 + 392) = v85;
      return v85;
    }
    else
    {
      *(_QWORD *)&v65 = sub_33EEAD0(*(_QWORD *)(a1 + 864), *(_QWORD *)(a2 + 8));
      v67 = sub_3406EB0(v60, 301, v4, 1, 0, v66, v92, v65);
      v69 = v67;
      v70 = v68;
      if ( v67 )
      {
        nullsub_1875(v67, v60, 0);
        *(_QWORD *)(v60 + 384) = v69;
        v61 = v70;
        goto LABEL_34;
      }
      *(_QWORD *)(v60 + 384) = 0;
      *(_DWORD *)(v60 + 392) = v68;
      return v68;
    }
  }
  else
  {
    v38 = *(_QWORD *)(a1 + 864);
    v96 = *(_DWORD *)(a3 + 24);
    if ( v96 > 0x40 )
      sub_C43780((__int64)&v95, (const void **)(a3 + 16));
    else
      v95 = *(_QWORD *)(a3 + 16);
    sub_C46B40((__int64)&v95, (__int64 *)a3);
    v98 = v96;
    v96 = 0;
    v97 = v95;
    *(_QWORD *)&v76 = sub_34007B0(v38, (unsigned int)&v97, v4, v77, v81, 0, 0);
    *((_QWORD *)&v76 + 1) = v39;
    v40 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)v13 + 528LL);
    v41 = *(_QWORD *)(a1 + 864);
    v79 = *(_QWORD *)(*(_QWORD *)(v84 + 48) + 16LL * (unsigned int)v83 + 8);
    v82 = *(unsigned __int16 *)(*(_QWORD *)(v84 + 48) + 16LL * (unsigned int)v83);
    v89 = *(_QWORD *)(v41 + 64);
    v42 = sub_2E79000(*(__int64 **)(v41 + 40));
    v43 = v40(v13, v42, v89, v82, v79);
    LODWORD(v82) = v44;
    LODWORD(v89) = v43;
    *(_QWORD *)&v45 = sub_33ED040(v38, 10);
    *((_QWORD *)&v71 + 1) = v83;
    *(_QWORD *)&v71 = v84;
    v47 = sub_340F900(v38, 208, v4, v89, v82, v46, v71, v76, v45);
    v49 = v48;
    if ( v98 > 0x40 && v97 )
      j_j___libc_free_0_0(v97);
    if ( v96 > 0x40 && v95 )
      j_j___libc_free_0_0(v95);
    v50 = *(_QWORD *)(a1 + 864);
    *(_QWORD *)&v51 = sub_33EEAD0(v50, *(_QWORD *)(a2 + 16));
    *((_QWORD *)&v72 + 1) = v49;
    *(_QWORD *)&v72 = v47;
    *(_QWORD *)&v93 = sub_340F900(v50, 305, v4, 1, 0, v52, v92, v72, v51);
    v53 = *(_QWORD *)(a2 + 8);
    *((_QWORD *)&v93 + 1) = v54;
    v55 = v93;
    if ( v53 != sub_3374B60(a1, a4) )
    {
      v56 = *(_QWORD *)(a1 + 864);
      *(_QWORD *)&v57 = sub_33EEAD0(v56, *(_QWORD *)(a2 + 8));
      v55 = sub_3406EB0(v56, 301, v4, 1, 0, v58, v93, v57);
      DWORD2(v93) = v59;
    }
    v60 = *(_QWORD *)(a1 + 864);
    if ( v55 )
    {
      nullsub_1875(v55, v60, 0);
      *(_QWORD *)(v60 + 384) = v55;
      v61 = DWORD2(v93);
LABEL_34:
      *(_DWORD *)(v60 + 392) = v61;
      return sub_33E2B60(v60, 0);
    }
    *(_QWORD *)(v60 + 384) = 0;
    *(_DWORD *)(v60 + 392) = DWORD2(v93);
    return DWORD2(v93);
  }
}
