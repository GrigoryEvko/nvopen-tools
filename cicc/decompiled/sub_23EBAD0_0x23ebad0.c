// Function: sub_23EBAD0
// Address: 0x23ebad0
//
void __fastcall sub_23EBAD0(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rax
  int v3; // edx
  int v4; // ebx
  unsigned __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rcx
  __int64 v8; // rdx
  __int64 v9; // r14
  _QWORD *v10; // rdi
  _QWORD *v11; // rax
  __int64 v12; // rcx
  __int64 v13; // rax
  __int64 v14; // rdx
  char *v15; // rbx
  __int64 v16; // r14
  __int64 v17; // rax
  __int64 *v18; // r9
  _BYTE *v19; // r14
  __int64 (__fastcall *v20)(__int64, unsigned int); // rax
  __int64 v21; // rbx
  __int64 *v22; // rax
  unsigned __int64 v23; // rax
  __int64 v24; // r13
  __int64 v25; // rdx
  __int64 v26; // rbx
  _QWORD *v27; // rdi
  __int64 v28; // rbx
  __int64 *v29; // rax
  unsigned __int64 v30; // rax
  __int64 v31; // r13
  __int64 v32; // rdx
  __int64 v33; // rbx
  const char *v34; // rax
  __int64 *v35; // rbx
  _BYTE *v36; // rsi
  unsigned __int64 v37; // rax
  __int64 v38; // rax
  __int64 v39; // rdx
  __int64 (__fastcall **v40)(); // rdi
  unsigned __int64 v41; // rax
  __int64 v42; // rax
  __int64 v43; // rdx
  _QWORD *v44; // rdi
  __int64 v45; // rbx
  __int64 *v46; // rax
  __int64 v47; // rax
  _QWORD *v48; // rdi
  __int64 v49; // rbx
  __int64 v50; // rdx
  __int64 *v51; // rax
  __int64 v52; // rdx
  int *i; // [rsp+48h] [rbp-2C8h]
  unsigned __int64 v55; // [rsp+50h] [rbp-2C0h]
  const char *v56; // [rsp+50h] [rbp-2C0h]
  __int64 v57; // [rsp+58h] [rbp-2B8h]
  __int64 v58; // [rsp+58h] [rbp-2B8h]
  __int64 *v59; // [rsp+58h] [rbp-2B8h]
  __int64 *v60; // [rsp+58h] [rbp-2B8h]
  __int64 v61; // [rsp+58h] [rbp-2B8h]
  __int64 *v62; // [rsp+60h] [rbp-2B0h]
  __int64 *v63; // [rsp+60h] [rbp-2B0h]
  unsigned __int64 v64; // [rsp+68h] [rbp-2A8h]
  __int64 v65; // [rsp+68h] [rbp-2A8h]
  unsigned __int64 v66; // [rsp+68h] [rbp-2A8h]
  __int64 v67; // [rsp+68h] [rbp-2A8h]
  __int64 v68; // [rsp+70h] [rbp-2A0h]
  __int64 v69; // [rsp+70h] [rbp-2A0h]
  __int64 v70; // [rsp+70h] [rbp-2A0h]
  __int64 v71; // [rsp+70h] [rbp-2A0h]
  __int64 v72; // [rsp+80h] [rbp-290h]
  unsigned __int64 v73; // [rsp+80h] [rbp-290h]
  _QWORD *v75; // [rsp+90h] [rbp-280h] BYREF
  unsigned __int64 v76; // [rsp+98h] [rbp-278h]
  _BYTE v77[16]; // [rsp+A0h] [rbp-270h] BYREF
  _QWORD *v78; // [rsp+B0h] [rbp-260h] BYREF
  __int64 v79; // [rsp+B8h] [rbp-258h]
  _QWORD v80[2]; // [rsp+C0h] [rbp-250h] BYREF
  _BYTE *v81; // [rsp+D0h] [rbp-240h]
  __int64 v82; // [rsp+D8h] [rbp-238h]
  _BYTE v83[32]; // [rsp+E0h] [rbp-230h] BYREF
  __int64 v84; // [rsp+100h] [rbp-210h]
  __int64 v85; // [rsp+108h] [rbp-208h]
  __int16 v86; // [rsp+110h] [rbp-200h]
  _QWORD *v87; // [rsp+118h] [rbp-1F8h]
  void **v88; // [rsp+120h] [rbp-1F0h]
  void **v89; // [rsp+128h] [rbp-1E8h]
  __int64 v90; // [rsp+130h] [rbp-1E0h]
  int v91; // [rsp+138h] [rbp-1D8h]
  __int16 v92; // [rsp+13Ch] [rbp-1D4h]
  char v93; // [rsp+13Eh] [rbp-1D2h]
  __int64 v94; // [rsp+140h] [rbp-1D0h]
  __int64 v95; // [rsp+148h] [rbp-1C8h]
  void *v96; // [rsp+150h] [rbp-1C0h] BYREF
  void *v97; // [rsp+158h] [rbp-1B8h] BYREF
  __int64 (__fastcall **v98)(); // [rsp+160h] [rbp-1B0h] BYREF
  __int64 v99; // [rsp+168h] [rbp-1A8h] BYREF
  __int64 v100; // [rsp+170h] [rbp-1A0h] BYREF
  __int64 v101; // [rsp+178h] [rbp-198h]
  unsigned __int64 v102; // [rsp+180h] [rbp-190h]
  _BYTE *v103; // [rsp+188h] [rbp-188h]
  unsigned __int64 v104; // [rsp+190h] [rbp-180h]
  __int64 v105; // [rsp+198h] [rbp-178h]
  volatile signed __int32 *v106; // [rsp+1A0h] [rbp-170h] BYREF
  int v107; // [rsp+1A8h] [rbp-168h]
  unsigned __int64 v108[2]; // [rsp+1B0h] [rbp-160h] BYREF
  _BYTE v109[16]; // [rsp+1C0h] [rbp-150h] BYREF
  _QWORD v110[28]; // [rsp+1D0h] [rbp-140h] BYREF
  __int16 v111; // [rsp+2B0h] [rbp-60h]
  __int64 v112; // [rsp+2B8h] [rbp-58h]
  __int64 v113; // [rsp+2C0h] [rbp-50h]
  __int64 v114; // [rsp+2C8h] [rbp-48h]
  __int64 v115; // [rsp+2D0h] [rbp-40h]

  v87 = (_QWORD *)a1[57];
  v88 = &v96;
  v89 = &v97;
  v81 = v83;
  v82 = 0x200000000LL;
  v96 = &unk_49DA100;
  v92 = 512;
  v97 = &unk_49DA0B0;
  v2 = a1[1];
  v90 = 0;
  v91 = 0;
  v93 = 7;
  v94 = 0;
  v95 = 0;
  v84 = 0;
  v85 = 0;
  v86 = 0;
  v3 = *(_DWORD *)(v2 + 88);
  if ( (unsigned int)(v3 - 1) > 1 )
  {
    if ( !*(_BYTE *)(v2 + 86) )
      goto LABEL_3;
LABEL_51:
    v45 = a1[58];
    v46 = (__int64 *)sub_BCB120(v87);
    v47 = sub_23EA340(a2, (__int64)"__asan_poison_stack_memory", 0x1Au, 0, v46, v45, v45);
    v48 = v87;
    v49 = a1[58];
    a1[666] = v50;
    a1[665] = v47;
    v51 = (__int64 *)sub_BCB120(v48);
    a1[667] = sub_23EA340(a2, (__int64)"__asan_unpoison_stack_memory", 0x1Cu, 0, v51, v49, v49);
    a1[668] = v52;
    goto LABEL_3;
  }
  v73 = 0;
  v34 = "__asan_stack_malloc_";
  if ( v3 == 2 )
    v34 = "__asan_stack_malloc_always_";
  v35 = a1 + 109;
  v56 = v34;
  do
  {
    if ( v73 )
    {
      BYTE4(v100) = v73 % 0xA + 48;
      if ( v73 == 10 )
      {
        BYTE3(v100) = 49;
        v36 = (char *)&v100 + 3;
      }
      else
      {
        v36 = (char *)&v100 + 4;
      }
    }
    else
    {
      BYTE4(v100) = 48;
      v36 = (char *)&v100 + 4;
    }
    v75 = v77;
    sub_23DC7D0((__int64 *)&v75, v36, (__int64)&v100 + 5);
    v62 = (__int64 *)a1[58];
    sub_8FD6D0((__int64)&v98, v56, &v75);
    v78 = v80;
    v79 = 0x100000001LL;
    v64 = v99;
    v68 = (__int64)v98;
    v80[0] = v62;
    v37 = sub_BCF480(v62, v80, 1, 0);
    v38 = sub_BA8C10(a2, v68, v64, v37, 0);
    if ( v78 != v80 )
    {
      v65 = v39;
      v69 = v38;
      _libc_free((unsigned __int64)v78);
      v39 = v65;
      v38 = v69;
    }
    v40 = v98;
    *v35 = v38;
    v35[1] = v39;
    if ( v40 != (__int64 (__fastcall **)())&v100 )
      j_j___libc_free_0((unsigned __int64)v40);
    v61 = a1[58];
    v63 = (__int64 *)sub_BCB120(v87);
    sub_8FD6D0((__int64)&v78, "__asan_stack_free_", &v75);
    v100 = v61;
    v101 = v61;
    v66 = v79;
    v70 = (__int64)v78;
    v98 = (__int64 (__fastcall **)())&v100;
    v99 = 0x200000002LL;
    v41 = sub_BCF480(v63, &v100, 2, 0);
    v42 = sub_BA8C10(a2, v70, v66, v41, 0);
    if ( v98 != (__int64 (__fastcall **)())&v100 )
    {
      v67 = v43;
      v71 = v42;
      _libc_free((unsigned __int64)v98);
      v43 = v67;
      v42 = v71;
    }
    v44 = v78;
    v35[22] = v42;
    v35[23] = v43;
    if ( v44 != v80 )
      j_j___libc_free_0((unsigned __int64)v44);
    if ( v75 != (_QWORD *)v77 )
      j_j___libc_free_0((unsigned __int64)v75);
    ++v73;
    v35 += 2;
  }
  while ( v73 != 11 );
  if ( *(_BYTE *)(a1[1] + 86LL) )
    goto LABEL_51;
LABEL_3:
  v4 = 0;
  for ( i = (int *)&unk_437FDA4; ; ++i )
  {
    v72 = v4;
    sub_222DF20((__int64)v110);
    v98 = (__int64 (__fastcall **)())qword_4A071C8;
    v110[27] = 0;
    v112 = 0;
    v110[0] = off_4A06798;
    v111 = 0;
    v113 = 0;
    v114 = 0;
    v115 = 0;
    *(__int64 (__fastcall ***)())((char *)&v98 + qword_4A071C8[-3]) = (__int64 (__fastcall **)())&unk_4A071F0;
    sub_222DD70((__int64)&v98 + (_QWORD)*(v98 - 3), 0);
    v100 = 0;
    v101 = 0;
    v102 = 0;
    v98 = off_4A07238;
    v103 = 0;
    v104 = 0;
    v110[0] = off_4A07260;
    v105 = 0;
    v99 = (__int64)off_4A07480;
    sub_220A990(&v106);
    v107 = 16;
    v109[0] = 0;
    v99 = (__int64)off_4A07080;
    v108[1] = 0;
    v108[0] = (unsigned __int64)v109;
    sub_222DD70((__int64)v110, (__int64)&v99);
    sub_223E0D0((__int64 *)&v98, "__asan_set_shadow_", 18);
    v13 = (__int64)v98;
    v14 = (__int64)*(v98 - 3);
    *(__int64 *)((char *)&v100 + v14) = 2;
    v15 = (char *)&v98 + *(_QWORD *)(v13 - 24);
    if ( !v15[225] )
    {
      v19 = (_BYTE *)*((_QWORD *)v15 + 30);
      if ( !v19 )
        sub_426219(&v98, "__asan_set_shadow_", v14, v12);
      if ( !v19[56] )
      {
        sub_2216D60(*((_QWORD *)v15 + 30));
        v20 = *(__int64 (__fastcall **)(__int64, unsigned int))(*(_QWORD *)v19 + 48LL);
        if ( v20 != sub_CE72A0 )
          v20((__int64)v19, 32u);
        v13 = (__int64)v98;
      }
      v15[225] = 1;
    }
    v15[224] = 48;
    *(_DWORD *)((char *)&v101 + *(_QWORD *)(v13 - 24)) = *(_DWORD *)((_BYTE *)&v101 + *(_QWORD *)(v13 - 24))
                                                       & 0xFFFFFFB5
                                                       | 8;
    sub_223E760((__int64 *)&v98, v72);
    v16 = a1[58];
    v17 = sub_BCB120(v87);
    v77[0] = 0;
    v76 = 0;
    v75 = v77;
    if ( v104 )
    {
      v59 = (__int64 *)v17;
      if ( v104 > v102 )
        sub_2241130((unsigned __int64 *)&v75, 0, 0, v103, v104 - (_QWORD)v103);
      else
        sub_2241130((unsigned __int64 *)&v75, 0, 0, v103, v102 - (_QWORD)v103);
      v18 = v59;
    }
    else
    {
      v60 = (__int64 *)v17;
      sub_2240AE0((unsigned __int64 *)&v75, v108);
      v18 = v60;
    }
    v80[0] = v16;
    v55 = v76;
    v57 = (__int64)v75;
    v80[1] = v16;
    v78 = v80;
    v79 = 0x200000002LL;
    v5 = sub_BCF480(v18, v80, 2, 0);
    v6 = sub_BA8C10(a2, v57, v55, v5, 0);
    v7 = v6;
    v9 = v8;
    if ( v78 != v80 )
    {
      v58 = v6;
      _libc_free((unsigned __int64)v78);
      v7 = v58;
    }
    v10 = v75;
    v11 = &a1[2 * v72];
    v11[153] = v7;
    v11[154] = v9;
    if ( v10 != (_QWORD *)v77 )
      j_j___libc_free_0((unsigned __int64)v10);
    v98 = off_4A07238;
    v110[0] = off_4A07260;
    v99 = (__int64)off_4A07080;
    if ( (_BYTE *)v108[0] != v109 )
      j_j___libc_free_0(v108[0]);
    v99 = (__int64)off_4A07480;
    sub_2209150(&v106);
    v98 = (__int64 (__fastcall **)())qword_4A071C8;
    *(__int64 (__fastcall ***)())((char *)&v98 + qword_4A071C8[-3]) = (__int64 (__fastcall **)())&unk_4A071F0;
    v110[0] = off_4A06798;
    sub_222E050((__int64)v110);
    if ( i == (int *)&unk_437FDD4 )
      break;
    v4 = *i;
  }
  v21 = a1[58];
  v22 = (__int64 *)sub_BCB120(v87);
  v100 = v21;
  v101 = v21;
  v98 = (__int64 (__fastcall **)())&v100;
  v99 = 0x200000002LL;
  v23 = sub_BCF480(v22, &v100, 2, 0);
  v24 = sub_BA8C10(a2, (__int64)"__asan_alloca_poison", 0x14u, v23, 0);
  v26 = v25;
  if ( v98 != (__int64 (__fastcall **)())&v100 )
    _libc_free((unsigned __int64)v98);
  v27 = v87;
  a1[669] = v24;
  a1[670] = v26;
  v28 = a1[58];
  v29 = (__int64 *)sub_BCB120(v27);
  v100 = v28;
  v101 = v28;
  v98 = (__int64 (__fastcall **)())&v100;
  v99 = 0x200000002LL;
  v30 = sub_BCF480(v29, &v100, 2, 0);
  v31 = sub_BA8C10(a2, (__int64)"__asan_allocas_unpoison", 0x17u, v30, 0);
  v33 = v32;
  if ( v98 != (__int64 (__fastcall **)())&v100 )
    _libc_free((unsigned __int64)v98);
  a1[671] = v31;
  a1[672] = v33;
  nullsub_61();
  v96 = &unk_49DA100;
  nullsub_63();
  if ( v81 != v83 )
    _libc_free((unsigned __int64)v81);
}
