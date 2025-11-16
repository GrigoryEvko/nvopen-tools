// Function: sub_180C840
// Address: 0x180c840
//
unsigned __int64 __fastcall sub_180C840(_QWORD *a1, __int64 a2)
{
  unsigned __int64 v2; // r12
  _BYTE *v3; // rsi
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // r8
  __int64 v7; // rax
  __int64 (__fastcall **v8)(); // rdi
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // r8
  __int64 v12; // rax
  _QWORD *v13; // rdi
  __int64 i; // rax
  __int64 v15; // r8
  __int64 *v16; // r11
  __int64 v17; // rax
  __int64 v18; // rsi
  __int64 v19; // r15
  __int64 v20; // rax
  __int64 v21; // rdx
  _QWORD *v22; // rdi
  __int64 v23; // rcx
  __int64 v24; // r8
  unsigned __int64 v25; // rax
  __int64 v26; // rdx
  char *v27; // r15
  __int64 v28; // r15
  __int64 v29; // rax
  __int64 v30; // rdi
  __int64 (__fastcall *v31)(__int64, unsigned int); // rax
  __int64 v32; // rbx
  __int64 *v33; // rax
  __int64 v34; // rax
  __int64 v35; // rax
  _QWORD *v36; // rdi
  __int64 v37; // rbx
  __int64 *v38; // rax
  __int64 v39; // rax
  __int64 v40; // rax
  __int64 v41; // rsi
  unsigned __int64 result; // rax
  __int64 v43; // r12
  __int64 *v44; // rax
  __int64 v45; // rax
  __int64 v46; // rax
  _QWORD *v47; // rdi
  __int64 v48; // r12
  __int64 *v49; // rax
  __int64 v50; // rax
  unsigned __int64 v51; // [rsp+0h] [rbp-2B0h]
  __int64 v52; // [rsp+20h] [rbp-290h]
  __int64 v53; // [rsp+28h] [rbp-288h]
  __int64 *v54; // [rsp+28h] [rbp-288h]
  __int64 v55; // [rsp+28h] [rbp-288h]
  __int64 *v56; // [rsp+28h] [rbp-288h]
  __int64 v57; // [rsp+38h] [rbp-278h]
  __int64 *v58; // [rsp+40h] [rbp-270h]
  __int64 *v59; // [rsp+40h] [rbp-270h]
  __int64 v60; // [rsp+48h] [rbp-268h]
  __int64 v61; // [rsp+48h] [rbp-268h]
  __int64 v62; // [rsp+48h] [rbp-268h]
  __int64 v63; // [rsp+50h] [rbp-260h]
  __int64 v64; // [rsp+50h] [rbp-260h]
  __int64 v65; // [rsp+50h] [rbp-260h]
  __int64 v66; // [rsp+50h] [rbp-260h]
  char *v67; // [rsp+50h] [rbp-260h]
  _QWORD *v70; // [rsp+70h] [rbp-240h] BYREF
  __int64 v71; // [rsp+78h] [rbp-238h]
  _QWORD v72[2]; // [rsp+80h] [rbp-230h] BYREF
  _QWORD *v73; // [rsp+90h] [rbp-220h] BYREF
  __int64 v74; // [rsp+98h] [rbp-218h]
  _QWORD v75[2]; // [rsp+A0h] [rbp-210h] BYREF
  _QWORD v76[3]; // [rsp+B0h] [rbp-200h] BYREF
  _QWORD *v77; // [rsp+C8h] [rbp-1E8h]
  __int64 v78; // [rsp+D0h] [rbp-1E0h]
  int v79; // [rsp+D8h] [rbp-1D8h]
  __int64 v80; // [rsp+E0h] [rbp-1D0h]
  __int64 v81; // [rsp+E8h] [rbp-1C8h]
  __int64 (__fastcall **v82)(); // [rsp+100h] [rbp-1B0h] BYREF
  __int64 v83; // [rsp+108h] [rbp-1A8h] BYREF
  __int64 v84; // [rsp+110h] [rbp-1A0h] BYREF
  __int64 v85; // [rsp+118h] [rbp-198h]
  unsigned __int64 v86; // [rsp+120h] [rbp-190h]
  __int64 v87; // [rsp+128h] [rbp-188h]
  unsigned __int64 v88; // [rsp+130h] [rbp-180h]
  __int64 v89; // [rsp+138h] [rbp-178h]
  char v90[8]; // [rsp+140h] [rbp-170h] BYREF
  int v91; // [rsp+148h] [rbp-168h]
  _QWORD v92[2]; // [rsp+150h] [rbp-160h] BYREF
  _QWORD v93[2]; // [rsp+160h] [rbp-150h] BYREF
  _QWORD v94[28]; // [rsp+170h] [rbp-140h] BYREF
  __int16 v95; // [rsp+250h] [rbp-60h]
  __int64 v96; // [rsp+258h] [rbp-58h]
  __int64 v97; // [rsp+260h] [rbp-50h]
  __int64 v98; // [rsp+268h] [rbp-48h]
  __int64 v99; // [rsp+270h] [rbp-40h]

  v2 = 0;
  v77 = (_QWORD *)a1[60];
  memset(v76, 0, sizeof(v76));
  v78 = 0;
  v79 = 0;
  v80 = 0;
  v81 = 0;
  do
  {
    if ( v2 )
    {
      BYTE4(v84) = v2 % 0xA + 48;
      if ( v2 == 10 )
      {
        BYTE3(v84) = 49;
        v3 = (char *)&v84 + 3;
      }
      else
      {
        v3 = (char *)&v84 + 4;
      }
    }
    else
    {
      BYTE4(v84) = 48;
      v3 = (char *)&v84 + 4;
    }
    v70 = v72;
    sub_1801C90((__int64 *)&v70, v3, (__int64)&v84 + 5);
    v58 = (__int64 *)a1[61];
    sub_8FD6D0((__int64)&v82, "__asan_stack_malloc_", &v70);
    v73 = v75;
    v74 = 0x100000001LL;
    v60 = v83;
    v63 = (__int64)v82;
    v75[0] = v58;
    v4 = sub_1644EA0(v58, v75, 1, 0);
    v5 = sub_1632080(a2, v63, v60, v4, 0);
    v6 = v5;
    if ( v73 != v75 )
    {
      v64 = v5;
      _libc_free((unsigned __int64)v73);
      v6 = v64;
    }
    v7 = sub_1B28080(v6);
    v8 = v82;
    a1[v2 + 113] = v7;
    if ( v8 != (__int64 (__fastcall **)())&v84 )
      j_j___libc_free_0(v8, v84 + 1);
    v57 = a1[61];
    v59 = (__int64 *)sub_1643270(v77);
    sub_8FD6D0((__int64)&v73, "__asan_stack_free_", &v70);
    v84 = v57;
    v85 = v57;
    v61 = v74;
    v65 = (__int64)v73;
    v82 = (__int64 (__fastcall **)())&v84;
    v83 = 0x200000002LL;
    v9 = sub_1644EA0(v59, &v84, 2, 0);
    v10 = sub_1632080(a2, v65, v61, v9, 0);
    v11 = v10;
    if ( v82 != (__int64 (__fastcall **)())&v84 )
    {
      v66 = v10;
      _libc_free((unsigned __int64)v82);
      v11 = v66;
    }
    v12 = sub_1B28080(v11);
    v13 = v73;
    a1[v2 + 124] = v12;
    if ( v13 != v75 )
      j_j___libc_free_0(v13, v75[0] + 1LL);
    if ( v70 != v72 )
      j_j___libc_free_0(v70, v72[0] + 1LL);
    ++v2;
  }
  while ( v2 != 11 );
  if ( *(_BYTE *)(a1[1] + 230LL) )
  {
    v43 = a1[61];
    v44 = (__int64 *)sub_1643270(v77);
    v45 = sub_18093A0(a2, (__int64)"__asan_poison_stack_memory", 26, 0, v44, v43, v43);
    v46 = sub_1B28080(v45);
    v47 = v77;
    v48 = a1[61];
    a1[391] = v46;
    v49 = (__int64 *)sub_1643270(v47);
    v50 = sub_18093A0(a2, (__int64)"__asan_unpoison_stack_memory", 28, 0, v49, v48, v48);
    a1[392] = sub_1B28080(v50);
  }
  v67 = (char *)&unk_42B8410;
  for ( i = 0; ; i = *(int *)v67 )
  {
    v62 = i;
    sub_222DF20(v94);
    v95 = 0;
    v82 = (__int64 (__fastcall **)())qword_4A071C8;
    v94[0] = off_4A06798;
    v94[27] = 0;
    v96 = 0;
    v97 = 0;
    v98 = 0;
    v99 = 0;
    *(__int64 (__fastcall ***)())((char *)&v82 + qword_4A071C8[-3]) = (__int64 (__fastcall **)())&unk_4A071F0;
    sub_222DD70((char *)&v82 + (_QWORD)*(v82 - 3), 0);
    v84 = 0;
    v85 = 0;
    v86 = 0;
    v82 = off_4A07238;
    v87 = 0;
    v88 = 0;
    v94[0] = off_4A07260;
    v89 = 0;
    v83 = (__int64)off_4A07480;
    sub_220A990(v90);
    v91 = 16;
    LOBYTE(v93[0]) = 0;
    v83 = (__int64)off_4A07080;
    v92[1] = 0;
    v92[0] = v93;
    sub_222DD70(v94, &v83);
    sub_223E0D0(&v82, "__asan_set_shadow_", 18);
    v25 = (unsigned __int64)v82;
    v26 = (__int64)*(v82 - 3);
    *(__int64 *)((char *)&v84 + v26) = 2;
    v27 = (char *)&v82 + *(_QWORD *)(v25 - 24);
    if ( !v27[225] )
    {
      v30 = *((_QWORD *)v27 + 30);
      if ( !v30 )
        sub_426219();
      if ( !*(_BYTE *)(v30 + 56) )
      {
        v55 = *((_QWORD *)v27 + 30);
        sub_2216D60(v30, "__asan_set_shadow_", v26, v23, v24);
        v31 = *(__int64 (__fastcall **)(__int64, unsigned int))(*(_QWORD *)v55 + 48LL);
        if ( v31 != sub_CE72A0 )
          v31(v55, 32u);
        v25 = (unsigned __int64)v82;
      }
      v27[225] = 1;
    }
    v27[224] = 48;
    *(_DWORD *)((char *)&v85 + *(_QWORD *)(v25 - 24)) = *(_DWORD *)((_BYTE *)&v85 + *(_QWORD *)(v25 - 24)) & 0xFFFFFFB5
                                                      | 8;
    sub_223E760(&v82, v62);
    v28 = a1[61];
    v29 = sub_1643270(v77);
    LOBYTE(v72[0]) = 0;
    v71 = 0;
    v70 = v72;
    if ( v88 )
    {
      v54 = (__int64 *)v29;
      if ( v88 > v86 )
        v15 = v88 - v87;
      else
        v15 = v86 - v87;
      sub_2241130(&v70, 0, 0, v87, v15);
      v16 = v54;
    }
    else
    {
      v56 = (__int64 *)v29;
      sub_2240AE0(&v70, v92);
      v16 = v56;
    }
    v75[0] = v28;
    v52 = v71;
    v53 = (__int64)v70;
    v75[1] = v28;
    v73 = v75;
    v74 = 0x200000002LL;
    v17 = sub_1644EA0(v16, v75, 2, 0);
    v18 = v53;
    v19 = sub_1632080(a2, v53, v52, v17, 0);
    if ( v73 != v75 )
      _libc_free((unsigned __int64)v73);
    v20 = sub_1B28080(v19);
    v21 = v62;
    v22 = v70;
    a1[v62 + 135] = v20;
    if ( v22 != v72 )
    {
      v18 = v72[0] + 1LL;
      j_j___libc_free_0(v22, v72[0] + 1LL);
    }
    v82 = off_4A07238;
    v94[0] = off_4A07260;
    v83 = (__int64)off_4A07080;
    if ( (_QWORD *)v92[0] != v93 )
    {
      v18 = v93[0] + 1LL;
      j_j___libc_free_0(v92[0], v93[0] + 1LL);
    }
    v83 = (__int64)off_4A07480;
    sub_2209150(v90, v18, v21);
    v82 = (__int64 (__fastcall **)())qword_4A071C8;
    *(__int64 (__fastcall ***)())((char *)&v82 + qword_4A071C8[-3]) = (__int64 (__fastcall **)())&unk_4A071F0;
    v94[0] = off_4A06798;
    sub_222E050(v94);
    v67 += 4;
    if ( v67 == "<unknown type>" )
      break;
  }
  v32 = a1[61];
  v33 = (__int64 *)sub_1643270(v77);
  v34 = sub_18093A0(a2, (__int64)"__asan_alloca_poison", 20, 0, v33, v32, v32);
  v35 = sub_1B28080(v34);
  v36 = v77;
  v37 = a1[61];
  a1[393] = v35;
  v38 = (__int64 *)sub_1643270(v36);
  v39 = sub_18093A0(a2, (__int64)"__asan_allocas_unpoison", 23, 0, v38, v37, v37);
  v40 = sub_1B28080(v39);
  v41 = v76[0];
  a1[394] = v40;
  result = v51;
  if ( v41 )
    return sub_161E7C0((__int64)v76, v41);
  return result;
}
