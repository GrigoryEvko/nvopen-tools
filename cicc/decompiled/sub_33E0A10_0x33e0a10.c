// Function: sub_33E0A10
// Address: 0x33e0a10
//
__int64 __fastcall sub_33E0A10(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  unsigned int v4; // r14d
  unsigned __int16 *v8; // rdx
  int v9; // eax
  __int64 *v10; // rbx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // rdx
  char v16; // cl
  unsigned __int8 v17; // r8
  __int64 v18; // r8
  __int64 v19; // r9
  int v20; // eax
  unsigned int v21; // r13d
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 *v24; // rdx
  __int64 v25; // rax
  __int64 v26; // r8
  __int64 v27; // r9
  __int64 v28; // rdx
  unsigned int v29; // ebx
  __int64 v30; // rax
  __int64 (*v31)(); // rax
  __int64 v32; // rcx
  __int64 v33; // r8
  __int64 v34; // rax
  __int64 v35; // rax
  unsigned int v36; // edx
  __int64 v37; // rdi
  unsigned int v38; // ebx
  __int64 v39; // rdx
  char *v40; // rcx
  __int64 v41; // rax
  __int64 v42; // rax
  int v43; // eax
  __int64 v44; // rdx
  char *v45; // r14
  int v46; // eax
  char *v47; // rcx
  __int64 v48; // rdx
  int v49; // eax
  char *v50; // rcx
  __int64 v51; // rdx
  int v52; // eax
  char *v53; // rcx
  signed __int64 v54; // rax
  int v55; // eax
  int v56; // r14d
  int v57; // eax
  int v58; // r15d
  int v59; // eax
  int v60; // r15d
  int v61; // eax
  int v62; // r15d
  unsigned int v63; // r13d
  __int64 v64; // r8
  __int64 v65; // r9
  int v66; // r14d
  int v67; // edx
  int v68; // edx
  __int64 *v69; // rax
  __int64 v70; // rcx
  __int64 v71; // r8
  int v72; // r15d
  int v73; // eax
  int v74; // r15d
  int v75; // eax
  int v76; // r14d
  __int64 *v77; // rax
  int v78; // eax
  int v79; // r14d
  char *v80; // [rsp+8h] [rbp-118h]
  char *v81; // [rsp+10h] [rbp-110h]
  unsigned int v82; // [rsp+1Ch] [rbp-104h]
  char v83; // [rsp+20h] [rbp-100h]
  __int64 v84; // [rsp+20h] [rbp-100h]
  char *v85; // [rsp+20h] [rbp-100h]
  char *v86; // [rsp+20h] [rbp-100h]
  char *v87; // [rsp+20h] [rbp-100h]
  char *v88; // [rsp+20h] [rbp-100h]
  char *v89; // [rsp+20h] [rbp-100h]
  char *v90; // [rsp+20h] [rbp-100h]
  char *v91; // [rsp+20h] [rbp-100h]
  int v92; // [rsp+20h] [rbp-100h]
  __m128i v94; // [rsp+30h] [rbp-F0h]
  __m128i v95; // [rsp+40h] [rbp-E0h]
  __int16 v96; // [rsp+50h] [rbp-D0h] BYREF
  __int64 *v97; // [rsp+58h] [rbp-C8h]
  __int64 v98; // [rsp+60h] [rbp-C0h] BYREF
  __int64 v99; // [rsp+68h] [rbp-B8h]
  __int64 v100; // [rsp+70h] [rbp-B0h]
  __int64 v101; // [rsp+78h] [rbp-A8h]
  int v102; // [rsp+80h] [rbp-A0h] BYREF
  __int64 v103; // [rsp+88h] [rbp-98h] BYREF
  __int64 (__fastcall *v104)(_QWORD *, _DWORD *, int); // [rsp+90h] [rbp-90h]
  __int64 (__fastcall *v105)(unsigned int *, __int64); // [rsp+98h] [rbp-88h]
  __int64 v106; // [rsp+A0h] [rbp-80h]
  unsigned __int64 v107; // [rsp+B0h] [rbp-70h] BYREF
  __int64 *v108; // [rsp+B8h] [rbp-68h]
  void (__fastcall *v109)(unsigned __int64 *, unsigned __int64 *, __int64); // [rsp+C0h] [rbp-60h] BYREF
  __int64 v110[4]; // [rsp+C8h] [rbp-58h] BYREF
  int v111; // [rsp+E8h] [rbp-38h]
  char v112; // [rsp+ECh] [rbp-34h]

  if ( a4 > 5 )
    return 0;
  v8 = (unsigned __int16 *)(*(_QWORD *)(a2 + 48) + 16LL * (unsigned int)a3);
  v9 = *v8;
  v10 = (__int64 *)*((_QWORD *)v8 + 1);
  v96 = v9;
  v97 = v10;
  if ( (_WORD)v9 )
  {
    if ( (unsigned __int16)(v9 - 17) > 0xD3u )
    {
      LOWORD(v107) = v9;
      v108 = v10;
      goto LABEL_7;
    }
    LOWORD(v9) = word_4456580[v9 - 1];
    v24 = 0;
  }
  else
  {
    if ( !sub_30070B0((__int64)&v96) )
    {
      v108 = v10;
      LOWORD(v107) = 0;
LABEL_12:
      v11 = sub_3007260((__int64)&v107);
      v100 = v11;
      v101 = v15;
      goto LABEL_13;
    }
    LOWORD(v9) = sub_3009970((__int64)&v96, a2, v12, v13, v14);
  }
  LOWORD(v107) = v9;
  v108 = v24;
  if ( !(_WORD)v9 )
    goto LABEL_12;
LABEL_7:
  if ( (_WORD)v9 == 1 || (unsigned __int16)(v9 - 504) <= 7u )
    BUG();
  v11 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v9 - 16];
LABEL_13:
  v82 = v11;
  v102 = v11;
  v105 = sub_33C9430;
  v104 = sub_33C7F50;
  v109 = 0;
  sub_33C7F50(&v107, &v102, 2);
  v110[0] = (__int64)v105;
  v109 = (void (__fastcall *)(_QWORD, _QWORD, _QWORD))v104;
  v83 = sub_33CA8D0((_QWORD *)a2, a3, (__int64)&v107, v16, v17);
  if ( v109 )
    v109(&v107, &v107, 3);
  if ( v104 )
    v104(&v102, &v102, 3);
  if ( v83 )
    return 1;
  v20 = *(_DWORD *)(a2 + 24);
  if ( v20 == 190 )
  {
    v25 = sub_33DFBC0(**(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL), 0, 0, v18, v19);
    if ( !v25 )
      goto LABEL_46;
    v28 = *(_QWORD *)(v25 + 96);
    v29 = *(_DWORD *)(v28 + 32);
    if ( v29 <= 0x40 )
    {
      v30 = *(_QWORD *)(v28 + 24);
    }
    else
    {
      v84 = *(_QWORD *)(v25 + 96);
      if ( v29 - (unsigned int)sub_C444A0(v28 + 24) > 0x40 )
        goto LABEL_46;
      v30 = **(_QWORD **)(v84 + 24);
    }
    if ( v30 == 1 )
      return 1;
LABEL_46:
    if ( (unsigned __int8)sub_33E0A10(
                            a1,
                            **(_QWORD **)(a2 + 40),
                            *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL),
                            a4 + 1,
                            v26,
                            v27) )
      return (unsigned int)sub_33DE9F0(a1, a2, a3, a4);
    return 0;
  }
  if ( v20 == 192 )
  {
    v34 = sub_33DFBC0(**(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL), 0, 0, v18, v19);
    if ( v34 )
    {
      v35 = *(_QWORD *)(v34 + 96);
      v36 = *(_DWORD *)(v35 + 32);
      v37 = *(_QWORD *)(v35 + 24);
      v38 = v36 - 1;
      if ( v36 <= 0x40 )
      {
        if ( v37 == 1LL << v38 )
          return 1;
      }
      else if ( (*(_QWORD *)(v37 + 8LL * (v38 >> 6)) & (1LL << v38)) != 0 && (unsigned int)sub_C44590(v35 + 24) == v38 )
      {
        return 1;
      }
    }
    goto LABEL_46;
  }
  if ( (unsigned int)(v20 - 193) <= 1 )
    return (unsigned int)sub_33E0A10(
                           a1,
                           **(_QWORD **)(a2 + 40),
                           *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL),
                           a4 + 1,
                           v18,
                           v19);
  if ( v20 != 156 )
  {
LABEL_22:
    if ( v20 == 168 )
    {
      v39 = **(_QWORD **)(a2 + 40);
      LOBYTE(v18) = *(_DWORD *)(v39 + 24) == 35 || *(_DWORD *)(v39 + 24) == 11;
      v4 = v18;
      if ( !(_BYTE)v18 )
        goto LABEL_38;
      sub_C44AB0((__int64)&v107, *(_QWORD *)(v39 + 96) + 24LL, v82);
      if ( (unsigned int)v108 > 0x40 )
      {
        v66 = sub_C44630((__int64)&v107);
        sub_969240((__int64 *)&v107);
        if ( v66 == 1 )
          return 1;
        v20 = *(_DWORD *)(a2 + 24);
      }
      else
      {
        if ( v107 && (v107 & (v107 - 1)) == 0 )
        {
          sub_969240((__int64 *)&v107);
          return v4;
        }
        sub_969240((__int64 *)&v107);
        v20 = *(_DWORD *)(a2 + 24);
      }
    }
    if ( v20 != 373 )
      goto LABEL_24;
    v31 = *(__int64 (**)())(**(_QWORD **)(a1 + 16) + 224LL);
    if ( v31 == sub_2FE2F60 )
      goto LABEL_38;
    if ( !(unsigned __int8)v31()
      || !(unsigned __int8)sub_33E0A10(
                             a1,
                             **(_QWORD **)(a2 + 40),
                             *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL),
                             a4 + 1,
                             v18,
                             v19) )
    {
      v20 = *(_DWORD *)(a2 + 24);
LABEL_24:
      if ( (unsigned int)(v20 - 180) <= 3 )
      {
        v21 = a4 + 1;
        if ( (unsigned __int8)sub_33E0A10(
                                a1,
                                *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
                                *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL),
                                v21,
                                v18,
                                v19) )
          return (unsigned int)sub_33E0A10(
                                 a1,
                                 **(_QWORD **)(a2 + 40),
                                 *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL),
                                 v21,
                                 v22,
                                 v23);
        return 0;
      }
      if ( (unsigned int)(v20 - 205) <= 1 )
      {
        v63 = a4 + 1;
        if ( (unsigned __int8)sub_33E0A10(
                                a1,
                                *(_QWORD *)(*(_QWORD *)(a2 + 40) + 80LL),
                                *(_QWORD *)(*(_QWORD *)(a2 + 40) + 88LL),
                                v63,
                                v18,
                                v19) )
          return (unsigned int)sub_33E0A10(
                                 a1,
                                 *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
                                 *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL),
                                 v63,
                                 v64,
                                 v65);
        return 0;
      }
LABEL_38:
      v98 = 0;
      LODWORD(v108) = 64;
      LODWORD(v99) = 0;
      v107 = 0;
      v102 = 57;
      sub_9865C0((__int64)&v103, (__int64)&v107);
      BYTE4(v106) = 0;
      v105 = (__int64 (__fastcall *)(unsigned int *, __int64))&v98;
      sub_969240((__int64 *)&v107);
      LODWORD(v107) = 186;
      v108 = &v98;
      LODWORD(v109) = v102;
      sub_9865C0((__int64)v110, (__int64)&v103);
      v112 = 0;
      v110[2] = (__int64)v105;
      v110[3] = v106;
      if ( (_DWORD)v107 != *(_DWORD *)(a2 + 24) )
        goto LABEL_39;
      v69 = v108;
      v95 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 40));
      *v108 = v95.m128i_i64[0];
      *((_DWORD *)v69 + 2) = v95.m128i_i32[2];
      if ( !sub_33D1E50((__int64)&v109, *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL), v95.m128i_u32[2], v32, v33) )
      {
        v77 = v108;
        v94 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a2 + 40) + 40LL));
        *v108 = v94.m128i_i64[0];
        *((_DWORD *)v77 + 2) = v94.m128i_i32[2];
        if ( !sub_33D1E50((__int64)&v109, **(_QWORD **)(a2 + 40), v94.m128i_u32[2], v70, v71) )
        {
LABEL_39:
          sub_969240(v110);
          sub_969240(&v103);
          goto LABEL_40;
        }
      }
      if ( v112 )
      {
        v92 = v111;
        v72 = *(_DWORD *)(a2 + 28) & v111;
        sub_969240(v110);
        sub_969240(&v103);
        if ( v92 != v72 )
        {
LABEL_40:
          if ( *(_DWORD *)(a2 + 24) != 214 )
            return 0;
          return (unsigned int)sub_33E0A10(
                                 a1,
                                 **(_QWORD **)(a2 + 40),
                                 *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL),
                                 a4 + 1,
                                 v18,
                                 v19);
        }
      }
      else
      {
        sub_969240(v110);
        sub_969240(&v103);
      }
      return (unsigned int)sub_33DE9F0(a1, v98, v99, a4);
    }
    return 1;
  }
  v40 = *(char **)(a2 + 40);
  v41 = 40LL * *(unsigned int *)(a2 + 64);
  v80 = &v40[v41];
  v42 = (__int64)(0xCCCCCCCCCCCCCCCDLL * (v41 >> 3)) >> 2;
  if ( v42 )
  {
    v81 = &v40[160 * v42];
    do
    {
      v43 = *(_DWORD *)(*(_QWORD *)v40 + 24LL);
      if ( v43 != 35 && v43 != 11 )
        goto LABEL_58;
      v85 = v40;
      sub_C44AB0((__int64)&v107, *(_QWORD *)(*(_QWORD *)v40 + 96LL) + 24LL, v82);
      v40 = v85;
      if ( (unsigned int)v108 > 0x40 )
      {
        v55 = sub_C44630((__int64)&v107);
        v40 = v85;
        v56 = v55;
        if ( v107 )
        {
          j_j___libc_free_0_0(v107);
          v40 = v85;
        }
        if ( v56 != 1 )
          goto LABEL_58;
      }
      else if ( !v107 || (v107 & (v107 - 1)) != 0 )
      {
        goto LABEL_58;
      }
      v44 = *((_QWORD *)v40 + 5);
      v45 = v40 + 40;
      v46 = *(_DWORD *)(v44 + 24);
      if ( v46 != 35 && v46 != 11 )
      {
LABEL_65:
        v40 = v45;
        goto LABEL_58;
      }
      v86 = v40;
      sub_C44AB0((__int64)&v107, *(_QWORD *)(v44 + 96) + 24LL, v82);
      v47 = v86;
      if ( (unsigned int)v108 > 0x40 )
      {
        v57 = sub_C44630((__int64)&v107);
        v47 = v86;
        v58 = v57;
        if ( v107 )
        {
          j_j___libc_free_0_0(v107);
          v47 = v86;
        }
        if ( v58 != 1 )
          goto LABEL_65;
      }
      else if ( !v107 || (v107 & (v107 - 1)) != 0 )
      {
        goto LABEL_65;
      }
      v48 = *((_QWORD *)v47 + 10);
      v45 = v47 + 80;
      v49 = *(_DWORD *)(v48 + 24);
      if ( v49 != 35 && v49 != 11 )
        goto LABEL_65;
      v87 = v47;
      sub_C44AB0((__int64)&v107, *(_QWORD *)(v48 + 96) + 24LL, v82);
      v50 = v87;
      if ( (unsigned int)v108 > 0x40 )
      {
        v59 = sub_C44630((__int64)&v107);
        v50 = v87;
        v60 = v59;
        if ( v107 )
        {
          j_j___libc_free_0_0(v107);
          v50 = v87;
        }
        if ( v60 != 1 )
          goto LABEL_65;
      }
      else if ( !v107 || (v107 & (v107 - 1)) != 0 )
      {
        goto LABEL_65;
      }
      v51 = *((_QWORD *)v50 + 15);
      v45 = v50 + 120;
      v52 = *(_DWORD *)(v51 + 24);
      if ( v52 != 11 && v52 != 35 )
        goto LABEL_65;
      v88 = v50;
      sub_C44AB0((__int64)&v107, *(_QWORD *)(v51 + 96) + 24LL, v82);
      v53 = v88;
      if ( (unsigned int)v108 > 0x40 )
      {
        v61 = sub_C44630((__int64)&v107);
        v53 = v88;
        v62 = v61;
        if ( v107 )
        {
          j_j___libc_free_0_0(v107);
          v53 = v88;
        }
        if ( v62 != 1 )
          goto LABEL_65;
      }
      else if ( !v107 || (v107 & (v107 - 1)) != 0 )
      {
        goto LABEL_65;
      }
      v40 = v53 + 160;
    }
    while ( v81 != v40 );
  }
  v54 = v80 - v40;
  if ( v80 - v40 != 80 )
  {
    if ( v54 != 120 )
    {
      if ( v54 != 40 )
        return 1;
      goto LABEL_83;
    }
    v67 = *(_DWORD *)(*(_QWORD *)v40 + 24LL);
    if ( v67 != 35 && v67 != 11 )
      goto LABEL_58;
    v90 = v40;
    sub_C44AB0((__int64)&v107, *(_QWORD *)(*(_QWORD *)v40 + 96LL) + 24LL, v82);
    v40 = v90;
    if ( (unsigned int)v108 > 0x40 )
    {
      v78 = sub_C44630((__int64)&v107);
      v40 = v90;
      v79 = v78;
      if ( v107 )
      {
        j_j___libc_free_0_0(v107);
        v40 = v90;
      }
      if ( v79 != 1 )
        goto LABEL_58;
    }
    else if ( !v107 || (v107 & (v107 - 1)) != 0 )
    {
      goto LABEL_58;
    }
    v40 += 40;
  }
  v68 = *(_DWORD *)(*(_QWORD *)v40 + 24LL);
  if ( v68 != 35 && v68 != 11 )
    goto LABEL_58;
  v91 = v40;
  sub_C44AB0((__int64)&v107, *(_QWORD *)(*(_QWORD *)v40 + 96LL) + 24LL, v82);
  v40 = v91;
  if ( (unsigned int)v108 > 0x40 )
  {
    v75 = sub_C44630((__int64)&v107);
    v40 = v91;
    v76 = v75;
    if ( v107 )
    {
      j_j___libc_free_0_0(v107);
      v40 = v91;
    }
    if ( v76 != 1 )
      goto LABEL_58;
  }
  else if ( !v107 || (v107 & (v107 - 1)) != 0 )
  {
    goto LABEL_58;
  }
  v40 += 40;
LABEL_83:
  LOBYTE(v18) = *(_DWORD *)(*(_QWORD *)v40 + 24LL) == 11 || *(_DWORD *)(*(_QWORD *)v40 + 24LL) == 35;
  v4 = v18;
  if ( !(_BYTE)v18 )
    goto LABEL_58;
  v89 = v40;
  sub_C44AB0((__int64)&v107, *(_QWORD *)(*(_QWORD *)v40 + 96LL) + 24LL, v82);
  v40 = v89;
  if ( (unsigned int)v108 <= 0x40 )
  {
    if ( v107 && (v107 & (v107 - 1)) == 0 )
      return v4;
LABEL_58:
    if ( v80 == v40 )
      return 1;
    v20 = *(_DWORD *)(a2 + 24);
    goto LABEL_22;
  }
  v73 = sub_C44630((__int64)&v107);
  v40 = v89;
  v74 = v73;
  if ( v107 )
  {
    j_j___libc_free_0_0(v107);
    v40 = v89;
  }
  if ( v74 != 1 )
    goto LABEL_58;
  return v4;
}
