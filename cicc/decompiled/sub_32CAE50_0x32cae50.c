// Function: sub_32CAE50
// Address: 0x32cae50
//
__int64 __fastcall sub_32CAE50(__int64 *a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 *v4; // rdi
  int v5; // r10d
  __int64 v6; // r13
  unsigned int v7; // r15d
  __int64 v8; // r12
  unsigned int v9; // r14d
  __int64 v10; // rax
  __int64 (*v11)(); // rax
  __int64 result; // rax
  char v13; // al
  __int64 v14; // rax
  int v15; // edx
  __int64 v16; // rax
  __int16 v17; // dx
  __int64 v18; // rax
  bool v19; // al
  __int64 v20; // rax
  __int16 v21; // dx
  __int64 v22; // rax
  bool v23; // al
  __int64 v24; // rsi
  unsigned int v25; // eax
  __int64 v26; // r8
  int v27; // r10d
  unsigned int v28; // r15d
  __int64 v29; // rcx
  __int64 v30; // rdx
  __int64 v31; // rax
  __int64 v32; // rdx
  unsigned int v33; // eax
  unsigned __int64 *v34; // r11
  int v35; // r10d
  signed __int64 v36; // rsi
  int v37; // eax
  __int64 v38; // r13
  __int64 v39; // rdi
  __int128 v40; // rax
  int v41; // r9d
  int v42; // esi
  __int64 v43; // r13
  unsigned int v44; // edx
  __int64 v45; // r8
  __int64 v46; // r9
  unsigned __int16 v47; // ax
  __int64 v48; // rdx
  unsigned int v49; // eax
  __int64 v50; // r8
  __int64 v51; // rcx
  __int64 v52; // rdx
  int v53; // eax
  __int64 v54; // rdi
  __int64 v55; // rax
  __int64 v56; // rax
  unsigned int v57; // eax
  int v58; // eax
  __int64 v59; // rdi
  unsigned int v60; // r15d
  unsigned __int64 v61; // [rsp+8h] [rbp-B8h]
  int v62; // [rsp+10h] [rbp-B0h]
  int v63; // [rsp+10h] [rbp-B0h]
  int v64; // [rsp+10h] [rbp-B0h]
  unsigned int v65; // [rsp+10h] [rbp-B0h]
  unsigned __int64 v66; // [rsp+10h] [rbp-B0h]
  int v67; // [rsp+18h] [rbp-A8h]
  int v68; // [rsp+18h] [rbp-A8h]
  int v69; // [rsp+18h] [rbp-A8h]
  int v70; // [rsp+18h] [rbp-A8h]
  int v71; // [rsp+18h] [rbp-A8h]
  int v72; // [rsp+18h] [rbp-A8h]
  int v73; // [rsp+18h] [rbp-A8h]
  unsigned int v74; // [rsp+18h] [rbp-A8h]
  int v75; // [rsp+18h] [rbp-A8h]
  int v76; // [rsp+18h] [rbp-A8h]
  int v77; // [rsp+20h] [rbp-A0h]
  __int128 v78; // [rsp+20h] [rbp-A0h]
  __int64 v79; // [rsp+20h] [rbp-A0h]
  __int64 v80; // [rsp+20h] [rbp-A0h]
  unsigned __int64 v81; // [rsp+28h] [rbp-98h]
  int v82; // [rsp+40h] [rbp-80h] BYREF
  __int64 v83; // [rsp+48h] [rbp-78h]
  unsigned __int64 v84; // [rsp+50h] [rbp-70h] BYREF
  unsigned int v85; // [rsp+58h] [rbp-68h]
  __int64 v86; // [rsp+60h] [rbp-60h] BYREF
  char v87; // [rsp+68h] [rbp-58h]
  signed __int64 v88; // [rsp+70h] [rbp-50h] BYREF
  __int64 v89; // [rsp+78h] [rbp-48h]
  __int64 v90; // [rsp+80h] [rbp-40h] BYREF
  __int64 v91; // [rsp+88h] [rbp-38h]

  v3 = *(_QWORD *)(a2 + 40);
  v4 = (__int64 *)a1[1];
  v5 = *(_DWORD *)(a2 + 24);
  v6 = *(_QWORD *)v3;
  v7 = *(_DWORD *)(v3 + 8);
  v8 = *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL);
  v9 = **(unsigned __int16 **)(a2 + 48);
  v10 = *v4;
  if ( v5 == 245 )
  {
    v11 = *(__int64 (**)())(v10 + 1600);
    if ( v11 == sub_2D566B0 )
      goto LABEL_3;
  }
  else
  {
    v11 = *(__int64 (**)())(v10 + 1592);
    if ( v11 == sub_2FE3530 )
      goto LABEL_3;
  }
  v77 = *(_DWORD *)(a2 + 24);
  a2 = **(unsigned __int16 **)(a2 + 48);
  v13 = ((__int64 (__fastcall *)(__int64 *, _QWORD, __int64))v11)(v4, v9, v8);
  v5 = v77;
  if ( v13 )
    return 0;
LABEL_3:
  if ( *(_DWORD *)(v6 + 24) != 234 )
    return 0;
  v14 = *(_QWORD *)(v6 + 56);
  if ( !v14 )
    return 0;
  v15 = 1;
  do
  {
    if ( *(_DWORD *)(v14 + 8) == v7 )
    {
      if ( !v15 )
        return 0;
      v14 = *(_QWORD *)(v14 + 32);
      if ( !v14 )
        goto LABEL_18;
      if ( v7 == *(_DWORD *)(v14 + 8) )
        return 0;
      v15 = 0;
    }
    v14 = *(_QWORD *)(v14 + 32);
  }
  while ( v14 );
  if ( v15 == 1 )
    return 0;
LABEL_18:
  v78 = (__int128)_mm_loadu_si128((const __m128i *)*(_QWORD *)(v6 + 40));
  v16 = *(_QWORD *)(**(_QWORD **)(v6 + 40) + 48LL) + 16LL * *(unsigned int *)(*(_QWORD *)(v6 + 40) + 8LL);
  v17 = *(_WORD *)v16;
  v18 = *(_QWORD *)(v16 + 8);
  LOWORD(v82) = v17;
  v83 = v18;
  if ( !v17 )
  {
    v62 = v5;
    if ( sub_3007070((__int64)&v82) )
    {
      v19 = sub_30070B0((__int64)&v82);
      v5 = v62;
      if ( !v19 )
        goto LABEL_21;
    }
    return 0;
  }
  v47 = v17 - 17;
  if ( (unsigned __int16)(v17 - 2) > 7u && v47 > 0x6Cu && (unsigned __int16)(v17 - 176) > 0x1Fu || v47 <= 0xD3u )
    return 0;
LABEL_21:
  v85 = 1;
  v20 = *(_QWORD *)(v6 + 48) + 16LL * v7;
  v84 = 0;
  v21 = *(_WORD *)v20;
  v22 = *(_QWORD *)(v20 + 8);
  LOWORD(v90) = v21;
  v91 = v22;
  if ( v21 )
  {
    v23 = (unsigned __int16)(v21 - 17) <= 0xD3u;
  }
  else
  {
    v72 = v5;
    v23 = sub_30070B0((__int64)&v90);
    v5 = v72;
  }
  v67 = v5;
  if ( v23 )
  {
    v24 = v7;
    v25 = sub_3263630(v6, v7);
    v27 = v67;
    LODWORD(v91) = v25;
    v28 = v25;
    if ( v25 > 0x40 )
    {
      v24 = 0;
      sub_C43690((__int64)&v90, 0, 0);
      v29 = v28 - 1;
      v27 = v67;
      v30 = 1LL << ((unsigned __int8)v28 - 1);
      if ( (unsigned int)v91 > 0x40 )
      {
        v55 = (unsigned int)v29 >> 6;
        v29 = v90;
        *(_QWORD *)(v90 + 8 * v55) |= v30;
        goto LABEL_27;
      }
    }
    else
    {
      v90 = 0;
      v29 = v25 - 1;
      v30 = 1LL << ((unsigned __int8)v25 - 1);
    }
    v90 |= v30;
LABEL_27:
    if ( v85 > 0x40 && v84 )
    {
      v68 = v27;
      j_j___libc_free_0_0(v84);
      v27 = v68;
    }
    v84 = v90;
    v85 = v91;
    if ( v27 == 245 )
    {
      if ( (unsigned int)v91 > 0x40 )
      {
        v24 = (__int64)&v84;
        sub_C43780((__int64)&v90, (const void **)&v84);
        v27 = 245;
      }
      v76 = v27;
      sub_987160((__int64)&v90, v24, v30, v29, v26);
      v60 = v91;
      LODWORD(v91) = 0;
      v27 = v76;
      if ( v85 > 0x40 && v84 )
      {
        v66 = v90;
        j_j___libc_free_0_0(v84);
        v85 = v60;
        v27 = v76;
        v84 = v66;
        if ( (unsigned int)v91 > 0x40 && v90 )
        {
          j_j___libc_free_0_0(v90);
          v27 = v76;
        }
      }
      else
      {
        v84 = v90;
        v85 = v60;
      }
    }
    if ( (_WORD)v82 )
    {
      if ( (_WORD)v82 == 1 || (unsigned __int16)(v82 - 504) <= 7u )
        BUG();
      v32 = 16LL * ((unsigned __int16)v82 - 1);
      v31 = *(_QWORD *)&byte_444C4A0[v32];
      LOBYTE(v32) = byte_444C4A0[v32 + 8];
    }
    else
    {
      v69 = v27;
      v31 = sub_3007260((__int64)&v82);
      v27 = v69;
      v90 = v31;
      v91 = v32;
    }
    v63 = v27;
    v87 = v32;
    v86 = v31;
    v33 = sub_CA1930(&v86);
    sub_C47700((__int64)&v88, v33, (__int64)&v84);
    v34 = &v84;
    v35 = v63;
    if ( v85 > 0x40 && v84 )
    {
      j_j___libc_free_0_0(v84);
      v34 = &v84;
      v35 = v63;
    }
    v36 = *(_QWORD *)(v6 + 80);
    v84 = v88;
    v88 = v36;
    v85 = v89;
    if ( !v36 )
      goto LABEL_38;
    goto LABEL_37;
  }
  v88 = sub_2D5B750((unsigned __int16 *)&v82);
  v89 = v48;
  v49 = sub_CA1930(&v88);
  v35 = v67;
  LODWORD(v91) = v49;
  if ( v49 > 0x40 )
  {
    a2 = 0;
    v64 = v67;
    v74 = v49;
    sub_C43690((__int64)&v90, 0, 0);
    v35 = v64;
    v51 = v74 - 1;
    v52 = 1LL << ((unsigned __int8)v74 - 1);
    if ( (unsigned int)v91 > 0x40 )
    {
      v51 = v90;
      *(_QWORD *)(v90 + 8LL * ((v74 - 1) >> 6)) |= v52;
      goto LABEL_55;
    }
  }
  else
  {
    v90 = 0;
    v51 = v49 - 1;
    v52 = 1LL << ((unsigned __int8)v49 - 1);
  }
  v90 |= v52;
LABEL_55:
  if ( v85 > 0x40 && v84 )
  {
    v73 = v35;
    j_j___libc_free_0_0(v84);
    v35 = v73;
  }
  v84 = v90;
  v85 = v91;
  if ( v35 != 245 )
  {
    v36 = *(_QWORD *)(v6 + 80);
    v88 = v36;
    if ( !v36 )
    {
      v53 = *(_DWORD *)(v6 + 72);
      v38 = *a1;
      v54 = *a1;
      LODWORD(v89) = v53;
      *(_QWORD *)&v40 = sub_34007B0(v54, (unsigned int)&v84, (unsigned int)&v88, v82, v83, 0, 0);
      v42 = 188;
      goto LABEL_39;
    }
LABEL_37:
    v70 = v35;
    sub_B96E90((__int64)&v88, v36, 1);
    LODWORD(v34) = (unsigned int)&v84;
    v35 = v70;
LABEL_38:
    v37 = *(_DWORD *)(v6 + 72);
    v38 = *a1;
    v39 = *a1;
    v71 = v35;
    LODWORD(v89) = v37;
    *(_QWORD *)&v40 = sub_34007B0(v39, (_DWORD)v34, (unsigned int)&v88, v82, v83, 0, 0);
    v42 = 2 * (v71 != 245) + 186;
    goto LABEL_39;
  }
  if ( (unsigned int)v91 > 0x40 )
  {
    a2 = (__int64)&v84;
    sub_C43780((__int64)&v90, (const void **)&v84);
    v35 = 245;
  }
  v75 = v35;
  sub_987160((__int64)&v90, a2, v52, v51, v50);
  v57 = v91;
  LODWORD(v91) = 0;
  v35 = v75;
  if ( v85 > 0x40 && v84 )
  {
    v61 = v90;
    v65 = v57;
    j_j___libc_free_0_0(v84);
    v35 = v75;
    v84 = v61;
    v85 = v65;
    if ( (unsigned int)v91 > 0x40 && v90 )
    {
      j_j___libc_free_0_0(v90);
      v35 = v75;
    }
  }
  else
  {
    v84 = v90;
    v85 = v57;
  }
  v36 = *(_QWORD *)(v6 + 80);
  v88 = v36;
  if ( v36 )
    goto LABEL_37;
  v58 = *(_DWORD *)(v6 + 72);
  v38 = *a1;
  v59 = *a1;
  LODWORD(v89) = v58;
  *(_QWORD *)&v40 = sub_34007B0(v59, (unsigned int)&v84, (unsigned int)&v88, v82, v83, 0, 0);
  v42 = 186;
LABEL_39:
  v43 = sub_3406EB0(v38, v42, (unsigned int)&v88, v82, v83, v41, v78, v40);
  v81 = v44 | *((_QWORD *)&v78 + 1) & 0xFFFFFFFF00000000LL;
  if ( *(_DWORD *)(v43 + 24) != 328 )
  {
    v86 = v43;
    sub_32B3B20((__int64)(a1 + 71), &v86);
    v46 = *(unsigned int *)(v43 + 88);
    if ( (int)v46 < 0 )
    {
      *(_DWORD *)(v43 + 88) = *((_DWORD *)a1 + 12);
      v56 = *((unsigned int *)a1 + 12);
      if ( v56 + 1 > (unsigned __int64)*((unsigned int *)a1 + 13) )
      {
        sub_C8D5F0((__int64)(a1 + 5), a1 + 7, v56 + 1, 8u, v45, v46);
        v56 = *((unsigned int *)a1 + 12);
      }
      *(_QWORD *)(a1[5] + 8 * v56) = v43;
      ++*((_DWORD *)a1 + 12);
    }
  }
  result = sub_33FB890(*a1, v9, v8, v43, v81);
  if ( v88 )
  {
    v79 = result;
    sub_B91220((__int64)&v88, v88);
    result = v79;
  }
  if ( v85 > 0x40 )
  {
    if ( v84 )
    {
      v80 = result;
      j_j___libc_free_0_0(v84);
      return v80;
    }
  }
  return result;
}
