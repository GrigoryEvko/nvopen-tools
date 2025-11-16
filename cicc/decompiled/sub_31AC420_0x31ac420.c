// Function: sub_31AC420
// Address: 0x31ac420
//
_QWORD *__fastcall sub_31AC420(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r10
  unsigned int v9; // esi
  __int64 v10; // r9
  int v11; // r11d
  __int64 v12; // rdi
  __int64 *v13; // r15
  __int64 v14; // rcx
  __int64 *v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // r14
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 *v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  _QWORD **v25; // r14
  __int64 v26; // r8
  char v27; // al
  _QWORD **v28; // r15
  unsigned int v29; // eax
  __int64 v30; // r8
  _QWORD **v31; // r9
  unsigned int v32; // eax
  __int64 v33; // r9
  unsigned int v34; // eax
  _QWORD **v35; // r9
  __int64 v36; // rax
  _QWORD *result; // rax
  __int64 *v38; // rdx
  __int64 v39; // r8
  __int64 v40; // r9
  __int64 v41; // rsi
  __int64 *v42; // rax
  __int64 v43; // rcx
  __int64 *v44; // rax
  __int64 v45; // rax
  __int64 v46; // r8
  __int64 v47; // r9
  __int64 v48; // rdi
  __int64 v49; // rcx
  __int64 v50; // rdx
  __int64 v51; // rax
  __int64 v52; // rax
  __int64 v53; // rsi
  __int64 v54; // rax
  unsigned int v55; // r15d
  _BYTE *v56; // rdi
  unsigned int v57; // r9d
  __int64 v58; // rax
  int v59; // eax
  int v60; // edx
  __int64 v61; // rdx
  __int64 *v62; // r10
  unsigned __int64 v63; // rdi
  unsigned __int64 v64; // rsi
  int v65; // eax
  __int64 v66; // rsi
  __int64 v67; // r14
  __int64 v68; // rax
  __int64 v69; // rax
  __int64 v70; // rdx
  _BYTE *v71; // rdi
  __int64 v72; // rax
  __int64 v73; // rax
  __int64 v74; // rax
  __int64 v75; // rax
  int v76; // eax
  int v77; // ecx
  __int64 v78; // rsi
  unsigned int v79; // eax
  __int64 v80; // rdi
  int v81; // r10d
  __int64 *v82; // r9
  unsigned __int64 v83; // r14
  __int64 v84; // rdi
  int v85; // eax
  int v86; // eax
  __int64 v87; // rsi
  int v88; // r9d
  unsigned int v89; // r14d
  __int64 *v90; // rdi
  __int64 v91; // rcx
  __int64 v92; // [rsp+8h] [rbp-F8h]
  __int64 v93; // [rsp+10h] [rbp-F0h]
  __int64 v94; // [rsp+10h] [rbp-F0h]
  unsigned int v95; // [rsp+10h] [rbp-F0h]
  char *v96; // [rsp+10h] [rbp-F0h]
  __int64 v97; // [rsp+10h] [rbp-F0h]
  _QWORD v99[2]; // [rsp+20h] [rbp-E0h] BYREF
  __int64 v100; // [rsp+30h] [rbp-D0h]
  __int128 v101; // [rsp+38h] [rbp-C8h]
  __int64 v102; // [rsp+48h] [rbp-B8h]
  _BYTE *v103; // [rsp+50h] [rbp-B0h]
  __int64 v104; // [rsp+58h] [rbp-A8h]
  _BYTE v105[16]; // [rsp+60h] [rbp-A0h] BYREF
  __int64 v106; // [rsp+70h] [rbp-90h] BYREF
  _QWORD v107[2]; // [rsp+78h] [rbp-88h] BYREF
  __int64 v108; // [rsp+88h] [rbp-78h]
  int v109; // [rsp+90h] [rbp-70h]
  __int64 v110; // [rsp+98h] [rbp-68h]
  __int64 v111; // [rsp+A0h] [rbp-60h]
  _BYTE *v112; // [rsp+A8h] [rbp-58h]
  __int64 v113; // [rsp+B0h] [rbp-50h]
  _BYTE v114[72]; // [rsp+B8h] [rbp-48h] BYREF

  v5 = a1 + 128;
  v9 = *(_DWORD *)(a1 + 152);
  if ( !v9 )
  {
    ++*(_QWORD *)(a1 + 128);
    goto LABEL_104;
  }
  v10 = v9 - 1;
  v11 = 1;
  v12 = *(_QWORD *)(a1 + 136);
  v13 = 0;
  v14 = (unsigned int)v10 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v15 = (__int64 *)(v12 + 16 * v14);
  v16 = *v15;
  if ( *v15 == a2 )
  {
LABEL_3:
    v17 = *((unsigned int *)v15 + 2);
    goto LABEL_4;
  }
  while ( v16 != -4096 )
  {
    if ( !v13 && v16 == -8192 )
      v13 = v15;
    a5 = (unsigned int)(v11 + 1);
    v14 = (unsigned int)v10 & (v11 + (_DWORD)v14);
    v15 = (__int64 *)(v12 + 16LL * (unsigned int)v14);
    v16 = *v15;
    if ( *v15 == a2 )
      goto LABEL_3;
    ++v11;
  }
  if ( !v13 )
    v13 = v15;
  v59 = *(_DWORD *)(a1 + 144);
  ++*(_QWORD *)(a1 + 128);
  v60 = v59 + 1;
  if ( 4 * (v59 + 1) >= 3 * v9 )
  {
LABEL_104:
    sub_29906F0(v5, 2 * v9);
    v76 = *(_DWORD *)(a1 + 152);
    if ( v76 )
    {
      v77 = v76 - 1;
      v78 = *(_QWORD *)(a1 + 136);
      v79 = (v76 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v60 = *(_DWORD *)(a1 + 144) + 1;
      v13 = (__int64 *)(v78 + 16LL * v79);
      v80 = *v13;
      if ( *v13 != a2 )
      {
        v81 = 1;
        v82 = 0;
        while ( v80 != -4096 )
        {
          if ( !v82 && v80 == -8192 )
            v82 = v13;
          v79 = v77 & (v81 + v79);
          v13 = (__int64 *)(v78 + 16LL * v79);
          v80 = *v13;
          if ( *v13 == a2 )
            goto LABEL_70;
          ++v81;
        }
        if ( v82 )
          v13 = v82;
      }
      goto LABEL_70;
    }
    goto LABEL_131;
  }
  if ( v9 - *(_DWORD *)(a1 + 148) - v60 <= v9 >> 3 )
  {
    sub_29906F0(v5, v9);
    v85 = *(_DWORD *)(a1 + 152);
    if ( v85 )
    {
      v86 = v85 - 1;
      v87 = *(_QWORD *)(a1 + 136);
      v88 = 1;
      v89 = v86 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v60 = *(_DWORD *)(a1 + 144) + 1;
      v90 = 0;
      v13 = (__int64 *)(v87 + 16LL * v89);
      v91 = *v13;
      if ( *v13 != a2 )
      {
        while ( v91 != -4096 )
        {
          if ( !v90 && v91 == -8192 )
            v90 = v13;
          v89 = v86 & (v88 + v89);
          v13 = (__int64 *)(v87 + 16LL * v89);
          v91 = *v13;
          if ( *v13 == a2 )
            goto LABEL_70;
          ++v88;
        }
        if ( v90 )
          v13 = v90;
      }
      goto LABEL_70;
    }
LABEL_131:
    ++*(_DWORD *)(a1 + 144);
    BUG();
  }
LABEL_70:
  *(_DWORD *)(a1 + 144) = v60;
  if ( *v13 != -4096 )
    --*(_DWORD *)(a1 + 148);
  *v13 = a2;
  v10 = (__int64)v114;
  *((_DWORD *)v13 + 2) = 0;
  v61 = *(unsigned int *)(a1 + 168);
  v14 = (__int64)v105;
  v62 = &v106;
  v63 = *(unsigned int *)(a1 + 172);
  v113 = 0x200000000LL;
  v64 = v61 + 1;
  v112 = v114;
  v104 = 0x200000000LL;
  v65 = v61;
  v102 = 0;
  v99[0] = 6;
  v99[1] = 0;
  v100 = 0;
  v103 = v105;
  v106 = a2;
  v107[0] = 6;
  v107[1] = 0;
  v108 = 0;
  v109 = 0;
  v110 = 0;
  v111 = 0;
  v101 = 0;
  if ( v61 + 1 > v63 )
  {
    v83 = *(_QWORD *)(a1 + 160);
    v84 = a1 + 160;
    if ( v83 > (unsigned __int64)&v106 || (unsigned __int64)&v106 >= v83 + 88 * v61 )
    {
      sub_31AA470(v84, v64, v61, (__int64)v105, a5, (__int64)v114);
      v61 = *(unsigned int *)(a1 + 168);
      v66 = *(_QWORD *)(a1 + 160);
      v62 = &v106;
      v14 = (__int64)v105;
      v10 = (__int64)v114;
      v65 = *(_DWORD *)(a1 + 168);
    }
    else
    {
      sub_31AA470(v84, v64, v61, (__int64)v105, a5, (__int64)v114);
      v66 = *(_QWORD *)(a1 + 160);
      v61 = *(unsigned int *)(a1 + 168);
      v10 = (__int64)v114;
      v14 = (__int64)v105;
      v62 = (_QWORD *)((char *)&v107[-1] + v66 - v83);
      v65 = *(_DWORD *)(a1 + 168);
    }
  }
  else
  {
    v66 = *(_QWORD *)(a1 + 160);
  }
  v67 = v66 + 88 * v61;
  if ( v67 )
  {
    v68 = *v62;
    *(_QWORD *)(v67 + 8) = 6;
    *(_QWORD *)(v67 + 16) = 0;
    *(_QWORD *)v67 = v68;
    v69 = v62[3];
    *(_QWORD *)(v67 + 24) = v69;
    if ( v69 != -4096 && v69 != 0 && v69 != -8192 )
    {
      v96 = (char *)v62;
      sub_BD6050((unsigned __int64 *)(v67 + 8), v62[1] & 0xFFFFFFFFFFFFFFF8LL);
      v14 = (__int64)v105;
      v10 = (__int64)v114;
      v62 = (__int64 *)v96;
    }
    *(_DWORD *)(v67 + 32) = *((_DWORD *)v62 + 8);
    *(_QWORD *)(v67 + 40) = v62[5];
    *(_QWORD *)(v67 + 48) = v62[6];
    *(_QWORD *)(v67 + 56) = v67 + 72;
    *(_QWORD *)(v67 + 64) = 0x200000000LL;
    v70 = *((unsigned int *)v62 + 16);
    if ( (_DWORD)v70 )
    {
      sub_31A3A30(v67 + 56, (char **)v62 + 7, v70, (__int64)v105, a5, (__int64)v114);
      v65 = *(_DWORD *)(a1 + 168);
      v14 = (__int64)v105;
      v10 = (__int64)v114;
    }
    else
    {
      v65 = *(_DWORD *)(a1 + 168);
    }
  }
  v71 = v112;
  *(_DWORD *)(a1 + 168) = v65 + 1;
  if ( v71 != v114 )
  {
    _libc_free((unsigned __int64)v71);
    v14 = (__int64)v105;
  }
  if ( v108 != -4096 && v108 != 0 && v108 != -8192 )
  {
    sub_BD60C0(v107);
    v14 = (__int64)v105;
  }
  if ( v103 != v105 )
    _libc_free((unsigned __int64)v103);
  LOBYTE(v14) = v100 != 0;
  if ( v100 != -4096 && v100 != 0 && v100 != -8192 )
    sub_BD60C0(v99);
  v17 = (unsigned int)(*(_DWORD *)(a1 + 168) - 1);
  *((_DWORD *)v13 + 2) = v17;
LABEL_4:
  v18 = *(_QWORD *)(a1 + 160) + 88 * v17;
  v19 = *(_QWORD *)(a3 + 16);
  v20 = *(_QWORD *)(v18 + 24);
  if ( v20 != v19 )
  {
    if ( v20 != -4096 && v20 != 0 && v20 != -8192 )
    {
      sub_BD60C0((_QWORD *)(v18 + 8));
      v19 = *(_QWORD *)(a3 + 16);
    }
    *(_QWORD *)(v18 + 24) = v19;
    LOBYTE(v14) = v19 != -4096;
    LOBYTE(v20) = v19 != 0;
    if ( ((v19 != 0) & (unsigned __int8)v14) != 0 && v19 != -8192 )
      sub_BD6050((unsigned __int64 *)(v18 + 8), *(_QWORD *)a3 & 0xFFFFFFFFFFFFFFF8LL);
  }
  *(_DWORD *)(v18 + 32) = *(_DWORD *)(a3 + 24);
  *(_QWORD *)(v18 + 40) = *(_QWORD *)(a3 + 32);
  *(_QWORD *)(v18 + 48) = *(_QWORD *)(a3 + 40);
  sub_31A3C40(v18 + 56, a3 + 48, v20, v14, a5, v10);
  if ( !*(_DWORD *)(a3 + 56) )
    goto LABEL_12;
  v41 = **(_QWORD **)(a3 + 48);
  if ( !*(_BYTE *)(a1 + 204) )
  {
LABEL_93:
    sub_C8CC70(a1 + 176, v41, (__int64)v21, v22, v23, v24);
    goto LABEL_12;
  }
  v42 = *(__int64 **)(a1 + 184);
  v22 = *(unsigned int *)(a1 + 196);
  v21 = &v42[v22];
  if ( v42 == v21 )
  {
LABEL_30:
    if ( (unsigned int)v22 < *(_DWORD *)(a1 + 192) )
    {
      *(_DWORD *)(a1 + 196) = v22 + 1;
      *v21 = v41;
      ++*(_QWORD *)(a1 + 176);
      goto LABEL_12;
    }
    goto LABEL_93;
  }
  while ( v41 != *v42 )
  {
    if ( v21 == ++v42 )
      goto LABEL_30;
  }
LABEL_12:
  v25 = *(_QWORD ***)(a2 + 8);
  v26 = sub_B43CC0(a2);
  v27 = *((_BYTE *)v25 + 8);
  if ( (v27 & 0xFD) == 0xC )
  {
    v28 = *(_QWORD ***)(a1 + 336);
    if ( v28 )
    {
      if ( v27 == 14 )
      {
        v97 = v26;
        v75 = sub_AE4420(v26, (__int64)*v25, *((_DWORD *)v25 + 2) >> 8);
        v30 = v97;
        v31 = (_QWORD **)v75;
      }
      else
      {
        v93 = v26;
        v29 = sub_BCB060((__int64)v25);
        v30 = v93;
        v31 = v25;
        if ( v29 <= 0x1F )
        {
          v73 = sub_BCB2D0(*v25);
          v30 = v93;
          v31 = (_QWORD **)v73;
        }
      }
      v94 = (__int64)v31;
      if ( *((_BYTE *)v28 + 8) == 14 )
      {
        v74 = sub_AE4420(v30, (__int64)*v28, *((_DWORD *)v28 + 2) >> 8);
        v33 = v94;
        v28 = (_QWORD **)v74;
      }
      else
      {
        v32 = sub_BCB060((__int64)v28);
        v33 = v94;
        if ( v32 <= 0x1F )
        {
          v72 = sub_BCB2D0(*v28);
          v33 = v94;
          v28 = (_QWORD **)v72;
        }
      }
      v92 = v33;
      v95 = sub_BCB060(v33);
      v34 = sub_BCB060((__int64)v28);
      v35 = (_QWORD **)v92;
      if ( v95 <= v34 )
        v35 = v28;
      *(_QWORD *)(a1 + 336) = v35;
    }
    else
    {
      if ( v27 == 14 )
      {
        v58 = sub_AE4420(v26, (__int64)*v25, *((_DWORD *)v25 + 2) >> 8);
      }
      else
      {
        v57 = sub_BCB060((__int64)v25);
        v58 = (__int64)v25;
        if ( v57 <= 0x1F )
          v58 = sub_BCB2D0(*v25);
      }
      *(_QWORD *)(a1 + 336) = v58;
    }
  }
  if ( *(_DWORD *)(a3 + 24) == 1 && sub_1023590(a3) )
  {
    v54 = sub_1023590(a3);
    v55 = *(_DWORD *)(v54 + 32);
    if ( v55 <= 0x40 )
    {
      if ( *(_QWORD *)(v54 + 24) != 1 )
        goto LABEL_24;
    }
    else if ( (unsigned int)sub_C444A0(v54 + 24) != v55 - 1 )
    {
      goto LABEL_24;
    }
    v56 = *(_BYTE **)(a3 + 16);
    if ( *v56 <= 0x15u && sub_AC30F0((__int64)v56) && (!*(_QWORD *)(a1 + 72) || *(_QWORD ***)(a1 + 336) == v25) )
      *(_QWORD *)(a1 + 72) = a2;
  }
LABEL_24:
  v36 = sub_D9B120(*(_QWORD *)(a1 + 16));
  result = (_QWORD *)(*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v36 + 8LL))(v36);
  if ( !(_BYTE)result )
    return result;
  v43 = a4;
  if ( !*(_BYTE *)(a4 + 28) )
    goto LABEL_92;
  v44 = *(__int64 **)(a4 + 8);
  v43 = *(unsigned int *)(a4 + 20);
  v38 = &v44[v43];
  if ( v44 == v38 )
  {
LABEL_95:
    if ( (unsigned int)v43 < *(_DWORD *)(a4 + 16) )
    {
      *(_DWORD *)(a4 + 20) = v43 + 1;
      *v38 = a2;
      ++*(_QWORD *)a4;
      goto LABEL_37;
    }
LABEL_92:
    sub_C8CC70(a4, a2, (__int64)v38, v43, v39, v40);
    goto LABEL_37;
  }
  while ( *v44 != a2 )
  {
    if ( v38 == ++v44 )
      goto LABEL_95;
  }
LABEL_37:
  v45 = sub_D47930(*(_QWORD *)a1);
  v48 = *(_QWORD *)(a2 - 8);
  v49 = v45;
  v50 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  if ( (*(_DWORD *)(a2 + 4) & 0x7FFFFFF) != 0 )
  {
    v51 = 0;
    while ( v49 != *(_QWORD *)(v48 + 32LL * *(unsigned int *)(a2 + 72) + 8 * v51) )
    {
      if ( (_DWORD)v50 == (_DWORD)++v51 )
        goto LABEL_91;
    }
    v52 = 32 * v51;
  }
  else
  {
LABEL_91:
    v52 = 0x1FFFFFFFE0LL;
  }
  v53 = *(_QWORD *)(v48 + v52);
  if ( !*(_BYTE *)(a4 + 28) )
    return sub_C8CC70(a4, v53, v50, v49, v46, v47);
  result = *(_QWORD **)(a4 + 8);
  v49 = *(unsigned int *)(a4 + 20);
  v50 = (__int64)&result[v49];
  if ( result == (_QWORD *)v50 )
  {
LABEL_46:
    result = (_QWORD *)a4;
    if ( (unsigned int)v49 < *(_DWORD *)(a4 + 16) )
    {
      *(_DWORD *)(a4 + 20) = v49 + 1;
      *(_QWORD *)v50 = v53;
      ++*(_QWORD *)a4;
      return result;
    }
    return sub_C8CC70(a4, v53, v50, v49, v46, v47);
  }
  while ( v53 != *result )
  {
    if ( (_QWORD *)v50 == ++result )
      goto LABEL_46;
  }
  return result;
}
