// Function: sub_EBB490
// Address: 0xebb490
//
__int64 __fastcall sub_EBB490(__int64 a1, __int64 a2, __int64 a3)
{
  int *v4; // rax
  int v5; // edx
  const char *v6; // rbx
  __int64 v7; // rax
  unsigned __int64 v8; // rdx
  _DWORD *v9; // rax
  char v10; // al
  const char *v11; // rcx
  __int64 v12; // rax
  unsigned __int64 v13; // rcx
  _QWORD *v14; // r13
  __int64 *v15; // rsi
  __int64 *v16; // rax
  __int64 *v17; // r8
  __int64 *v18; // rbx
  __int64 v19; // rax
  unsigned __int64 v21; // rcx
  _DWORD *v22; // rbx
  int v23; // edx
  unsigned __int64 v24; // r14
  __m128i v25; // xmm0
  bool v26; // cc
  __int64 v27; // rdi
  __int64 v28; // rax
  __int64 v29; // rdx
  int *v30; // rax
  __int64 v31; // rdi
  __int64 v32; // rcx
  __int64 v33; // r8
  __int64 v34; // r9
  void **v35; // r13
  unsigned __int64 v36; // rcx
  _DWORD *v37; // rbx
  int v38; // edx
  unsigned __int64 v39; // r14
  __m128i v40; // xmm1
  __int64 v41; // rdi
  __int64 v42; // rax
  __int64 v43; // rdx
  int *v44; // rax
  __int64 v45; // rdi
  void **v46; // r12
  __int64 v47; // rcx
  __int64 v48; // r8
  __int64 v49; // r9
  const char *v50; // rax
  void **v51; // r13
  void **v52; // r13
  const char *v53; // [rsp+20h] [rbp-110h]
  __int64 *v54; // [rsp+38h] [rbp-F8h]
  char v56; // [rsp+4Eh] [rbp-E2h]
  unsigned __int8 v57; // [rsp+4Fh] [rbp-E1h]
  __int64 v58; // [rsp+50h] [rbp-E0h] BYREF
  __int64 v59; // [rsp+58h] [rbp-D8h] BYREF
  __int64 v60; // [rsp+60h] [rbp-D0h] BYREF
  __int64 v61; // [rsp+68h] [rbp-C8h] BYREF
  unsigned __int64 v62; // [rsp+70h] [rbp-C0h]
  unsigned __int64 v63; // [rsp+78h] [rbp-B8h] BYREF
  __int64 v64; // [rsp+80h] [rbp-B0h] BYREF
  __int64 v65; // [rsp+88h] [rbp-A8h] BYREF
  unsigned __int64 v66; // [rsp+90h] [rbp-A0h] BYREF
  unsigned __int64 v67; // [rsp+98h] [rbp-98h]
  __int64 v68; // [rsp+A0h] [rbp-90h] BYREF
  unsigned int v69; // [rsp+A8h] [rbp-88h]
  const char *v70; // [rsp+B0h] [rbp-80h] BYREF
  void **v71; // [rsp+B8h] [rbp-78h]
  const char *v72; // [rsp+D0h] [rbp-60h] BYREF
  void **v73; // [rsp+D8h] [rbp-58h]
  __int64 v74; // [rsp+E8h] [rbp-48h]
  unsigned int v75; // [rsp+F0h] [rbp-40h]

  v4 = *(int **)(a1 + 48);
  v5 = *v4;
  if ( *v4 == 13 )
  {
    v21 = *(unsigned int *)(a1 + 56);
    *(_BYTE *)(a1 + 155) = 0;
    v22 = v4 + 10;
    v23 = v21;
    v21 *= 40LL;
    v24 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)(v21 - 40) >> 3);
    if ( v21 > 0x28 )
    {
      do
      {
        v25 = _mm_loadu_si128((const __m128i *)(v22 + 2));
        v26 = *(v22 - 2) <= 0x40u;
        *(v22 - 10) = *v22;
        *((__m128i *)v22 - 2) = v25;
        if ( !v26 )
        {
          v27 = *((_QWORD *)v22 - 2);
          if ( v27 )
            j_j___libc_free_0_0(v27);
        }
        v28 = *((_QWORD *)v22 + 3);
        v22 += 10;
        *((_QWORD *)v22 - 7) = v28;
        LODWORD(v28) = *(v22 - 2);
        *(v22 - 2) = 0;
        *(v22 - 12) = v28;
        --v24;
      }
      while ( v24 );
      v23 = *(_DWORD *)(a1 + 56);
      v4 = *(int **)(a1 + 48);
    }
    v29 = (unsigned int)(v23 - 1);
    *(_DWORD *)(a1 + 56) = v29;
    v30 = &v4[10 * v29];
    if ( (unsigned int)v30[8] > 0x40 )
    {
      v31 = *((_QWORD *)v30 + 3);
      if ( v31 )
        j_j___libc_free_0_0(v31);
    }
    if ( *(_DWORD *)(a1 + 56) )
    {
      v56 = 1;
      v5 = **(_DWORD **)(a1 + 48);
    }
    else
    {
      sub_1097F60(&v72, a1 + 40);
      sub_EAA0A0(a1 + 48, *(_QWORD *)(a1 + 48), (unsigned __int64)&v72, v32, v33, v34);
      if ( v75 > 0x40 && v74 )
        j_j___libc_free_0_0(v74);
      v56 = 1;
      v5 = **(_DWORD **)(a1 + 48);
    }
  }
  else
  {
    v56 = 0;
    if ( v5 == 12 )
    {
      v36 = *(unsigned int *)(a1 + 56);
      *(_BYTE *)(a1 + 155) = 0;
      v37 = v4 + 10;
      v38 = v36;
      v36 *= 40LL;
      v39 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)(v36 - 40) >> 3);
      if ( v36 > 0x28 )
      {
        do
        {
          v40 = _mm_loadu_si128((const __m128i *)(v37 + 2));
          v26 = *(v37 - 2) <= 0x40u;
          *(v37 - 10) = *v37;
          *((__m128i *)v37 - 2) = v40;
          if ( !v26 )
          {
            v41 = *((_QWORD *)v37 - 2);
            if ( v41 )
              j_j___libc_free_0_0(v41);
          }
          v42 = *((_QWORD *)v37 + 3);
          v37 += 10;
          *((_QWORD *)v37 - 7) = v42;
          LODWORD(v42) = *(v37 - 2);
          *(v37 - 2) = 0;
          *(v37 - 12) = v42;
          --v39;
        }
        while ( v39 );
        v38 = *(_DWORD *)(a1 + 56);
        v4 = *(int **)(a1 + 48);
      }
      v43 = (unsigned int)(v38 - 1);
      *(_DWORD *)(a1 + 56) = v43;
      v44 = &v4[10 * v43];
      if ( (unsigned int)v44[8] > 0x40 )
      {
        v45 = *((_QWORD *)v44 + 3);
        if ( v45 )
          j_j___libc_free_0_0(v45);
      }
      if ( *(_DWORD *)(a1 + 56) )
      {
        v56 = 0;
        v5 = **(_DWORD **)(a1 + 48);
      }
      else
      {
        sub_1097F60(&v72, a1 + 40);
        sub_EAA0A0(a1 + 48, *(_QWORD *)(a1 + 48), (unsigned __int64)&v72, v47, v48, v49);
        if ( v75 > 0x40 && v74 )
          j_j___libc_free_0_0(v74);
        v56 = 0;
        v5 = **(_DWORD **)(a1 + 48);
      }
    }
  }
  if ( v5 == 1 )
  {
    LOWORD(v75) = 260;
    v72 = (const char *)(a1 + 112);
    return (unsigned __int8)sub_ECE0E0(a1, &v72, 0, 0);
  }
  v57 = v5 != 4 && (v5 & 0xFFFFFFFB) != 2;
  if ( v57 )
  {
    v72 = "unexpected token in directive";
    LOWORD(v75) = 259;
    return (unsigned __int8)sub_ECE0E0(a1, &v72, 0, 0);
  }
  v6 = (const char *)sub_C33340();
  if ( (const char *)a2 == v6 )
    sub_C3C460(&v70, a2);
  else
    sub_C37380(&v70, a2);
  v7 = sub_ECD7B0(a1);
  v8 = *(_QWORD *)(v7 + 16);
  v66 = *(_QWORD *)(v7 + 8);
  v9 = *(_DWORD **)(a1 + 48);
  v67 = v8;
  if ( *v9 != 2 )
  {
    sub_C43000((__int64)&v72, (void **)&v70, v66, v67, 1u);
    v10 = (char)v73;
    LOBYTE(v73) = (unsigned __int8)v73 & 0xFD;
    if ( (v10 & 1) != 0
      && (v11 = v72,
          v72 = 0,
          v12 = (unsigned __int64)v11 | 1,
          v13 = (unsigned __int64)v11 & 0xFFFFFFFFFFFFFFFELL,
          v58 = v12,
          (v14 = (_QWORD *)v13) != 0) )
    {
      v58 = 0;
      v15 = (__int64 *)&unk_4F84052;
      v59 = 0;
      v60 = 0;
      v61 = 0;
      if ( (*(unsigned __int8 (__fastcall **)(unsigned __int64, void *))(*(_QWORD *)v13 + 48LL))(v13, &unk_4F84052) )
      {
        v16 = (__int64 *)v14[2];
        v17 = (__int64 *)v14[1];
        v62 = 1;
        v54 = v16;
        if ( v17 == v16 )
        {
          v19 = 1;
        }
        else
        {
          v53 = v6;
          v18 = v17;
          do
          {
            v65 = *v18;
            *v18 = 0;
            sub_EA3500(&v64, &v65);
            v15 = &v68;
            v68 = v62 | 1;
            sub_9CDB40(&v63, (unsigned __int64 *)&v68, (unsigned __int64 *)&v64);
            v62 = v63 | 1;
            if ( (v68 & 1) != 0 || (v68 & 0xFFFFFFFFFFFFFFFELL) != 0 )
              sub_C63C30(&v68, (__int64)&v68);
            if ( (v64 & 1) != 0 || (v64 & 0xFFFFFFFFFFFFFFFELL) != 0 )
              sub_C63C30(&v64, (__int64)&v68);
            if ( v65 )
              (*(void (__fastcall **)(__int64))(*(_QWORD *)v65 + 8LL))(v65);
            ++v18;
          }
          while ( v54 != v18 );
          v6 = v53;
          v19 = v62 | 1;
        }
        v65 = v19;
        (*(void (__fastcall **)(_QWORD *))(*v14 + 8LL))(v14);
      }
      else
      {
        v15 = &v68;
        v68 = (__int64)v14;
        sub_EA3500(&v65, &v68);
        if ( v68 )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v68 + 8LL))(v68);
      }
      if ( (v65 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        BUG();
      if ( (v61 & 1) != 0 || (v61 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        sub_C63C30(&v61, (__int64)v15);
      if ( (v60 & 1) != 0 || (v60 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        sub_C63C30(&v60, (__int64)v15);
      if ( (v59 & 1) != 0 || (v59 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        sub_C63C30(&v59, (__int64)v15);
      if ( (v58 & 1) != 0 || (v58 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        sub_C63C30(&v58, (__int64)v15);
    }
    else
    {
      v14 = 0;
    }
    if ( ((unsigned __int8)v73 & 2) != 0 )
      sub_C432A0(&v72);
    if ( ((unsigned __int8)v73 & 1) != 0 && v72 )
      (*(void (__fastcall **)(const char *))(*(_QWORD *)v72 + 8LL))(v72);
    if ( v14 )
      goto LABEL_85;
LABEL_50:
    if ( v56 )
    {
      if ( v6 == v70 )
        sub_C3CCB0((__int64)&v70);
      else
        sub_C34440((unsigned __int8 *)&v70);
    }
    sub_EABFE0(a1);
    if ( v6 == v70 )
      sub_C3E660((__int64)&v72, (__int64)&v70);
    else
      sub_C3A850((__int64)&v72, (__int64 *)&v70);
    if ( *(_DWORD *)(a3 + 8) > 0x40u && *(_QWORD *)a3 )
      j_j___libc_free_0_0(*(_QWORD *)a3);
    *(_QWORD *)a3 = v72;
    *(_DWORD *)(a3 + 8) = (_DWORD)v73;
    goto LABEL_57;
  }
  if ( !(unsigned int)sub_C92E90(&v66, (__int64)"infinity", 8u) || !(unsigned int)sub_C92E90(&v66, (__int64)"inf", 3u) )
  {
    if ( (const char *)a2 == v6 )
      sub_C3C500(&v72, (__int64)v6);
    else
      sub_C373C0(&v72, a2);
    if ( v6 == v72 )
      sub_C3CF20((__int64)&v72, 0);
    else
      sub_C36EF0((_DWORD **)&v72, 0);
    if ( v6 == v70 )
    {
      if ( v72 == v6 )
      {
        sub_969EE0((__int64)&v70);
        sub_C3C840(&v70, &v72);
        goto LABEL_48;
      }
      if ( !v71 )
        goto LABEL_114;
      v51 = &v71[3 * (_QWORD)*(v71 - 1)];
      while ( v71 != v51 )
      {
        v51 -= 3;
        if ( v6 == *v51 )
          sub_969EE0((__int64)v51);
        else
          sub_C338F0((__int64)v51);
      }
      j_j_j___libc_free_0_0(v51 - 1);
      v50 = v72;
    }
    else
    {
      if ( v72 != v6 )
      {
        sub_C33870((__int64)&v70, (__int64)&v72);
        goto LABEL_48;
      }
      sub_C338F0((__int64)&v70);
      v50 = v72;
    }
    if ( v6 != v50 )
    {
LABEL_114:
      sub_C338E0((__int64)&v70, (__int64)&v72);
      goto LABEL_48;
    }
    sub_C3C840(&v70, &v72);
LABEL_48:
    if ( v6 == v72 )
    {
      if ( !v73 )
        goto LABEL_50;
      v35 = &v73[3 * (_QWORD)*(v73 - 1)];
      while ( v73 != v35 )
      {
        v35 -= 3;
        if ( v6 == *v35 )
          sub_969EE0((__int64)v35);
        else
          sub_C338F0((__int64)v35);
      }
      goto LABEL_138;
    }
LABEL_49:
    sub_C338F0((__int64)&v72);
    goto LABEL_50;
  }
  if ( !(unsigned int)sub_C92E90(&v66, (__int64)"nan", 3u) )
  {
    v69 = 64;
    v68 = -1;
    if ( (const char *)a2 == v6 )
      sub_C3C500(&v72, (__int64)v6);
    else
      sub_C373C0(&v72, a2);
    if ( v6 == v72 )
      sub_C3D480((__int64)&v72, 0, 0, (unsigned __int64 *)&v68);
    else
      sub_C36070((__int64)&v72, 0, 0, (unsigned __int64 *)&v68);
    if ( v69 > 0x40 && v68 )
      j_j___libc_free_0_0(v68);
    if ( v6 == v70 )
    {
      if ( v72 == v6 )
      {
        sub_969EE0((__int64)&v70);
        sub_C3C840(&v70, &v72);
        goto LABEL_73;
      }
      if ( v71 )
      {
        v52 = &v71[3 * (_QWORD)*(v71 - 1)];
        while ( v71 != v52 )
        {
          v52 -= 3;
          if ( v6 == *v52 )
            sub_969EE0((__int64)v52);
          else
            sub_C338F0((__int64)v52);
        }
        j_j_j___libc_free_0_0(v52 - 1);
      }
    }
    else
    {
      if ( v72 != v6 )
      {
        sub_C33870((__int64)&v70, (__int64)&v72);
        goto LABEL_73;
      }
      sub_C338F0((__int64)&v70);
    }
    if ( v6 == v72 )
      sub_C3C840(&v70, &v72);
    else
      sub_C338E0((__int64)&v70, (__int64)&v72);
LABEL_73:
    if ( v6 == v72 )
    {
      if ( !v73 )
        goto LABEL_50;
      v35 = &v73[3 * (_QWORD)*(v73 - 1)];
      while ( v73 != v35 )
      {
        v35 -= 3;
        if ( v6 == *v35 )
          sub_969EE0((__int64)v35);
        else
          sub_C338F0((__int64)v35);
      }
LABEL_138:
      j_j_j___libc_free_0_0(v35 - 1);
      goto LABEL_50;
    }
    goto LABEL_49;
  }
LABEL_85:
  v72 = "invalid floating point literal";
  LOWORD(v75) = 259;
  v57 = sub_ECE0E0(a1, &v72, 0, 0);
LABEL_57:
  if ( v6 == v70 )
  {
    if ( v71 )
    {
      v46 = &v71[3 * (_QWORD)*(v71 - 1)];
      while ( v71 != v46 )
      {
        v46 -= 3;
        if ( v6 == *v46 )
          sub_969EE0((__int64)v46);
        else
          sub_C338F0((__int64)v46);
      }
      j_j_j___libc_free_0_0(v46 - 1);
    }
  }
  else
  {
    sub_C338F0((__int64)&v70);
  }
  return v57;
}
