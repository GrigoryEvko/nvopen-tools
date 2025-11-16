// Function: sub_EBA960
// Address: 0xeba960
//
__int64 __fastcall sub_EBA960(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // r15
  int *v5; // rdx
  int *v6; // rsi
  int v7; // eax
  __int64 v8; // rax
  _DWORD *v9; // r13
  int v10; // edx
  unsigned __int64 v11; // r14
  __m128i v12; // xmm0
  bool v13; // cc
  __int64 v14; // rdi
  __int64 v15; // rdx
  __int64 v16; // rdx
  int *v17; // rax
  __int64 v18; // rdi
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  int *v22; // rax
  __int64 v23; // rax
  __int64 result; // rax
  __int64 v25; // rcx
  __int64 v26; // rax
  __int64 v27; // rdi
  __int64 v28; // rcx
  __int64 v29; // r9
  __int64 v30; // r14
  __int64 v31; // r15
  __int64 v32; // rdx
  int v33; // ecx
  unsigned __int64 v34; // rbx
  __m128i v35; // xmm1
  __int64 v36; // rdi
  __int64 v37; // rcx
  __int64 v38; // rax
  __int64 v39; // rdi
  __int64 (*v40)(); // rax
  __int64 v41; // r14
  __int64 v42; // rbx
  __int64 v43; // rdx
  int v44; // ecx
  unsigned __int64 v45; // r15
  __m128i v46; // xmm2
  __int64 v47; // rdi
  __int64 v48; // rcx
  __int64 v49; // rcx
  __int64 v50; // rax
  __int64 v51; // rdi
  __int64 v52; // rcx
  __int64 v53; // r9
  __m128i v54; // xmm5
  void (*v55)(void); // rax
  __int64 v56; // rdi
  __int64 v57; // r8
  __int64 v58; // rax
  __int64 v59; // [rsp+8h] [rbp-F8h]
  __int64 v60; // [rsp+10h] [rbp-F0h]
  __int64 v61; // [rsp+10h] [rbp-F0h]
  char v62; // [rsp+1Fh] [rbp-E1h]
  int v64; // [rsp+28h] [rbp-D8h]
  unsigned __int64 v65; // [rsp+28h] [rbp-D8h]
  unsigned __int8 v66; // [rsp+28h] [rbp-D8h]
  unsigned __int8 v67; // [rsp+28h] [rbp-D8h]
  __int64 v68; // [rsp+38h] [rbp-C8h] BYREF
  const char *v69; // [rsp+40h] [rbp-C0h] BYREF
  __int64 v70; // [rsp+48h] [rbp-B8h]
  const char *v71; // [rsp+50h] [rbp-B0h] BYREF
  __int64 v72; // [rsp+58h] [rbp-A8h]
  __int64 v73; // [rsp+60h] [rbp-A0h] BYREF
  int v74; // [rsp+70h] [rbp-90h] BYREF
  __m128i v75; // [rsp+78h] [rbp-88h] BYREF
  const void *v76; // [rsp+88h] [rbp-78h] BYREF
  unsigned int v77; // [rsp+90h] [rbp-70h]
  const char *v78; // [rsp+A0h] [rbp-60h] BYREF
  __m128i v79; // [rsp+A8h] [rbp-58h]
  const void *v80; // [rsp+B8h] [rbp-48h] BYREF
  unsigned int v81; // [rsp+C0h] [rbp-40h]

  v3 = sub_ECD7B0(a1);
  v4 = *(_QWORD *)(a1 + 920);
  v5 = (int *)v3;
  v6 = *(int **)(v4 + 8);
  v7 = *v6;
  if ( *v6 == 9 )
  {
    if ( *(_QWORD *)(sub_ECD7B0(a1) + 16)
      && **(_BYTE **)(sub_ECD7B0(a1) + 8) != 13
      && **(_BYTE **)(sub_ECD7B0(a1) + 8) != 10 )
    {
LABEL_69:
      sub_EABFE0(a1);
      return 0;
    }
LABEL_67:
    v55 = *(void (**)(void))(**(_QWORD **)(a1 + 928) + 160LL);
    if ( v55 != nullsub_99 )
      v55();
    goto LABEL_69;
  }
  v64 = *v5;
  if ( v7 == 11 )
  {
    do
    {
      v8 = *(unsigned int *)(v4 + 16);
      *(_BYTE *)(v4 + 115) = 0;
      v9 = v6 + 10;
      v10 = v8;
      v11 = 0xCCCCCCCCCCCCCCCDLL * ((40 * v8 - 40) >> 3);
      if ( (unsigned __int64)(40 * v8) > 0x28 )
      {
        do
        {
          v12 = _mm_loadu_si128((const __m128i *)(v9 + 2));
          v13 = *(v9 - 2) <= 0x40u;
          *(v9 - 10) = *v9;
          *((__m128i *)v9 - 2) = v12;
          if ( !v13 )
          {
            v14 = *((_QWORD *)v9 - 2);
            if ( v14 )
              j_j___libc_free_0_0(v14);
          }
          v15 = *((_QWORD *)v9 + 3);
          v9 += 10;
          *((_QWORD *)v9 - 7) = v15;
          LODWORD(v15) = *(v9 - 2);
          *(v9 - 2) = 0;
          *(v9 - 12) = v15;
          --v11;
        }
        while ( v11 );
        v10 = *(_DWORD *)(v4 + 16);
        v6 = *(int **)(v4 + 8);
      }
      v16 = (unsigned int)(v10 - 1);
      *(_DWORD *)(v4 + 16) = v16;
      v17 = &v6[10 * v16];
      if ( (unsigned int)v17[8] > 0x40 )
      {
        v18 = *((_QWORD *)v17 + 3);
        if ( v18 )
          j_j___libc_free_0_0(v18);
      }
      if ( !*(_DWORD *)(v4 + 16) )
      {
        (**(void (__fastcall ***)(const char **, __int64))v4)(&v78, v4);
        sub_EAA0A0(v4 + 8, *(_QWORD *)(v4 + 8), (unsigned __int64)&v78, v19, v20, v21);
        if ( v81 > 0x40 )
        {
          if ( v80 )
            j_j___libc_free_0_0(v80);
        }
      }
      v4 = *(_QWORD *)(a1 + 920);
      v6 = *(int **)(v4 + 8);
      v7 = *v6;
    }
    while ( *v6 == 11 );
  }
  if ( v7 == 9 && (**(_BYTE **)(sub_ECD7B0(a1) + 8) == 10 || **(_BYTE **)(sub_ECD7B0(a1) + 8) == 13) )
    goto LABEL_67;
  if ( v64 == 11 )
    goto LABEL_19;
  v38 = sub_ECD7B0(a1);
  v74 = *(_DWORD *)v38;
  v75 = _mm_loadu_si128((const __m128i *)(v38 + 8));
  v77 = *(_DWORD *)(v38 + 32);
  if ( v77 > 0x40 )
    sub_C43780((__int64)&v76, (const void **)(v38 + 24));
  else
    v76 = *(const void **)(v38 + 24);
  v69 = 0;
  v68 = sub_ECD6A0(&v74);
  v70 = 0;
  v62 = sub_EB61F0(a1, (__int64 *)&v69);
  if ( v62 )
  {
    v78 = "The HLASM Label has to be an Identifier";
    LOWORD(v81) = 259;
    goto LABEL_90;
  }
  v39 = *(_QWORD *)(a1 + 8);
  v40 = *(__int64 (**)())(*(_QWORD *)v39 + 144LL);
  if ( (v40 == sub_EA21C0 || ((unsigned __int8 (__fastcall *)(__int64, int *))v40)(v39, &v74))
    && (*(_BYTE *)(a1 + 869) || !(unsigned __int8)sub_EA2540(a1)) )
  {
    while ( 1 )
    {
      v41 = *(_QWORD *)(a1 + 920);
      v42 = *(_QWORD *)(v41 + 8);
      if ( *(_DWORD *)v42 != 11 )
        break;
      v43 = *(unsigned int *)(v41 + 16);
      *(_BYTE *)(v41 + 115) = 0;
      v44 = v43;
      v45 = 0xCCCCCCCCCCCCCCCDLL * ((40 * v43 - 40) >> 3);
      if ( (unsigned __int64)(40 * v43) > 0x28 )
      {
        do
        {
          v46 = _mm_loadu_si128((const __m128i *)(v42 + 48));
          v13 = *(_DWORD *)(v42 + 32) <= 0x40u;
          *(_DWORD *)v42 = *(_DWORD *)(v42 + 40);
          *(__m128i *)(v42 + 8) = v46;
          if ( !v13 )
          {
            v47 = *(_QWORD *)(v42 + 24);
            if ( v47 )
              j_j___libc_free_0_0(v47);
          }
          v48 = *(_QWORD *)(v42 + 64);
          v42 += 40;
          *(_QWORD *)(v42 - 16) = v48;
          LODWORD(v48) = *(_DWORD *)(v42 + 32);
          *(_DWORD *)(v42 + 32) = 0;
          *(_DWORD *)(v42 - 8) = v48;
          --v45;
        }
        while ( v45 );
        v44 = *(_DWORD *)(v41 + 16);
        v42 = *(_QWORD *)(v41 + 8);
      }
      v49 = (unsigned int)(v44 - 1);
      *(_DWORD *)(v41 + 16) = v49;
      v50 = v42 + 40 * v49;
      if ( *(_DWORD *)(v50 + 32) > 0x40u )
      {
        v51 = *(_QWORD *)(v50 + 24);
        if ( v51 )
          j_j___libc_free_0_0(v51);
      }
      if ( !*(_DWORD *)(v41 + 16) )
      {
        (**(void (__fastcall ***)(const char **, __int64))v41)(&v78, v41);
        sub_EAA0A0(v41 + 8, *(_QWORD *)(v41 + 8), (unsigned __int64)&v78, v52, *(_QWORD *)(v41 + 8), v53);
        if ( v81 > 0x40 )
        {
          if ( v80 )
            j_j___libc_free_0_0(v80);
        }
      }
    }
    if ( *(_DWORD *)sub_ECD7B0(a1) != 9 )
    {
      if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(a1 + 224) + 152LL) + 22LL) )
      {
        v59 = *(_QWORD *)(a1 + 224);
        sub_C93170((__int64 *)&v71, (__int64)&v69);
        LOWORD(v81) = 261;
        v78 = v71;
        v79.m128i_i64[0] = v72;
        v58 = sub_E6C460(v59, &v78);
        v57 = v58;
        if ( v71 != (const char *)&v73 )
        {
          v61 = v58;
          j_j___libc_free_0(v71, v73 + 1);
          v57 = v61;
        }
      }
      else
      {
        v56 = *(_QWORD *)(a1 + 224);
        LOWORD(v81) = 261;
        v78 = v69;
        v79.m128i_i64[0] = v70;
        v57 = sub_E6C460(v56, &v78);
      }
      v60 = v57;
      (*(void (__fastcall **)(_QWORD, __int64, __int64))(**(_QWORD **)(a1 + 928) + 208LL))(
        *(_QWORD *)(a1 + 928),
        v57,
        v68);
      if ( (unsigned __int8)sub_EAA750(a1) )
        sub_E787B0(v60, *(_QWORD **)(a1 + 232), *(__int64 **)(a1 + 248), (unsigned __int64 *)&v68);
      if ( v77 <= 0x40 )
        goto LABEL_19;
      goto LABEL_82;
    }
    v78 = "Cannot have just a label for an HLASM inline asm statement";
    LOWORD(v81) = 259;
LABEL_90:
    v62 = sub_ECDA70(a1, v68, &v78, 0, 0);
    if ( v77 <= 0x40 )
      goto LABEL_84;
    goto LABEL_82;
  }
  if ( v77 <= 0x40 )
  {
LABEL_85:
    sub_EB4E00(a1);
    return 1;
  }
  v62 = 1;
LABEL_82:
  if ( v76 )
    j_j___libc_free_0_0(v76);
LABEL_84:
  if ( v62 )
    goto LABEL_85;
LABEL_19:
  v22 = *(int **)(*(_QWORD *)(a1 + 920) + 8LL);
  v74 = *v22;
  v75 = _mm_loadu_si128((const __m128i *)(v22 + 2));
  v77 = v22[8];
  if ( v77 > 0x40 )
    sub_C43780((__int64)&v76, (const void **)v22 + 3);
  else
    v76 = (const void *)*((_QWORD *)v22 + 3);
  v23 = sub_ECD6A0(&v74);
  v71 = 0;
  v65 = v23;
  v72 = 0;
  if ( (unsigned __int8)sub_EB61F0(a1, (__int64 *)&v71) )
  {
    v78 = "unexpected token at start of statement";
    LOWORD(v81) = 259;
    result = sub_ECDA70(a1, v65, &v78, 0, 0);
  }
  else
  {
    while ( 1 )
    {
      v30 = *(_QWORD *)(a1 + 920);
      v31 = *(_QWORD *)(v30 + 8);
      if ( *(_DWORD *)v31 != 11 )
        break;
      v32 = *(unsigned int *)(v30 + 16);
      *(_BYTE *)(v30 + 115) = 0;
      v33 = v32;
      v34 = 0xCCCCCCCCCCCCCCCDLL * ((40 * v32 - 40) >> 3);
      if ( (unsigned __int64)(40 * v32) > 0x28 )
      {
        do
        {
          v35 = _mm_loadu_si128((const __m128i *)(v31 + 48));
          v13 = *(_DWORD *)(v31 + 32) <= 0x40u;
          *(_DWORD *)v31 = *(_DWORD *)(v31 + 40);
          *(__m128i *)(v31 + 8) = v35;
          if ( !v13 )
          {
            v36 = *(_QWORD *)(v31 + 24);
            if ( v36 )
              j_j___libc_free_0_0(v36);
          }
          v37 = *(_QWORD *)(v31 + 64);
          v31 += 40;
          *(_QWORD *)(v31 - 16) = v37;
          LODWORD(v37) = *(_DWORD *)(v31 + 32);
          *(_DWORD *)(v31 + 32) = 0;
          *(_DWORD *)(v31 - 8) = v37;
          --v34;
        }
        while ( v34 );
        v33 = *(_DWORD *)(v30 + 16);
        v31 = *(_QWORD *)(v30 + 8);
      }
      v25 = (unsigned int)(v33 - 1);
      *(_DWORD *)(v30 + 16) = v25;
      v26 = v31 + 40 * v25;
      if ( *(_DWORD *)(v26 + 32) > 0x40u )
      {
        v27 = *(_QWORD *)(v26 + 24);
        if ( v27 )
          j_j___libc_free_0_0(v27);
      }
      if ( !*(_DWORD *)(v30 + 16) )
      {
        (**(void (__fastcall ***)(const char **, __int64))v30)(&v78, v30);
        sub_EAA0A0(v30 + 8, *(_QWORD *)(v30 + 8), (unsigned __int64)&v78, v28, *(_QWORD *)(v30 + 8), v29);
        if ( v81 > 0x40 )
        {
          if ( v80 )
            j_j___libc_free_0_0(v80);
        }
      }
    }
    v54 = _mm_loadu_si128(&v75);
    LODWORD(v78) = v74;
    v79 = v54;
    v81 = v77;
    if ( v77 > 0x40 )
      sub_C43780((__int64)&v80, &v76);
    else
      v80 = v76;
    result = sub_EAA8B0(a1, a2, (__int64)v71, v72, (int *)&v78, v65);
    if ( v81 > 0x40 && v80 )
    {
      v67 = result;
      j_j___libc_free_0_0(v80);
      result = v67;
    }
  }
  if ( v77 > 0x40 )
  {
    if ( v76 )
    {
      v66 = result;
      j_j___libc_free_0_0(v76);
      return v66;
    }
  }
  return result;
}
