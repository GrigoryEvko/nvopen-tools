// Function: sub_902D10
// Address: 0x902d10
//
__int64 __fastcall sub_902D10(
        const char *a1,
        int a2,
        __int64 *a3,
        const char *a4,
        const char *a5,
        int a6,
        unsigned __int8 a7,
        unsigned __int8 a8,
        unsigned __int8 a9)
{
  __int64 v11; // rax
  char *v12; // r12
  __int64 v13; // r8
  unsigned int v14; // r15d
  size_t v15; // rdx
  size_t v16; // rax
  unsigned __int64 v17; // rsi
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rcx
  unsigned int *v21; // r8
  __int64 v22; // r9
  int v23; // r10d
  int v24; // r13d
  unsigned int v25; // r13d
  char *v26; // rdi
  __int64 v27; // rdi
  __int64 v28; // r8
  __int64 v29; // rbx
  __int64 v30; // rdi
  __int64 v31; // r8
  __int64 v32; // rbx
  __int64 v33; // rdi
  int v35; // eax
  size_t v36; // r15
  _QWORD *v37; // rax
  __int64 v38; // rax
  __m128i *v39; // rdx
  __m128i si128; // xmm0
  __int64 v41; // rax
  size_t v42; // rax
  size_t v43; // r14
  _QWORD *v44; // rdx
  char *v45; // rdi
  _QWORD *v46; // rdi
  unsigned int v48; // [rsp+20h] [rbp-2A0h]
  _QWORD *src; // [rsp+78h] [rbp-248h]
  char v52; // [rsp+8Ah] [rbp-236h] BYREF
  unsigned __int8 v53; // [rsp+8Bh] [rbp-235h] BYREF
  unsigned int v54; // [rsp+8Ch] [rbp-234h] BYREF
  unsigned int v55; // [rsp+90h] [rbp-230h] BYREF
  unsigned int v56; // [rsp+94h] [rbp-22Ch] BYREF
  __int64 v57; // [rsp+98h] [rbp-228h] BYREF
  _QWORD *v58; // [rsp+A0h] [rbp-220h] BYREF
  __int64 v59; // [rsp+A8h] [rbp-218h]
  _QWORD v60[2]; // [rsp+B0h] [rbp-210h] BYREF
  _QWORD v61[2]; // [rsp+C0h] [rbp-200h] BYREF
  _QWORD v62[2]; // [rsp+D0h] [rbp-1F0h] BYREF
  _BYTE *v63; // [rsp+E0h] [rbp-1E0h] BYREF
  __int64 v64; // [rsp+E8h] [rbp-1D8h]
  _QWORD v65[2]; // [rsp+F0h] [rbp-1D0h] BYREF
  unsigned __int64 v66; // [rsp+100h] [rbp-1C0h] BYREF
  size_t n; // [rsp+108h] [rbp-1B8h]
  _QWORD v68[2]; // [rsp+110h] [rbp-1B0h] BYREF
  char *v69; // [rsp+120h] [rbp-1A0h] BYREF
  size_t v70; // [rsp+128h] [rbp-198h]
  __int64 v71; // [rsp+130h] [rbp-190h]
  char v72[8]; // [rsp+138h] [rbp-188h] BYREF
  __int16 v73; // [rsp+140h] [rbp-180h]
  char v74[8]; // [rsp+1C0h] [rbp-100h] BYREF
  _QWORD v75[2]; // [rsp+1C8h] [rbp-F8h] BYREF
  _QWORD v76[2]; // [rsp+1D8h] [rbp-E8h] BYREF
  _QWORD v77[2]; // [rsp+1E8h] [rbp-D8h] BYREF
  _QWORD v78[2]; // [rsp+1F8h] [rbp-C8h] BYREF
  __int16 v79; // [rsp+208h] [rbp-B8h]
  char v80; // [rsp+20Ah] [rbp-B6h]
  int v81; // [rsp+20Ch] [rbp-B4h]
  __int64 v82; // [rsp+210h] [rbp-B0h]
  _QWORD v83[2]; // [rsp+228h] [rbp-98h] BYREF
  _QWORD v84[2]; // [rsp+238h] [rbp-88h] BYREF
  _QWORD *v85; // [rsp+248h] [rbp-78h]
  __int64 v86; // [rsp+250h] [rbp-70h]
  _QWORD v87[2]; // [rsp+258h] [rbp-68h] BYREF
  __int16 v88; // [rsp+268h] [rbp-58h]
  char v89; // [rsp+26Ah] [rbp-56h]
  int v90; // [rsp+26Ch] [rbp-54h]
  __int64 v91; // [rsp+270h] [rbp-50h]

  v11 = sub_22077B0(8);
  v12 = (char *)v11;
  if ( v11 )
    sub_B6EEA0(v11);
  v58 = v60;
  v75[0] = v76;
  v77[0] = v78;
  v83[0] = v84;
  v79 = 0;
  v85 = v87;
  v88 = 0;
  v57 = 0;
  v52 = 0;
  v53 = 0;
  v54 = 0;
  v59 = 0;
  LOBYTE(v60[0]) = 0;
  v75[1] = 0;
  LOBYTE(v76[0]) = 0;
  v77[1] = 0;
  LOBYTE(v78[0]) = 0;
  v80 = 0;
  v81 = 0;
  v82 = 0;
  v83[1] = 0;
  LOBYTE(v84[0]) = 0;
  v86 = 0;
  LOBYTE(v87[0]) = 0;
  v89 = 0;
  v90 = 0;
  v91 = 0;
  v63 = v65;
  v61[0] = v62;
  v61[1] = 0;
  LOBYTE(v62[0]) = 0;
  v64 = 0;
  LOBYTE(v65[0]) = 0;
  v55 = 0;
  v56 = 0;
  if ( (unsigned int)sub_900130(
                       a1,
                       a4,
                       a5,
                       a3,
                       a8,
                       (__int64)v74,
                       (__int64)&v58,
                       &v52,
                       &v53,
                       (__int64)&v63,
                       (__int64)&v55,
                       (__int64)&v56) )
  {
    sub_223E0D0(&unk_4FD4BE0, "\n Error processing command line: ", 33);
    v25 = 1;
    v41 = sub_223E0D0(&unk_4FD4BE0, v58, v59);
    v17 = (unsigned __int64)"\n";
    sub_223E0D0(v41, "\n", 1);
    goto LABEL_12;
  }
  if ( v64 )
  {
    v13 = v56;
    v14 = v55;
    v15 = 0;
    if ( a4 )
    {
      v48 = v56;
      v16 = strlen(a4);
      v13 = v48;
      v15 = v16;
    }
    sub_C98ED0(0, a4, v15, v14, v13);
  }
  v17 = (unsigned __int64)v12;
  v54 = 0;
  v18 = sub_905880(a2, (_DWORD)v12, (unsigned int)v75, (unsigned int)&v57, (unsigned int)&v54, a7, 0, a9, 0);
  v21 = &v54;
  v22 = a7;
  v23 = a9;
  v24 = v18;
  if ( v18 || a9 == 1 )
  {
    if ( !v80 )
    {
      v73 = 260;
      v69 = (char *)v77;
      sub_C823F0(&v69, 1);
      LODWORD(v22) = a7;
      v23 = a9;
    }
    v54 = 0;
    sub_905EE0(
      (_DWORD)a1,
      v24,
      (unsigned int)v83,
      a6,
      v53,
      (unsigned int)v61,
      (__int64)&v57,
      (__int64)&v54,
      v22,
      a8,
      0,
      v23);
    v17 = v54;
    if ( v54 && !a9 )
    {
      v25 = 1;
      goto LABEL_12;
    }
    if ( !((__int64 (*)(void))sub_C96F30)() )
    {
LABEL_67:
      v25 = 0;
      goto LABEL_12;
    }
    v35 = sub_2241AC0(&v63, "-");
    v66 = (unsigned __int64)v68;
    if ( v35 )
    {
      sub_8FC5C0((__int64 *)&v66, v63, (__int64)&v63[v64]);
      goto LABEL_56;
    }
    if ( !a5 )
      sub_426248((__int64)"basic_string::_M_construct null not valid");
    v42 = strlen(a5);
    v69 = (char *)v42;
    v43 = v42;
    if ( v42 > 0xF )
    {
      v66 = sub_22409D0(&v66, &v69, 0);
      v46 = (_QWORD *)v66;
      v68[0] = v69;
    }
    else
    {
      if ( v42 == 1 )
      {
        LOBYTE(v68[0]) = *a5;
        v44 = v68;
LABEL_73:
        n = v42;
        *((_BYTE *)v44 + v42) = 0;
LABEL_56:
        v36 = n;
        v70 = 0;
        v69 = v72;
        v37 = (_QWORD *)v66;
        v71 = 128;
        if ( n > 0x80 )
        {
          src = (_QWORD *)v66;
          sub_C8D290(&v69, v72, n, 1);
          v37 = src;
          v45 = &v69[v70];
        }
        else
        {
          if ( !n )
          {
LABEL_58:
            v70 = v36;
            if ( v37 != v68 )
            {
              j_j___libc_free_0(v37, v68[0] + 1LL);
              v36 = v70;
            }
            v17 = (unsigned __int64)v69;
            sub_C9C600(&v66, v69, v36, "-", 1);
            if ( (v66 & 0xFFFFFFFFFFFFFFFELL) != 0 )
            {
              v66 = v66 & 0xFFFFFFFFFFFFFFFELL | 1;
              v38 = sub_CB72A0();
              v39 = *(__m128i **)(v38 + 32);
              if ( *(_QWORD *)(v38 + 24) - (_QWORD)v39 <= 0x2Au )
              {
                v17 = (unsigned __int64)"Error: Failed to write time profiler data.\n";
                sub_CB6200(v38, "Error: Failed to write time profiler data.\n", 43);
              }
              else
              {
                si128 = _mm_load_si128((const __m128i *)&xmmword_3E9F970);
                qmemcpy(&v39[2], "iler data.\n", 11);
                *v39 = si128;
                v39[1] = _mm_load_si128((const __m128i *)&xmmword_3E9F980);
                *(_QWORD *)(v38 + 32) += 43LL;
              }
            }
            else
            {
              v66 = 0;
            }
            sub_C99310();
            if ( (v66 & 1) != 0 || (v66 & 0xFFFFFFFFFFFFFFFELL) != 0 )
              sub_C63C30(&v66);
            if ( v69 != v72 )
              _libc_free(v69, v17);
            goto LABEL_67;
          }
          v45 = v72;
        }
        memcpy(v45, v37, v36);
        v37 = (_QWORD *)v66;
        v36 += v70;
        goto LABEL_58;
      }
      if ( !v42 )
      {
        v44 = v68;
        goto LABEL_73;
      }
      v46 = v68;
    }
    memcpy(v46, a5, v43);
    v42 = (size_t)v69;
    v44 = (_QWORD *)v66;
    goto LABEL_73;
  }
  v25 = 1;
  if ( v12 )
  {
    sub_B6E710(v12);
    v26 = v12;
    v17 = 8;
    v12 = 0;
    j_j___libc_free_0(v26, 8);
  }
LABEL_12:
  v27 = v57;
  if ( v57 )
  {
    j_j___libc_free_0_0(v57);
    v57 = 0;
  }
  if ( sub_C96F30(v27, v17, v19, v20, v21, v22) )
    sub_C99310();
  if ( v63 != (_BYTE *)v65 )
    j_j___libc_free_0(v63, v65[0] + 1LL);
  if ( (_QWORD *)v61[0] != v62 )
    j_j___libc_free_0(v61[0], v62[0] + 1LL);
  v28 = v91;
  if ( v91 )
  {
    if ( v90 <= 0 )
      goto LABEL_27;
    v29 = 0;
    do
    {
      v30 = *(_QWORD *)(v28 + 8 * v29);
      if ( v30 )
      {
        j_j___libc_free_0_0(v30);
        v28 = v91;
      }
      ++v29;
    }
    while ( v90 > (int)v29 );
    if ( v28 )
LABEL_27:
      j_j___libc_free_0_0(v28);
  }
  if ( v85 != v87 )
    j_j___libc_free_0(v85, v87[0] + 1LL);
  if ( (_QWORD *)v83[0] != v84 )
    j_j___libc_free_0(v83[0], v84[0] + 1LL);
  v31 = v82;
  if ( v82 )
  {
    if ( v81 <= 0 )
      goto LABEL_39;
    v32 = 0;
    do
    {
      v33 = *(_QWORD *)(v31 + 8 * v32);
      if ( v33 )
      {
        j_j___libc_free_0_0(v33);
        v31 = v82;
      }
      ++v32;
    }
    while ( v81 > (int)v32 );
    if ( v31 )
LABEL_39:
      j_j___libc_free_0_0(v31);
  }
  if ( (_QWORD *)v77[0] != v78 )
    j_j___libc_free_0(v77[0], v78[0] + 1LL);
  if ( (_QWORD *)v75[0] != v76 )
    j_j___libc_free_0(v75[0], v76[0] + 1LL);
  if ( v58 != v60 )
    j_j___libc_free_0(v58, v60[0] + 1LL);
  if ( v12 )
  {
    sub_B6E710(v12);
    j_j___libc_free_0(v12, 8);
  }
  return v25;
}
