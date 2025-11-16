// Function: sub_1CF19E0
// Address: 0x1cf19e0
//
void __fastcall sub_1CF19E0(
        __int64 a1,
        __m128 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        __m128 a9)
{
  __int64 *v9; // rdi
  _BYTE *v10; // rdx
  _BYTE *v11; // rax
  unsigned __int64 v12; // rbx
  __int64 v13; // rax
  __int64 *v14; // rbx
  __int64 *v15; // rax
  char v16; // cl
  unsigned __int64 v17; // rcx
  __int64 v18; // rax
  _BYTE *v19; // rdx
  __int64 v20; // r13
  __int64 v21; // r15
  __int64 v22; // rsi
  __int64 v23; // r12
  unsigned __int64 v24; // rax
  unsigned __int64 v25; // rdi
  int v26; // eax
  unsigned int v27; // esi
  __int64 v28; // rdi
  __int64 v29; // r15
  __int64 *v30; // rax
  char v31; // dl
  __int64 v32; // rdx
  __int64 *v33; // rax
  char v34; // si
  __int64 v35; // rbx
  __int64 v36; // r14
  _BYTE *v37; // r15
  _QWORD *v38; // rax
  __int64 v39; // rdx
  __int64 v40; // r13
  _BYTE *v41; // rsi
  __int64 v42; // r15
  __int64 v43; // rax
  double v44; // xmm4_8
  double v45; // xmm5_8
  _QWORD *v46; // r13
  unsigned __int64 *v47; // rcx
  unsigned __int64 v48; // rdx
  double v49; // xmm4_8
  double v50; // xmm5_8
  __int64 *v51; // rsi
  unsigned int v52; // r9d
  __int64 *v53; // r8
  unsigned __int64 v54; // rax
  __int64 v55; // r15
  unsigned int v56; // r12d
  int v57; // r13d
  unsigned int v58; // esi
  __int64 v59; // rax
  __int64 v60; // rdi
  __int64 v61; // rax
  __int64 v62; // rbx
  __int64 v63; // r12
  __int64 v65; // [rsp+10h] [rbp-140h]
  __int64 v66; // [rsp+28h] [rbp-128h]
  __int64 v67; // [rsp+30h] [rbp-120h]
  __int64 *v68; // [rsp+38h] [rbp-118h] BYREF
  __int64 *v69; // [rsp+40h] [rbp-110h]
  char *v70; // [rsp+48h] [rbp-108h]
  __int64 v71[3]; // [rsp+50h] [rbp-100h] BYREF
  char v72; // [rsp+68h] [rbp-E8h]
  __int64 v73; // [rsp+70h] [rbp-E0h] BYREF
  _BYTE *v74; // [rsp+78h] [rbp-D8h]
  _BYTE *v75; // [rsp+80h] [rbp-D0h]
  __int64 v76; // [rsp+88h] [rbp-C8h]
  _BYTE *v77; // [rsp+98h] [rbp-B8h]
  _BYTE *v78; // [rsp+A0h] [rbp-B0h]
  __int64 v79; // [rsp+A8h] [rbp-A8h]
  __int64 v80; // [rsp+B0h] [rbp-A0h] BYREF
  _BYTE *v81; // [rsp+B8h] [rbp-98h]
  _BYTE *v82; // [rsp+C0h] [rbp-90h]
  __int64 v83; // [rsp+C8h] [rbp-88h]
  int v84; // [rsp+D0h] [rbp-80h]
  _BYTE v85[120]; // [rsp+D8h] [rbp-78h] BYREF

  v81 = v85;
  v82 = v85;
  v71[0] = a1;
  v9 = &v73;
  v80 = 0;
  v83 = 8;
  v84 = 0;
  sub_1CF17E0(&v73, v71, (__int64)&v80);
  v10 = v74;
  v68 = 0;
  v69 = 0;
  v67 = v73;
  v11 = v75;
  v70 = 0;
  v12 = v75 - v74;
  if ( v75 == v74 )
  {
    v9 = 0;
  }
  else
  {
    if ( v12 > 0x7FFFFFFFFFFFFFE0LL )
      goto LABEL_109;
    v13 = sub_22077B0(v75 - v74);
    v10 = v74;
    v9 = (__int64 *)v13;
    v11 = v75;
  }
  v68 = v9;
  v69 = v9;
  v70 = (char *)v9 + v12;
  if ( v11 == v10 )
  {
    v14 = v9;
  }
  else
  {
    v14 = (__int64 *)((char *)v9 + v11 - v10);
    v15 = v9;
    do
    {
      if ( v15 )
      {
        *v15 = *(_QWORD *)v10;
        v16 = v10[24];
        *((_BYTE *)v15 + 24) = v16;
        if ( v16 )
        {
          a2 = (__m128)_mm_loadu_si128((const __m128i *)(v10 + 8));
          *(__m128 *)(v15 + 1) = a2;
        }
      }
      v15 += 4;
      v10 += 32;
    }
    while ( v15 != v14 );
  }
  v10 = v77;
  v69 = v14;
  v17 = v78 - v77;
  v65 = v78 - v77;
  if ( v78 == v77 )
  {
    v20 = 0;
    goto LABEL_87;
  }
  if ( v17 > 0x7FFFFFFFFFFFFFE0LL )
LABEL_109:
    sub_4261EA(v9, v71, v10);
  v18 = sub_22077B0(v78 - v77);
  v19 = v77;
  v9 = v68;
  v20 = v18;
  v14 = v69;
  if ( v78 == v77 )
  {
LABEL_87:
    v66 = 0;
    goto LABEL_19;
  }
  v21 = v78 - v77;
  v22 = v18 + v78 - v77;
  do
  {
    if ( v18 )
    {
      *(_QWORD *)v18 = *(_QWORD *)v19;
      v17 = (unsigned __int8)v19[24];
      *(_BYTE *)(v18 + 24) = v17;
      if ( (_BYTE)v17 )
      {
        a3 = (__m128)_mm_loadu_si128((const __m128i *)(v19 + 8));
        *(__m128 *)(v18 + 8) = a3;
      }
    }
    v18 += 32;
    v19 += 32;
  }
  while ( v18 != v22 );
  v66 = v21;
LABEL_19:
  while ( (char *)v14 - (char *)v9 != v66 )
  {
LABEL_20:
    while ( 2 )
    {
      v23 = *(v14 - 4);
      if ( !*((_BYTE *)v14 - 8) )
      {
        v24 = sub_157EBA0(*(v14 - 4));
        *((_BYTE *)v14 - 8) = 1;
        *(v14 - 3) = v24;
        *((_DWORD *)v14 - 4) = 0;
      }
      while ( 1 )
      {
        v25 = sub_157EBA0(v23);
        v26 = 0;
        if ( v25 )
          v26 = sub_15F4D60(v25);
        v27 = *((_DWORD *)v14 - 4);
        if ( v27 == v26 )
          break;
        v28 = *(v14 - 3);
        *((_DWORD *)v14 - 4) = v27 + 1;
        v29 = sub_15F4DF0(v28, v27);
        v30 = *(__int64 **)(v67 + 8);
        if ( *(__int64 **)(v67 + 16) != v30 )
          goto LABEL_26;
        v51 = &v30[*(unsigned int *)(v67 + 28)];
        v52 = *(_DWORD *)(v67 + 28);
        if ( v30 == v51 )
        {
LABEL_79:
          if ( v52 < *(_DWORD *)(v67 + 24) )
          {
            *(_DWORD *)(v67 + 28) = v52 + 1;
            *v51 = v29;
            ++*(_QWORD *)v67;
LABEL_27:
            v71[0] = v29;
            v72 = 0;
            sub_144A690((__int64 *)&v68, (__int64)v71);
            v14 = v69;
            v9 = v68;
            if ( (char *)v69 - (char *)v68 != v66 )
              goto LABEL_20;
            goto LABEL_28;
          }
LABEL_26:
          sub_16CCBA0(v67, v29);
          if ( v31 )
            goto LABEL_27;
        }
        else
        {
          v53 = 0;
          while ( v29 != *v30 )
          {
            if ( *v30 == -2 )
            {
              v53 = v30;
              if ( v51 == v30 + 1 )
                goto LABEL_76;
              ++v30;
            }
            else if ( v51 == ++v30 )
            {
              if ( !v53 )
                goto LABEL_79;
LABEL_76:
              *v53 = v29;
              --*(_DWORD *)(v67 + 32);
              ++*(_QWORD *)v67;
              goto LABEL_27;
            }
          }
        }
      }
      v69 -= 4;
      v9 = v68;
      v14 = v69;
      if ( v69 != v68 )
        continue;
      break;
    }
  }
LABEL_28:
  if ( v14 != v9 )
  {
    v32 = v20;
    v33 = v9;
    while ( *v33 == *(_QWORD *)v32 )
    {
      v17 = *((unsigned __int8 *)v33 + 24);
      v34 = *(_BYTE *)(v32 + 24);
      if ( (_BYTE)v17 && v34 )
      {
        v17 = *(unsigned int *)(v32 + 16);
        if ( *((_DWORD *)v33 + 4) != (_DWORD)v17 )
          goto LABEL_20;
        v33 += 4;
        v32 += 32;
        if ( v33 == v14 )
          goto LABEL_35;
      }
      else
      {
        if ( v34 != (_BYTE)v17 )
          goto LABEL_20;
        v33 += 4;
        v32 += 32;
        if ( v33 == v14 )
          goto LABEL_35;
      }
    }
    goto LABEL_20;
  }
LABEL_35:
  if ( v20 )
  {
    j_j___libc_free_0(v20, v65);
    v9 = v68;
  }
  if ( v9 )
    j_j___libc_free_0(v9, v70 - (char *)v9);
  if ( v77 )
    j_j___libc_free_0(v77, v79 - (_QWORD)v77);
  if ( v74 )
    j_j___libc_free_0(v74, v76 - (_QWORD)v74);
  v73 = 0;
  v74 = 0;
  v35 = *(_QWORD *)(a1 + 80);
  v36 = a1 + 72;
  v75 = 0;
  if ( a1 + 72 != v35 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v40 = v35 - 24;
        if ( !v35 )
          v40 = 0;
        v38 = v81;
        if ( v82 == v81 )
          break;
        v37 = &v82[8 * (unsigned int)v83];
        v38 = sub_16CC9F0((__int64)&v80, v40);
        if ( v40 == *v38 )
        {
          if ( v82 == v81 )
          {
            v17 = HIDWORD(v83);
            v39 = (__int64)&v82[8 * HIDWORD(v83)];
          }
          else
          {
            v17 = (unsigned int)v83;
            v39 = (__int64)&v82[8 * (unsigned int)v83];
          }
          goto LABEL_60;
        }
        if ( v82 == v81 )
        {
          v38 = &v82[8 * HIDWORD(v83)];
          v39 = (__int64)v38;
          goto LABEL_60;
        }
        v39 = (unsigned int)v83;
        v38 = &v82[8 * (unsigned int)v83];
LABEL_48:
        if ( v37 == (_BYTE *)v38 )
          goto LABEL_62;
LABEL_49:
        v35 = *(_QWORD *)(v35 + 8);
        if ( v36 == v35 )
          goto LABEL_97;
      }
      v37 = &v81[8 * HIDWORD(v83)];
      if ( v81 == v37 )
      {
        v39 = (__int64)v81;
      }
      else
      {
        do
        {
          if ( v40 == *v38 )
            break;
          ++v38;
        }
        while ( v37 != (_BYTE *)v38 );
        v39 = (__int64)&v81[8 * HIDWORD(v83)];
      }
LABEL_60:
      while ( (_QWORD *)v39 != v38 )
      {
        if ( *v38 < 0xFFFFFFFFFFFFFFFELL )
          goto LABEL_48;
        ++v38;
      }
      if ( v37 != (_BYTE *)v38 )
        goto LABEL_49;
LABEL_62:
      v71[0] = v40;
      v41 = v74;
      if ( v74 == v75 )
      {
        sub_1292090((__int64)&v73, v74, v71);
        v40 = v71[0];
      }
      else
      {
        if ( v74 )
        {
          *(_QWORD *)v74 = v40;
          v41 = v74;
        }
        v41 += 8;
        v74 = v41;
      }
      v42 = *(_QWORD *)(v40 + 48);
      if ( !v42 )
LABEL_69:
        BUG();
      while ( *(_BYTE *)(v42 - 8) == 77 )
      {
        v43 = sub_15A06D0(*(__int64 ***)(v42 - 24), (__int64)v41, v39, v17);
        sub_164D160(v42 - 24, v43, a2, *(double *)a3.m128_u64, a4, a5, v44, v45, a8, a9);
        v46 = *(_QWORD **)(v71[0] + 48);
        v41 = v46 - 3;
        sub_157EA20(v71[0] + 40, (__int64)(v46 - 3));
        v47 = (unsigned __int64 *)v46[1];
        v48 = *v46 & 0xFFFFFFFFFFFFFFF8LL;
        *v47 = v48 | *v47 & 7;
        *(_QWORD *)(v48 + 8) = v47;
        *v46 &= 7uLL;
        v46[1] = 0;
        sub_164BEC0(
          (__int64)(v46 - 3),
          (__int64)(v46 - 3),
          v48,
          (__int64)v47,
          a2,
          *(double *)a3.m128_u64,
          a4,
          a5,
          v49,
          v50,
          a8,
          a9);
        v40 = v71[0];
        v42 = *(_QWORD *)(v71[0] + 48);
        if ( !v42 )
          goto LABEL_69;
      }
      v54 = sub_157EBA0(v40);
      v55 = v54;
      if ( v54 )
      {
        v56 = 0;
        v57 = sub_15F4D60(v54);
        if ( v57 )
        {
          do
          {
            v58 = v56++;
            v59 = sub_15F4DF0(v55, v58);
            sub_157F2D0(v59, v71[0], 0);
          }
          while ( v56 != v57 );
        }
        v40 = v71[0];
      }
      sub_157EE90(v40);
      v35 = *(_QWORD *)(v35 + 8);
      if ( v36 == v35 )
      {
LABEL_97:
        v60 = v73;
        v61 = (__int64)&v74[-v73] >> 3;
        if ( (_DWORD)v61 )
        {
          v62 = 0;
          v63 = 8LL * (unsigned int)(v61 - 1);
          while ( 1 )
          {
            sub_157F980(*(_QWORD *)(v60 + v62));
            v60 = v73;
            if ( v62 == v63 )
              break;
            v62 += 8;
          }
        }
        if ( v60 )
          j_j___libc_free_0(v60, &v75[-v60]);
        break;
      }
    }
  }
  if ( v82 != v81 )
    _libc_free((unsigned __int64)v82);
}
