// Function: sub_195CAE0
// Address: 0x195cae0
//
__int64 __fastcall sub_195CAE0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128 a7,
        double a8,
        double a9,
        double a10,
        double a11,
        double a12,
        double a13,
        __m128 a14,
        char a15,
        __int64 *a16,
        __int64 *a17)
{
  __int64 *v18; // r15
  __int64 v19; // r13
  unsigned __int64 v20; // rdi
  unsigned __int64 v21; // rdi
  __int64 v22; // r14
  char *v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rax
  bool v26; // dl
  bool v27; // zf
  __int64 v28; // rax
  __int64 v29; // r13
  unsigned __int64 v30; // rdi
  unsigned __int64 v31; // rdi
  __int64 v32; // rax
  __int64 *v33; // r13
  __int64 v34; // rdi
  __int64 v35; // r13
  __int64 v36; // rbx
  __int64 v37; // rdi
  unsigned int v38; // ecx
  __int64 *v39; // rdx
  __int64 v40; // r9
  __int64 v41; // rax
  __int64 v42; // rsi
  __int64 *v43; // rax
  double v44; // xmm4_8
  double v45; // xmm5_8
  __int64 v46; // rbx
  int v47; // r13d
  _QWORD *v48; // r15
  _QWORD *v49; // rax
  __int64 v50; // r14
  int v51; // eax
  __int64 v52; // rax
  __int64 v53; // r15
  unsigned __int64 v54; // rax
  unsigned __int64 v55; // r15
  int v56; // eax
  _QWORD *v57; // rax
  __int64 v58; // rax
  _QWORD *v59; // rcx
  void *v60; // rdi
  unsigned int v61; // eax
  __int64 v62; // rdx
  __int64 *v64; // rdi
  __int64 *v65; // rcx
  int v66; // edx
  int v67; // r10d
  _QWORD *v68; // r15
  _QWORD *v69; // rcx
  void *v70; // r9
  __int64 v71; // rdx
  __int64 v72; // rcx
  __int64 v73; // rsi
  _QWORD *v74; // rcx
  _QWORD *v75; // r14
  _QWORD *v76; // r15
  void *v77; // rcx
  __int64 v78; // rdx
  __int64 v79; // [rsp+0h] [rbp-130h]
  unsigned __int8 v80; // [rsp+8h] [rbp-128h]
  void *v81; // [rsp+8h] [rbp-128h]
  __int64 v83; // [rsp+18h] [rbp-118h]
  _QWORD *v84; // [rsp+18h] [rbp-118h]
  void *v85; // [rsp+18h] [rbp-118h]
  __int64 v86; // [rsp+28h] [rbp-108h] BYREF
  __int64 v87; // [rsp+30h] [rbp-100h]
  __int64 v88; // [rsp+38h] [rbp-F8h]
  __int64 v89; // [rsp+40h] [rbp-F0h]
  char *v90; // [rsp+50h] [rbp-E0h] BYREF
  __int64 v91; // [rsp+58h] [rbp-D8h] BYREF
  _BYTE *v92; // [rsp+60h] [rbp-D0h]
  __int64 v93; // [rsp+68h] [rbp-C8h]
  __int64 v94; // [rsp+70h] [rbp-C0h]
  _BYTE v95[184]; // [rsp+78h] [rbp-B8h] BYREF

  v18 = *(__int64 **)(a1 + 32);
  *(_QWORD *)a1 = a3;
  *(_QWORD *)(a1 + 8) = a4;
  *(_QWORD *)(a1 + 16) = a5;
  *(_QWORD *)(a1 + 24) = a6;
  *(_QWORD *)(a1 + 32) = 0;
  if ( v18 )
  {
    sub_1368A00(v18);
    j_j___libc_free_0(v18, 8);
  }
  v19 = *(_QWORD *)(a1 + 40);
  *(_QWORD *)(a1 + 40) = 0;
  if ( v19 )
  {
    v20 = *(_QWORD *)(v19 + 256);
    if ( v20 != *(_QWORD *)(v19 + 248) )
      _libc_free(v20);
    v21 = *(_QWORD *)(v19 + 88);
    if ( v21 != *(_QWORD *)(v19 + 80) )
      _libc_free(v21);
    j___libc_free_0(*(_QWORD *)(v19 + 40));
    if ( *(_DWORD *)(v19 + 24) )
    {
      v86 = 2;
      v87 = 0;
      v88 = -8;
      v89 = 0;
      v91 = 2;
      v92 = 0;
      v93 = -16;
      v90 = (char *)&unk_49E8A80;
      v94 = 0;
      v68 = *(_QWORD **)(v19 + 8);
      v69 = &v68[5 * *(unsigned int *)(v19 + 24)];
      if ( v68 != v69 )
      {
        v70 = &unk_49EE2B0;
        do
        {
          v71 = v68[3];
          *v68 = v70;
          if ( v71 != -8 && v71 != 0 && v71 != -16 )
          {
            v81 = v70;
            v84 = v69;
            sub_1649B30(v68 + 1);
            v70 = v81;
            v69 = v84;
          }
          v68 += 5;
        }
        while ( v69 != v68 );
        v90 = (char *)&unk_49EE2B0;
        if ( v93 != 0 && v93 != -16 && v93 != -8 )
          sub_1649B30(&v91);
      }
      if ( v88 != -8 && v88 != 0 && v88 != -16 )
        sub_1649B30(&v86);
    }
    j___libc_free_0(*(_QWORD *)(v19 + 8));
    j_j___libc_free_0(v19, 408);
  }
  *(_BYTE *)(a1 + 48) = a15;
  v22 = *(_QWORD *)(a2 + 40);
  v23 = sub_15E0FD0(79);
  v25 = sub_16321A0(v22, (__int64)v23, v24);
  v26 = 0;
  if ( v25 )
    v26 = *(_QWORD *)(v25 + 8) != 0;
  v27 = *(_BYTE *)(a1 + 48) == 0;
  *(_BYTE *)(a1 + 49) = v26;
  if ( !v27 )
  {
    v28 = *a17;
    *a17 = 0;
    v29 = *(_QWORD *)(a1 + 40);
    *(_QWORD *)(a1 + 40) = v28;
    if ( v29 )
    {
      v30 = *(_QWORD *)(v29 + 256);
      if ( v30 != *(_QWORD *)(v29 + 248) )
        _libc_free(v30);
      v31 = *(_QWORD *)(v29 + 88);
      if ( v31 != *(_QWORD *)(v29 + 80) )
        _libc_free(v31);
      j___libc_free_0(*(_QWORD *)(v29 + 40));
      if ( *(_DWORD *)(v29 + 24) )
      {
        v86 = 2;
        v87 = 0;
        v88 = -8;
        v89 = 0;
        v91 = 2;
        v92 = 0;
        v93 = -16;
        v90 = (char *)&unk_49E8A80;
        v94 = 0;
        v75 = *(_QWORD **)(v29 + 8);
        v76 = &v75[5 * *(unsigned int *)(v29 + 24)];
        if ( v75 != v76 )
        {
          v77 = &unk_49EE2B0;
          do
          {
            v78 = v75[3];
            *v75 = v77;
            if ( v78 != -8 && v78 != 0 && v78 != -16 )
            {
              v85 = v77;
              sub_1649B30(v75 + 1);
              v77 = v85;
            }
            v75 += 5;
          }
          while ( v76 != v75 );
        }
        v90 = (char *)&unk_49EE2B0;
        sub_1455FA0((__int64)&v91);
        sub_1455FA0((__int64)&v86);
      }
      j___libc_free_0(*(_QWORD *)(v29 + 8));
      j_j___libc_free_0(v29, 408);
    }
    v32 = *a16;
    *a16 = 0;
    v33 = *(__int64 **)(a1 + 32);
    *(_QWORD *)(a1 + 32) = v32;
    if ( v33 )
    {
      sub_1368A00(v33);
      j_j___libc_free_0(v33, 8);
    }
  }
  v34 = *(_QWORD *)(a1 + 24);
  v90 = 0;
  v91 = (__int64)v95;
  v92 = v95;
  v93 = 16;
  LODWORD(v94) = 0;
  v35 = sub_15DC150(v34);
  v36 = *(_QWORD *)(a2 + 80);
  v83 = a2 + 72;
  if ( v36 != a2 + 72 )
  {
    while ( 1 )
    {
      v41 = *(unsigned int *)(v35 + 48);
      v42 = v36 - 24;
      if ( !v36 )
        v42 = 0;
      if ( (_DWORD)v41 )
      {
        v37 = *(_QWORD *)(v35 + 32);
        v38 = (v41 - 1) & (((unsigned int)v42 >> 9) ^ ((unsigned int)v42 >> 4));
        v39 = (__int64 *)(v37 + 16LL * v38);
        v40 = *v39;
        if ( v42 == *v39 )
        {
LABEL_25:
          if ( v39 != (__int64 *)(v37 + 16 * v41) && v39[1] )
            goto LABEL_27;
        }
        else
        {
          v66 = 1;
          while ( v40 != -8 )
          {
            v67 = v66 + 1;
            v38 = (v41 - 1) & (v66 + v38);
            v39 = (__int64 *)(v37 + 16LL * v38);
            v40 = *v39;
            if ( v42 == *v39 )
              goto LABEL_25;
            v66 = v67;
          }
        }
      }
      v43 = (__int64 *)v91;
      if ( v92 == (_BYTE *)v91 )
      {
        v64 = (__int64 *)(v91 + 8LL * HIDWORD(v93));
        if ( (__int64 *)v91 == v64 )
        {
LABEL_128:
          if ( HIDWORD(v93) >= (unsigned int)v93 )
            goto LABEL_32;
          ++HIDWORD(v93);
          *v64 = v42;
          ++v90;
        }
        else
        {
          v65 = 0;
          while ( v42 != *v43 )
          {
            if ( *v43 == -2 )
              v65 = v43;
            if ( v64 == ++v43 )
            {
              if ( !v65 )
                goto LABEL_128;
              *v65 = v42;
              LODWORD(v94) = v94 - 1;
              ++v90;
              break;
            }
          }
        }
LABEL_27:
        v36 = *(_QWORD *)(v36 + 8);
        if ( v83 == v36 )
          break;
      }
      else
      {
LABEL_32:
        sub_16CCBA0((__int64)&v90, v42);
        v36 = *(_QWORD *)(v36 + 8);
        if ( v83 == v36 )
          break;
      }
    }
  }
  sub_1953080(a1, a2);
  v80 = 0;
  v79 = a1 + 56;
  while ( 1 )
  {
    v46 = *(_QWORD *)(a2 + 80);
    if ( v83 == v46 )
      break;
    v47 = 0;
    do
    {
      while ( 1 )
      {
        v50 = 0;
        v49 = (_QWORD *)v91;
        if ( v46 )
          v50 = v46 - 24;
        if ( v92 == (_BYTE *)v91 )
        {
          v48 = (_QWORD *)(v91 + 8LL * HIDWORD(v93));
          if ( (_QWORD *)v91 == v48 )
          {
            v59 = (_QWORD *)v91;
          }
          else
          {
            do
            {
              if ( v50 == *v49 )
                break;
              ++v49;
            }
            while ( v48 != v49 );
            v59 = (_QWORD *)(v91 + 8LL * HIDWORD(v93));
          }
LABEL_51:
          while ( v59 != v49 )
          {
            if ( *v49 < 0xFFFFFFFFFFFFFFFELL )
              goto LABEL_39;
            ++v49;
          }
          if ( v49 != v48 )
            goto LABEL_40;
        }
        else
        {
          v48 = &v92[8 * (unsigned int)v93];
          v49 = sub_16CC9F0((__int64)&v90, v50);
          if ( v50 == *v49 )
          {
            if ( v92 == (_BYTE *)v91 )
              v59 = &v92[8 * HIDWORD(v93)];
            else
              v59 = &v92[8 * (unsigned int)v93];
            goto LABEL_51;
          }
          if ( v92 == (_BYTE *)v91 )
          {
            v49 = &v92[8 * HIDWORD(v93)];
            v59 = v49;
            goto LABEL_51;
          }
          v49 = &v92[8 * (unsigned int)v93];
LABEL_39:
          if ( v49 != v48 )
            goto LABEL_40;
        }
        v51 = v47;
        do
        {
          v47 = v51;
          v51 = sub_195BF20(a1, v50, a7, a8, a9, a10, v44, v45, a13, a14);
        }
        while ( (_BYTE)v51 );
        v52 = *(_QWORD *)(a2 + 80);
        if ( v52 )
          v52 -= 24;
        if ( v50 != v52 && !sub_15CD6A0(*(_QWORD *)(a1 + 24), v50) )
          break;
LABEL_40:
        v46 = *(_QWORD *)(v46 + 8);
        if ( v83 == v46 )
          goto LABEL_75;
      }
      v53 = *(_QWORD *)(v50 + 8);
      if ( v53 )
      {
        while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v53) + 16) - 25) > 9u )
        {
          v53 = *(_QWORD *)(v53 + 8);
          if ( !v53 )
            goto LABEL_71;
        }
        v54 = sub_157EBA0(v50);
        v55 = v54;
        if ( *(_BYTE *)(v54 + 16) == 26
          && (*(_DWORD *)(v54 + 20) & 0xFFFFFFF) == 1
          && (unsigned int)*(unsigned __int8 *)(sub_157ED60(v50) + 16) - 25 <= 9
          && !sub_1377F70(v79, v50)
          && !sub_1377F70(v79, *(_QWORD *)(v55 - 24)) )
        {
          v56 = sub_1AF2B30(v50, *(_QWORD *)(a1 + 24));
          if ( (_BYTE)v56 )
          {
            v47 = v56;
            sub_13EB690(*(__int64 **)(a1 + 8), v50);
          }
        }
        goto LABEL_40;
      }
LABEL_71:
      v57 = *(_QWORD **)(a1 + 64);
      if ( *(_QWORD **)(a1 + 72) == v57 )
      {
        v74 = &v57[*(unsigned int *)(a1 + 84)];
        if ( v57 == v74 )
        {
LABEL_119:
          v57 = v74;
        }
        else
        {
          while ( v50 != *v57 )
          {
            if ( v74 == ++v57 )
              goto LABEL_119;
          }
        }
      }
      else
      {
        v57 = sub_16CC9F0(v79, v50);
        if ( v50 == *v57 )
        {
          v72 = *(_QWORD *)(a1 + 72);
          if ( v72 == *(_QWORD *)(a1 + 64) )
            v73 = *(unsigned int *)(a1 + 84);
          else
            v73 = *(unsigned int *)(a1 + 80);
          v74 = (_QWORD *)(v72 + 8 * v73);
        }
        else
        {
          v58 = *(_QWORD *)(a1 + 72);
          if ( v58 != *(_QWORD *)(a1 + 64) )
            goto LABEL_74;
          v57 = (_QWORD *)(v58 + 8LL * *(unsigned int *)(a1 + 84));
          v74 = v57;
        }
      }
      if ( v74 != v57 )
      {
        *v57 = -2;
        ++*(_DWORD *)(a1 + 88);
      }
LABEL_74:
      v47 = 1;
      sub_13EB690(*(__int64 **)(a1 + 8), v50);
      sub_1AA7270(v50, *(_QWORD *)(a1 + 24));
      v46 = *(_QWORD *)(v46 + 8);
    }
    while ( v83 != v46 );
LABEL_75:
    if ( !(_BYTE)v47 )
      break;
    v80 = v47;
  }
  ++*(_QWORD *)(a1 + 56);
  v60 = *(void **)(a1 + 72);
  if ( v60 == *(void **)(a1 + 64) )
    goto LABEL_83;
  v61 = 4 * (*(_DWORD *)(a1 + 84) - *(_DWORD *)(a1 + 88));
  v62 = *(unsigned int *)(a1 + 80);
  if ( v61 < 0x20 )
    v61 = 32;
  if ( v61 < (unsigned int)v62 )
  {
    sub_16CC920(v79);
  }
  else
  {
    memset(v60, -1, 8 * v62);
LABEL_83:
    *(_QWORD *)(a1 + 84) = 0;
  }
  sub_15DC150(*(_QWORD *)(a1 + 24));
  sub_13EBC50(*(__int64 **)(a1 + 8));
  if ( v92 != (_BYTE *)v91 )
    _libc_free((unsigned __int64)v92);
  return v80;
}
