// Function: sub_18C7D70
// Address: 0x18c7d70
//
bool __fastcall sub_18C7D70(
        __int64 a1,
        __m128 a2,
        double a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        __m128 a9)
{
  bool result; // al
  __int64 v11; // r15
  __int64 v12; // r12
  _QWORD *v13; // rax
  _QWORD *v14; // rbx
  __int64 v15; // rdi
  unsigned int v16; // r13d
  __int64 v17; // rax
  _QWORD *v18; // rax
  unsigned int v19; // ebx
  size_t v20; // rbx
  unsigned int v21; // eax
  __int64 *v22; // r12
  __int64 v23; // r15
  _BYTE *v24; // r13
  size_t v25; // r8
  __int64 **v26; // rax
  __int64 v27; // rdx
  char v28; // al
  _QWORD *v29; // rcx
  _QWORD *v30; // r13
  unsigned int v31; // r11d
  unsigned __int64 v32; // rbx
  _BYTE *v33; // r13
  unsigned __int64 v34; // r10
  unsigned int v35; // r12d
  __int64 v36; // rax
  _QWORD *v37; // rsi
  unsigned __int64 v38; // rax
  __int64 v39; // rax
  __int64 *v40; // rax
  __int64 v41; // rax
  unsigned __int64 v42; // rcx
  unsigned __int64 v43; // rax
  char *v44; // rsi
  __int64 **v45; // rax
  __int64 **v46; // rdi
  __int64 v47; // r13
  __int64 v48; // r14
  _QWORD *v49; // rax
  __int64 v50; // rax
  _QWORD *v51; // r9
  __int64 v52; // r15
  _QWORD *v53; // rax
  __int64 v54; // r13
  __int64 *v55; // r8
  int v56; // r9d
  __int64 v57; // rax
  unsigned __int64 v58; // rcx
  __int64 v59; // rdx
  unsigned int v60; // ebx
  __int64 v61; // rdx
  __int64 *v62; // r8
  __int64 v63; // rax
  double v64; // xmm4_8
  double v65; // xmm5_8
  __int64 v66; // rax
  double v67; // xmm4_8
  double v68; // xmm5_8
  __int64 v69; // [rsp-158h] [rbp-158h]
  void *v70; // [rsp-148h] [rbp-148h]
  unsigned int *v71; // [rsp-140h] [rbp-140h]
  unsigned int v72; // [rsp-138h] [rbp-138h]
  __int64 v73; // [rsp-130h] [rbp-130h]
  __int64 v74; // [rsp-130h] [rbp-130h]
  __int64 v75; // [rsp-128h] [rbp-128h]
  __int64 v76; // [rsp-120h] [rbp-120h]
  __int64 v77; // [rsp-118h] [rbp-118h]
  __int64 *v78; // [rsp-110h] [rbp-110h]
  char v79; // [rsp-108h] [rbp-108h]
  unsigned __int64 v80; // [rsp-108h] [rbp-108h]
  __int64 *v81; // [rsp-108h] [rbp-108h]
  unsigned int v82; // [rsp-100h] [rbp-100h]
  __int64 v83; // [rsp-100h] [rbp-100h]
  _QWORD v84[2]; // [rsp-F8h] [rbp-F8h] BYREF
  __m128i v85; // [rsp-E8h] [rbp-E8h] BYREF
  __int64 v86; // [rsp-D8h] [rbp-D8h]
  _QWORD v87[2]; // [rsp-C8h] [rbp-C8h] BYREF
  __int16 v88; // [rsp-B8h] [rbp-B8h]
  __m128 v89; // [rsp-A8h] [rbp-A8h] BYREF
  _QWORD v90[2]; // [rsp-98h] [rbp-98h] BYREF
  _BYTE *v91; // [rsp-88h] [rbp-88h] BYREF
  __int64 v92; // [rsp-80h] [rbp-80h]
  _BYTE v93[16]; // [rsp-78h] [rbp-78h] BYREF
  __int64 **v94; // [rsp-68h] [rbp-68h] BYREF
  __int64 v95; // [rsp-60h] [rbp-60h]
  _QWORD v96[11]; // [rsp-58h] [rbp-58h] BYREF

  result = 0;
  if ( (*(_BYTE *)(a1 + 32) & 0xFu) - 7 > 1 )
    return result;
  v11 = *(_QWORD *)(a1 - 24);
  if ( !v11 || *(_BYTE *)(v11 + 16) != 7 )
    return result;
  v12 = *(_QWORD *)(a1 + 8);
  if ( !v12 )
  {
LABEL_18:
    v91 = v93;
    v92 = 0x200000000LL;
    sub_1626560(a1, 19, (__int64)&v91);
    v17 = sub_1632FA0(*(_QWORD *)(a1 + 40));
    v71 = (unsigned int *)sub_15A9930(v17, *(_QWORD *)v11);
    v18 = (_QWORD *)sub_16498A0(a1);
    v75 = sub_1643350(v18);
    v19 = *(_DWORD *)(v11 + 20) & 0xFFFFFFF;
    if ( v19 )
    {
      v20 = 8LL * v19;
      v70 = (void *)sub_22077B0(v20);
      memset(v70, 0, v20);
      v69 = v20;
      v21 = *(_DWORD *)(v11 + 20) & 0xFFFFFFF;
      if ( v21 )
      {
        v82 = 0;
        v76 = v11;
        while ( 1 )
        {
          v73 = *(_QWORD *)(a1 + 40);
          v22 = *(__int64 **)(v76 + 24 * (v82 - (unsigned __int64)v21));
          v23 = *v22;
          v79 = *(_BYTE *)(a1 + 80) & 1;
          if ( !v82 )
          {
            BYTE4(v90[0]) = 48;
            v24 = (char *)v90 + 4;
            v94 = (__int64 **)v96;
LABEL_23:
            v25 = 1;
            LOBYTE(v96[0]) = *v24;
            v26 = (__int64 **)v96;
            goto LABEL_24;
          }
          v42 = v82;
          v24 = (char *)v90 + 5;
          do
          {
            *--v24 = v42 % 0xA + 48;
            v43 = v42;
            v42 /= 0xAu;
          }
          while ( v43 > 9 );
          v44 = (char *)((char *)v90 + 5 - v24);
          v87[0] = (char *)v90 + 5 - v24;
          v25 = (char *)v90 + 5 - v24;
          v94 = (__int64 **)v96;
          if ( (unsigned __int64)((char *)v90 + 5 - v24) > 0xF )
            break;
          if ( v44 == (char *)1 )
            goto LABEL_23;
          if ( v44 )
          {
            v46 = (__int64 **)v96;
            goto LABEL_56;
          }
          v26 = (__int64 **)v96;
LABEL_24:
          v95 = v25;
          *((_BYTE *)v26 + v25) = 0;
          v88 = 260;
          v87[0] = &v94;
          v84[0] = sub_1649960(a1);
          v84[1] = v27;
          v85.m128i_i64[0] = (__int64)v84;
          v85.m128i_i64[1] = (__int64)".";
          v28 = v88;
          LOWORD(v86) = 773;
          if ( (_BYTE)v88 )
          {
            if ( (_BYTE)v88 == 1 )
            {
              a2 = (__m128)_mm_loadu_si128(&v85);
              v89 = a2;
              v90[0] = v86;
            }
            else
            {
              v29 = (_QWORD *)v87[0];
              if ( HIBYTE(v88) != 1 )
              {
                v29 = v87;
                v28 = 2;
              }
              v89.m128_u64[1] = (unsigned __int64)v29;
              v89.m128_u64[0] = (unsigned __int64)&v85;
              LOBYTE(v90[0]) = 2;
              BYTE1(v90[0]) = v28;
            }
          }
          else
          {
            LOWORD(v90[0]) = 256;
          }
          v30 = sub_1648A60(88, 1u);
          if ( v30 )
            sub_15E51E0((__int64)v30, v73, v23, v79, 8, (__int64)v22, (__int64)&v89, 0, 0, 0, 0);
          if ( v94 != v96 )
            j_j___libc_free_0(v94, v96[0] + 1LL);
          *((_QWORD *)v70 + v82) = v30;
          v21 = *(_DWORD *)(v76 + 20) & 0xFFFFFFF;
          v72 = v82 + 1;
          if ( v21 - 1 == v82 )
            v31 = *v71;
          else
            v31 = v71[2 * v72 + 4];
          v32 = (unsigned __int64)v91;
          if ( v91 != &v91[8 * (unsigned int)v92] )
          {
            v74 = (__int64)v30;
            v33 = &v91[8 * (unsigned int)v92];
            v34 = (unsigned int)*(_QWORD *)&v71[2 * v82 + 4];
            v35 = v31;
            do
            {
              v36 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)v32 - 8LL * *(unsigned int *)(*(_QWORD *)v32 + 8LL)) + 136LL);
              v37 = *(_QWORD **)(v36 + 24);
              if ( *(_DWORD *)(v36 + 32) > 0x40u )
                v37 = (_QWORD *)*v37;
              v38 = (unsigned __int64)v37 - 1;
              v83 = *(_QWORD *)v32;
              if ( !v37 )
                v38 = 0;
              if ( v38 >= v34 && v35 > v38 )
              {
                v80 = v34;
                v39 = sub_159C470(v75, (__int64)v37 - v34, 0);
                v94 = (__int64 **)sub_1624210(v39);
                v95 = *(_QWORD *)(v83 + 8 * (1LL - *(unsigned int *)(v83 + 8)));
                v40 = (__int64 *)sub_16498A0(a1);
                v41 = sub_1627350(v40, (__int64 *)&v94, (__int64 *)2, 0, 1);
                sub_16267C0(v74, 0x13u, v41);
                v34 = v80;
              }
              v32 += 8LL;
            }
            while ( v33 != (_BYTE *)v32 );
            v21 = *(_DWORD *)(v76 + 20) & 0xFFFFFFF;
          }
          if ( v21 == v72 )
            goto LABEL_62;
          v82 = v72;
        }
        v45 = (__int64 **)sub_22409D0(&v94, v87, 0);
        v25 = (char *)v90 + 5 - v24;
        v94 = v45;
        v46 = v45;
        v96[0] = v87[0];
LABEL_56:
        memcpy(v46, v24, v25);
        v25 = v87[0];
        v26 = v94;
        goto LABEL_24;
      }
LABEL_62:
      v47 = *(_QWORD *)(a1 + 8);
      if ( !v47 )
      {
        sub_15E55B0(a1);
LABEL_86:
        j_j___libc_free_0(v70, v69);
LABEL_87:
        if ( v91 != v93 )
          _libc_free((unsigned __int64)v91);
        return 1;
      }
    }
    else
    {
      v69 = 0;
      v47 = *(_QWORD *)(a1 + 8);
      v70 = 0;
      if ( !v47 )
      {
        sub_15E55B0(a1);
        goto LABEL_87;
      }
    }
    v77 = a1;
    v48 = v47;
    do
    {
      v53 = sub_1648700(v48);
      v54 = (__int64)v53;
      if ( (*((_BYTE *)v53 + 23) & 0x40) != 0 )
        v49 = (_QWORD *)*(v53 - 1);
      else
        v49 = &v53[-3 * (*((_DWORD *)v53 + 5) & 0xFFFFFFF)];
      v50 = v49[6];
      v51 = *(_QWORD **)(v50 + 24);
      if ( *(_DWORD *)(v50 + 32) > 0x40u )
        v51 = (_QWORD *)*v51;
      v52 = (unsigned int)v51;
      if ( v69 >> 3 > (unsigned __int64)(unsigned int)v51 )
      {
        v94 = (__int64 **)v96;
        v95 = 0x400000000LL;
        v55 = (__int64 *)sub_159C470(v75, 0, 0);
        v57 = (unsigned int)v95;
        if ( (unsigned int)v95 >= HIDWORD(v95) )
        {
          v81 = v55;
          sub_16CD150((__int64)&v94, v96, 0, 8, (int)v55, v56);
          v57 = (unsigned int)v95;
          v55 = v81;
        }
        v94[v57] = v55;
        v58 = (unsigned int)(v95 + 1);
        LODWORD(v95) = v95 + 1;
        v59 = *(_DWORD *)(v54 + 20) & 0xFFFFFFF;
        if ( (_DWORD)v59 != 3 )
        {
          v60 = 3;
          do
          {
            if ( (*(_BYTE *)(v54 + 23) & 0x40) != 0 )
              v61 = *(_QWORD *)(v54 - 8);
            else
              v61 = v54 - 24 * v59;
            v62 = *(__int64 **)(v61 + 24LL * v60);
            if ( (unsigned int)v58 >= HIDWORD(v95) )
            {
              v78 = *(__int64 **)(v61 + 24LL * v60);
              sub_16CD150((__int64)&v94, v96, 0, 8, (int)v62, v56);
              v58 = (unsigned int)v95;
              v62 = v78;
            }
            ++v60;
            v94[v58] = v62;
            v58 = (unsigned int)(v95 + 1);
            LODWORD(v95) = v95 + 1;
            v59 = *(_DWORD *)(v54 + 20) & 0xFFFFFFF;
          }
          while ( v60 != (_DWORD)v59 );
        }
        v89.m128_i8[4] = 0;
        v63 = sub_15A2E80(
                **(_QWORD **)(*((_QWORD *)v70 + v52) - 24LL),
                *((_QWORD *)v70 + v52),
                v94,
                v58,
                (*(_BYTE *)(v54 + 17) & 2) != 0,
                (__int64)&v89,
                0);
        sub_164D160(v54, v63, a2, a3, a4, a5, v64, v65, a8, a9);
        if ( v94 != v96 )
          _libc_free((unsigned __int64)v94);
      }
      v48 = *(_QWORD *)(v48 + 8);
    }
    while ( v48 );
    if ( *(_QWORD *)(v77 + 8) )
    {
      v66 = sub_1599EF0(*(__int64 ***)v77);
      sub_164D160(v77, v66, a2, a3, a4, a5, v67, v68, a8, a9);
    }
    sub_15E55B0(v77);
    if ( !v70 )
      goto LABEL_87;
    goto LABEL_86;
  }
  while ( 1 )
  {
    v13 = sub_1648700(v12);
    if ( *((_BYTE *)v13 + 16) != 5 || *((_WORD *)v13 + 9) != 32 || *((_BYTE *)v13 + 17) >> 1 >> 1 != 2 )
      return 0;
    v14 = (*((_BYTE *)v13 + 23) & 0x40) != 0 ? (_QWORD *)*(v13 - 1) : &v13[-3 * (*((_DWORD *)v13 + 5) & 0xFFFFFFF)];
    v15 = v14[3];
    if ( *(_BYTE *)(v15 + 16) != 13 )
      return 0;
    v16 = *(_DWORD *)(v15 + 32);
    if ( v16 <= 0x40 )
      result = *(_QWORD *)(v15 + 24) == 0;
    else
      result = v16 == (unsigned int)sub_16A57B0(v15 + 24);
    if ( !result )
      return result;
    if ( *(_BYTE *)(v14[6] + 16LL) != 13 )
      return 0;
    v12 = *(_QWORD *)(v12 + 8);
    if ( !v12 )
      goto LABEL_18;
  }
}
