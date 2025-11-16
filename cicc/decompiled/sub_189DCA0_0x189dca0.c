// Function: sub_189DCA0
// Address: 0x189dca0
//
__int64 __fastcall sub_189DCA0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __m128 a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        __m128 a12)
{
  _QWORD *v13; // rax
  __int64 v14; // r13
  unsigned __int64 *v15; // r14
  unsigned __int64 *v16; // r12
  _QWORD *v17; // rax
  _QWORD *v18; // rdx
  char v19; // cl
  __int64 v20; // rcx
  __int64 v21; // rsi
  int v22; // r8d
  int v23; // r9d
  double v24; // xmm4_8
  double v25; // xmm5_8
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rbx
  unsigned int v29; // eax
  _QWORD *v30; // r13
  __int64 v31; // rsi
  __int64 v32; // r13
  _QWORD *v33; // rax
  __int64 v34; // rdx
  __int64 v35; // r14
  __int64 v36; // rbx
  __int64 v37; // rcx
  _QWORD *v38; // r9
  __int64 *v39; // r15
  __int64 *v40; // r13
  __int64 v41; // rax
  int v42; // esi
  int v43; // r8d
  _QWORD *v44; // rbx
  __int64 v45; // rax
  int v46; // ecx
  __int64 v47; // rdx
  unsigned __int64 *v48; // rdi
  __int64 *v49; // rdx
  __int64 v50; // rbx
  __int64 v51; // rax
  unsigned int v52; // edx
  __int64 v53; // rcx
  int v54; // r10d
  _QWORD *v56; // rbx
  _QWORD *v57; // r12
  __int64 v58; // rax
  __int64 v59; // rdx
  __int64 v60; // rax
  _QWORD *v61; // rbx
  _QWORD *v62; // r12
  __int64 v63; // rsi
  __int64 v65; // [rsp+18h] [rbp-258h]
  __int64 v67; // [rsp+58h] [rbp-218h]
  __int64 v68; // [rsp+70h] [rbp-200h] BYREF
  _QWORD *v69; // [rsp+78h] [rbp-1F8h]
  __int64 v70; // [rsp+80h] [rbp-1F0h]
  unsigned int v71; // [rsp+88h] [rbp-1E8h]
  _QWORD *v72; // [rsp+98h] [rbp-1D8h]
  unsigned int v73; // [rsp+A8h] [rbp-1C8h]
  char v74; // [rsp+B0h] [rbp-1C0h]
  char v75; // [rsp+B9h] [rbp-1B7h]
  _BYTE *v76; // [rsp+C0h] [rbp-1B0h] BYREF
  __int64 v77; // [rsp+C8h] [rbp-1A8h]
  _BYTE v78[64]; // [rsp+D0h] [rbp-1A0h] BYREF
  _BYTE *v79; // [rsp+110h] [rbp-160h] BYREF
  __int64 v80; // [rsp+118h] [rbp-158h]
  _BYTE v81[64]; // [rsp+120h] [rbp-150h] BYREF
  __int64 *v82; // [rsp+160h] [rbp-110h] BYREF
  __int64 v83; // [rsp+168h] [rbp-108h] BYREF
  __int64 v84; // [rsp+170h] [rbp-100h] BYREF
  __int64 v85; // [rsp+178h] [rbp-F8h]
  __int64 v86; // [rsp+180h] [rbp-F0h]
  __int64 v87; // [rsp+1B0h] [rbp-C0h]
  __int64 v88; // [rsp+1B8h] [rbp-B8h]
  __int64 v89; // [rsp+1C0h] [rbp-B0h]
  __int64 *v90; // [rsp+1D0h] [rbp-A0h] BYREF
  __int64 v91; // [rsp+1D8h] [rbp-98h] BYREF
  __int64 v92; // [rsp+1E0h] [rbp-90h] BYREF
  __int64 v93; // [rsp+1E8h] [rbp-88h]
  __int64 *i; // [rsp+1F0h] [rbp-80h]
  __int64 v95; // [rsp+220h] [rbp-50h]
  __int64 v96; // [rsp+228h] [rbp-48h]
  __int64 v97; // [rsp+230h] [rbp-40h]

  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = a1 + 32;
  *(_QWORD *)(a1 + 24) = 0x400000000LL;
  *(_BYTE *)(a1 + 96) = 0;
  *(_DWORD *)(a1 + 100) = 0;
  *(_QWORD *)(a1 + 104) = 0;
  *(_QWORD *)(a1 + 112) = 0;
  *(_QWORD *)(a1 + 120) = 0;
  *(_QWORD *)(a1 + 128) = a4;
  v13 = (_QWORD *)sub_22077B0(432);
  if ( v13 )
  {
    v13[1] = 0x400000000LL;
    *v13 = v13 + 2;
  }
  v14 = *(_QWORD *)(a1 + 112);
  *(_QWORD *)(a1 + 112) = v13;
  if ( v14 )
  {
    v15 = *(unsigned __int64 **)v14;
    v16 = (unsigned __int64 *)(*(_QWORD *)v14 + 104LL * *(unsigned int *)(v14 + 8));
    if ( *(unsigned __int64 **)v14 != v16 )
    {
      do
      {
        v16 -= 13;
        if ( (unsigned __int64 *)*v16 != v16 + 2 )
          _libc_free(*v16);
      }
      while ( v15 != v16 );
      v16 = *(unsigned __int64 **)v14;
    }
    if ( v16 != (unsigned __int64 *)(v14 + 16) )
      _libc_free((unsigned __int64)v16);
    j_j___libc_free_0(v14, 432);
  }
  v68 = 0;
  v71 = 128;
  v17 = (_QWORD *)sub_22077B0(0x2000);
  v70 = 0;
  v69 = v17;
  v91 = 2;
  v18 = v17 + 1024;
  v90 = (__int64 *)&unk_49E6B50;
  v92 = 0;
  v93 = -8;
  for ( i = 0; v18 != v17; v17 += 8 )
  {
    if ( v17 )
    {
      v19 = v91;
      v17[2] = 0;
      v17[3] = -8;
      *v17 = &unk_49E6B50;
      v17[1] = v19 & 6;
      v17[4] = i;
    }
  }
  v74 = 0;
  v75 = 1;
  v21 = sub_1AB6FF0(a2, &v68, 0);
  v67 = *(_QWORD *)a3;
  v20 = *(_QWORD *)a3;
  *(_QWORD *)(a1 + 8) = v21;
  v26 = *(unsigned int *)(a3 + 8);
  v27 = 3 * v26;
  v65 = v20 + 104 * v26;
  if ( v20 != v65 )
  {
    do
    {
      v82 = &v84;
      v83 = 0x800000000LL;
      if ( !*(_DWORD *)(v67 + 8) )
      {
        v31 = *(_QWORD *)(v67 + 80);
        v87 = v31;
        v88 = *(_QWORD *)(v67 + 88);
        v89 = *(_QWORD *)(v67 + 96);
        v76 = v78;
        v77 = 0x800000000LL;
        goto LABEL_36;
      }
      sub_18971B0((__int64)&v82, v67, v27, v20, v22, v23);
      v31 = *(_QWORD *)(v67 + 80);
      v39 = v82;
      v87 = v31;
      v40 = &v82[(unsigned int)v83];
      v88 = *(_QWORD *)(v67 + 88);
      v89 = *(_QWORD *)(v67 + 96);
      v76 = v78;
      v77 = 0x800000000LL;
      if ( v82 != v40 )
      {
        while ( 1 )
        {
          v41 = *v39;
          v91 = 2;
          v92 = 0;
          v93 = v41;
          if ( v41 != 0 && v41 != -8 && v41 != -16 )
            sub_164C220((__int64)&v91);
          v42 = v71;
          i = &v68;
          v90 = (__int64 *)&unk_49E6B50;
          if ( !v71 )
            break;
          v45 = v93;
          v43 = v71 - 1;
          v52 = (v71 - 1) & (((unsigned int)v93 >> 9) ^ ((unsigned int)v93 >> 4));
          v44 = &v69[8 * (unsigned __int64)v52];
          v53 = v44[3];
          if ( v53 != v93 )
          {
            v54 = 1;
            v38 = 0;
            while ( v53 != -8 )
            {
              if ( !v38 && v53 == -16 )
                v38 = v44;
              v52 = v43 & (v54 + v52);
              v44 = &v69[8 * (unsigned __int64)v52];
              v53 = v44[3];
              if ( v93 == v53 )
                goto LABEL_59;
              ++v54;
            }
            if ( v38 )
              v44 = v38;
            ++v68;
            v46 = v70 + 1;
            if ( 4 * ((int)v70 + 1) < 3 * v71 )
            {
              if ( v71 - HIDWORD(v70) - v46 <= v71 >> 3 )
              {
LABEL_48:
                sub_12E48B0((__int64)&v68, v42);
                sub_12E4800((__int64)&v68, (__int64)&v90, &v79);
                v44 = v79;
                v45 = v93;
                v46 = v70 + 1;
              }
              v47 = v44[3];
              LODWORD(v70) = v46;
              v48 = v44 + 1;
              if ( v47 == -8 )
              {
                if ( v45 != -8 )
                  goto LABEL_54;
              }
              else
              {
                --HIDWORD(v70);
                if ( v47 != v45 )
                {
                  if ( v47 && v47 != -16 )
                  {
                    sub_1649B30(v48);
                    v45 = v93;
                    v48 = v44 + 1;
                  }
LABEL_54:
                  v44[3] = v45;
                  if ( v45 != 0 && v45 != -8 && v45 != -16 )
                    sub_1649AC0(v48, v91 & 0xFFFFFFFFFFFFFFF8LL);
                  v45 = v93;
                }
              }
              v49 = i;
              v44[5] = 6;
              v44[6] = 0;
              v44[4] = v49;
              v44[7] = 0;
              goto LABEL_59;
            }
LABEL_47:
            v42 = 2 * v71;
            goto LABEL_48;
          }
LABEL_59:
          v90 = (__int64 *)&unk_49EE2B0;
          if ( v45 != -8 && v45 != 0 && v45 != -16 )
            sub_1649B30(&v91);
          v50 = v44[7];
          v51 = (unsigned int)v77;
          if ( (unsigned int)v77 >= HIDWORD(v77) )
          {
            sub_16CD150((__int64)&v76, v78, 0, 8, v43, (int)v38);
            v51 = (unsigned int)v77;
          }
          ++v39;
          *(_QWORD *)&v76[8 * v51] = v50;
          LODWORD(v77) = v77 + 1;
          if ( v40 == v39 )
          {
            v31 = v87;
            goto LABEL_36;
          }
        }
        ++v68;
        goto LABEL_47;
      }
LABEL_36:
      v32 = sub_189DAB0((__int64)&v68, v31)[2];
      v33 = sub_189DAB0((__int64)&v68, v88);
      v35 = v89;
      v36 = v33[2];
      if ( v89 )
        v35 = sub_189DAB0((__int64)&v68, v89)[2];
      v79 = v81;
      v80 = 0x800000000LL;
      if ( (_DWORD)v77 )
      {
        sub_18971B0((__int64)&v79, (__int64)&v76, v34, (unsigned int)v77, v22, v23);
        v90 = &v92;
        v91 = 0x800000000LL;
        if ( (_DWORD)v80 )
          sub_18971B0((__int64)&v90, (__int64)&v79, (unsigned int)v80, v37, v22, v23);
      }
      else
      {
        v91 = 0x800000000LL;
        v90 = &v92;
      }
      v95 = v32;
      v96 = v36;
      v97 = v35;
      if ( v79 != v81 )
        _libc_free((unsigned __int64)v79);
      v28 = *(_QWORD *)(a1 + 112);
      v29 = *(_DWORD *)(v28 + 8);
      if ( v29 >= *(_DWORD *)(v28 + 12) )
      {
        sub_18976A0(*(_QWORD *)(a1 + 112));
        v29 = *(_DWORD *)(v28 + 8);
      }
      v20 = 13LL * v29;
      v27 = *(_QWORD *)v28;
      v30 = (_QWORD *)(*(_QWORD *)v28 + 104LL * v29);
      if ( v30 )
      {
        *v30 = v30 + 2;
        v30[1] = 0x800000000LL;
        if ( (_DWORD)v91 )
          sub_18971B0((__int64)v30, (__int64)&v90, v27, v20, v22, v23);
        v30[10] = v95;
        v30[11] = v96;
        v30[12] = v97;
        v29 = *(_DWORD *)(v28 + 8);
      }
      *(_DWORD *)(v28 + 8) = v29 + 1;
      if ( v90 != &v92 )
        _libc_free((unsigned __int64)v90);
      if ( v76 != v78 )
        _libc_free((unsigned __int64)v76);
      if ( v82 != &v84 )
        _libc_free((unsigned __int64)v82);
      v67 += 104;
    }
    while ( v65 != v67 );
    v21 = *(_QWORD *)(a1 + 8);
  }
  sub_164D160(a2, v21, a5, a6, a7, a8, v24, v25, a11, a12);
  if ( v74 )
  {
    if ( v73 )
    {
      v61 = v72;
      v62 = &v72[2 * v73];
      do
      {
        if ( *v61 != -8 && *v61 != -4 )
        {
          v63 = v61[1];
          if ( v63 )
            sub_161E7C0((__int64)(v61 + 1), v63);
        }
        v61 += 2;
      }
      while ( v62 != v61 );
    }
    j___libc_free_0(v72);
  }
  if ( v71 )
  {
    v56 = v69;
    v83 = 2;
    v84 = 0;
    v57 = &v69[8 * (unsigned __int64)v71];
    v85 = -8;
    v58 = -8;
    v82 = (__int64 *)&unk_49E6B50;
    v86 = 0;
    v91 = 2;
    v92 = 0;
    v93 = -16;
    v90 = (__int64 *)&unk_49E6B50;
    i = 0;
    while ( 1 )
    {
      v59 = v56[3];
      if ( v59 != v58 )
      {
        v58 = v93;
        if ( v59 != v93 )
        {
          v60 = v56[7];
          if ( v60 != -8 && v60 != 0 && v60 != -16 )
          {
            sub_1649B30(v56 + 5);
            v59 = v56[3];
          }
          v58 = v59;
        }
      }
      *v56 = &unk_49EE2B0;
      if ( v58 != -8 && v58 != 0 && v58 != -16 )
        sub_1649B30(v56 + 1);
      v56 += 8;
      if ( v57 == v56 )
        break;
      v58 = v85;
    }
    v90 = (__int64 *)&unk_49EE2B0;
    if ( v93 != 0 && v93 != -8 && v93 != -16 )
      sub_1649B30(&v91);
    v82 = (__int64 *)&unk_49EE2B0;
    if ( v85 != 0 && v85 != -8 && v85 != -16 )
      sub_1649B30(&v83);
  }
  return j___libc_free_0(v69);
}
