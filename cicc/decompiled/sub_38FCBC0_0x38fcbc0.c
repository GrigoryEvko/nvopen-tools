// Function: sub_38FCBC0
// Address: 0x38fcbc0
//
__int64 __fastcall sub_38FCBC0(__int64 a1, char a2, char a3, __m128 a4, double a5, double a6)
{
  int v7; // eax
  __int64 v8; // rdi
  bool v9; // zf
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // r12
  int v13; // r9d
  __int64 v14; // rax
  unsigned __int64 v15; // r13
  const char *v16; // r14
  __int64 v17; // rdi
  __int64 v18; // rdi
  void (*v19)(); // rax
  __int64 v20; // rax
  unsigned __int64 v21; // rax
  __int64 *v22; // rdi
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // r12
  __int64 v26; // r15
  __int64 v27; // r12
  __int64 v28; // r10
  int v29; // r13d
  __int64 v30; // rax
  unsigned __int64 v31; // rax
  __int64 *v32; // rdi
  __int64 v33; // rdx
  __int64 v34; // r14
  __int64 *v35; // rdi
  int v36; // eax
  unsigned __int64 v37; // rsi
  unsigned __int64 *v38; // r15
  unsigned __int64 v39; // rax
  unsigned int v40; // r12d
  __int64 v42; // r14
  __int64 v43; // r13
  unsigned __int64 v44; // rax
  unsigned __int64 v45; // rsi
  __int64 *v46; // rdi
  __int64 v47; // rax
  unsigned int v48; // r8d
  size_t v49; // rdx
  char *v50; // rdi
  __int64 v51; // r13
  unsigned __int64 v52; // rdi
  unsigned __int64 v53; // r13
  __int64 v54; // rdi
  __int64 v55; // rax
  int v56; // ecx
  _QWORD *v57; // rsi
  __int64 *v58; // r12
  __int64 *v59; // r13
  unsigned __int64 *v60; // r10
  __int64 v61; // rax
  unsigned __int64 v62; // rcx
  __int64 v63; // rax
  __int64 *v64; // rdx
  __int64 *v65; // rax
  __int64 v66; // rdx
  __int64 v67; // r13
  __int64 *v68; // rax
  __int64 v69; // rdx
  unsigned __int64 v70; // rsi
  __int64 v71; // rax
  unsigned __int64 v72; // rax
  __int64 *v73; // rdi
  unsigned __int64 *v74; // rsi
  char v75; // [rsp+37h] [rbp-339h]
  int v77; // [rsp+44h] [rbp-32Ch]
  unsigned int v78; // [rsp+50h] [rbp-320h]
  unsigned int v79; // [rsp+50h] [rbp-320h]
  __int64 v80; // [rsp+80h] [rbp-2F0h]
  __int64 v81; // [rsp+88h] [rbp-2E8h]
  __int64 v82; // [rsp+88h] [rbp-2E8h]
  unsigned __int64 *v83; // [rsp+88h] [rbp-2E8h]
  unsigned __int64 v84; // [rsp+90h] [rbp-2E0h] BYREF
  unsigned __int64 v85; // [rsp+98h] [rbp-2D8h]
  unsigned __int64 v86[2]; // [rsp+A0h] [rbp-2D0h] BYREF
  __int16 v87; // [rsp+B0h] [rbp-2C0h]
  const char *v88; // [rsp+C0h] [rbp-2B0h] BYREF
  unsigned __int64 v89; // [rsp+C8h] [rbp-2A8h]
  _WORD v90[32]; // [rsp+D0h] [rbp-2A0h] BYREF
  int v91; // [rsp+110h] [rbp-260h]
  char v92; // [rsp+114h] [rbp-25Ch]
  unsigned __int64 *v93; // [rsp+118h] [rbp-258h]
  const char *v94; // [rsp+120h] [rbp-250h] BYREF
  const char *v95; // [rsp+128h] [rbp-248h] BYREF
  __int64 v96; // [rsp+130h] [rbp-240h]
  _BYTE dest[64]; // [rsp+138h] [rbp-238h] BYREF
  __m128 v98; // [rsp+178h] [rbp-1F8h]
  unsigned __int64 v99[2]; // [rsp+190h] [rbp-1E0h] BYREF
  _BYTE v100[464]; // [rsp+1A0h] [rbp-1D0h] BYREF

  if ( !a2 )
    (*(void (__fastcall **)(_QWORD, _QWORD))(**(_QWORD **)(a1 + 328) + 168LL))(*(_QWORD *)(a1 + 328), 0);
  sub_38EB180(a1);
  v7 = *(_DWORD *)(a1 + 380);
  v8 = *(_QWORD *)(a1 + 320);
  *(_BYTE *)(a1 + 17) = 0;
  v77 = v7;
  v9 = *(_BYTE *)(v8 + 1041) == 0;
  v75 = *(_BYTE *)(a1 + 385);
  v99[0] = (unsigned __int64)v100;
  v99[1] = 0x400000000LL;
  if ( !v9 )
  {
    v10 = *(_QWORD *)(a1 + 328);
    v11 = *(unsigned int *)(v10 + 120);
    if ( !(_DWORD)v11 )
      BUG();
    v12 = *(_QWORD *)(*(_QWORD *)(v10 + 112) + 32 * v11 - 32);
    if ( !*(_QWORD *)(v12 + 8) )
    {
      v67 = sub_38BFA60(v8, 1);
      (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a1 + 328) + 176LL))(*(_QWORD *)(a1 + 328), v67, 0);
      *(_QWORD *)(v12 + 8) = v67;
      v8 = *(_QWORD *)(a1 + 320);
    }
    v94 = (const char *)v12;
    sub_38EA790(v8 + 1048, (__int64 *)&v94);
  }
  if ( **(_DWORD **)(a1 + 152) )
  {
    while ( 1 )
    {
      v91 = -1;
      v88 = (const char *)v90;
      v89 = 0x800000000LL;
      v92 = 0;
      v93 = v99;
      if ( (unsigned __int8)sub_38F9390(a1, (__int64)&v88, 0, a4, a5, a6) )
        break;
      v53 = (unsigned __int64)v88;
      v16 = &v88[8 * (unsigned int)v89];
      if ( v88 == v16 )
        goto LABEL_18;
      do
      {
        v54 = *((_QWORD *)v16 - 1);
        v16 -= 8;
        if ( v54 )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v54 + 8LL))(v54);
      }
      while ( (const char *)v53 != v16 );
LABEL_17:
      v16 = v88;
LABEL_18:
      if ( v16 != (const char *)v90 )
        _libc_free((unsigned __int64)v16);
      if ( !**(_DWORD **)(a1 + 152) )
        goto LABEL_21;
    }
    v14 = *(unsigned int *)(a1 + 32);
    if ( !(_DWORD)v14 )
    {
      if ( **(_DWORD **)(a1 + 152) != 1 )
      {
LABEL_12:
        v9 = *(_BYTE *)(a1 + 258) == 0;
        *(_DWORD *)(a1 + 32) = 0;
        if ( v9 )
LABEL_66:
          sub_38F0630(a1);
LABEL_13:
        v15 = (unsigned __int64)v88;
        v16 = &v88[8 * (unsigned int)v89];
        if ( v88 == v16 )
          goto LABEL_18;
        do
        {
          v17 = *((_QWORD *)v16 - 1);
          v16 -= 8;
          if ( v17 )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v17 + 8LL))(v17);
        }
        while ( (const char *)v15 != v16 );
        goto LABEL_17;
      }
      sub_38EB180(a1);
      v14 = *(unsigned int *)(a1 + 32);
    }
    v42 = *(_QWORD *)(a1 + 24);
    v80 = v42 + 104 * v14;
    if ( v42 != v80 )
    {
      v43 = *(_QWORD *)(a1 + 24);
      do
      {
        v47 = *(_QWORD *)v43;
        v95 = dest;
        v94 = (const char *)v47;
        v96 = 0x4000000000LL;
        v48 = *(_DWORD *)(v43 + 16);
        if ( v48 && &v95 != (const char **)(v43 + 8) )
        {
          v49 = v48;
          v50 = dest;
          if ( v48 <= 0x40
            || (v79 = *(_DWORD *)(v43 + 16),
                sub_16CD150((__int64)&v95, dest, v48, 1, v48, v13),
                v49 = *(unsigned int *)(v43 + 16),
                v50 = (char *)v95,
                v48 = v79,
                *(_DWORD *)(v43 + 16)) )
          {
            v78 = v48;
            memcpy(v50, *(const void **)(v43 + 8), v49);
            v48 = v78;
          }
          LODWORD(v96) = v48;
        }
        v44 = *(_QWORD *)(v43 + 88);
        a4 = (__m128)_mm_loadu_si128((const __m128i *)(v43 + 88));
        v45 = (unsigned __int64)v94;
        *(_BYTE *)(a1 + 17) = 1;
        v98 = a4;
        v87 = 262;
        v46 = *(__int64 **)(a1 + 344);
        v86[0] = (unsigned __int64)&v95;
        v84 = v44;
        v85 = a4.m128_u64[1];
        sub_16D14E0(v46, v45, 0, (__int64)v86, &v84, 1, 0, 0, 1u);
        sub_38E35B0((_QWORD *)a1);
        if ( v95 != dest )
          _libc_free((unsigned __int64)v95);
        v43 += 104;
      }
      while ( v80 != v43 );
      v14 = *(unsigned int *)(a1 + 32);
      v42 = *(_QWORD *)(a1 + 24);
    }
    v51 = v42 + 104 * v14;
    while ( v42 != v51 )
    {
      while ( 1 )
      {
        v51 -= 104;
        v52 = *(_QWORD *)(v51 + 8);
        if ( v52 == v51 + 24 )
          break;
        _libc_free(v52);
        if ( v42 == v51 )
        {
          v9 = *(_BYTE *)(a1 + 258) == 0;
          *(_DWORD *)(a1 + 32) = 0;
          if ( !v9 )
            goto LABEL_13;
          goto LABEL_66;
        }
      }
    }
    goto LABEL_12;
  }
LABEL_21:
  v18 = *(_QWORD *)(a1 + 8);
  v19 = *(void (**)())(*(_QWORD *)v18 + 168LL);
  if ( v19 != nullsub_1961 )
    ((void (__fastcall *)(__int64, _QWORD))v19)(v18, *(_QWORD *)(a1 + 328));
  if ( *(_DWORD *)(a1 + 380) != v77 || *(_BYTE *)(a1 + 385) != v75 )
  {
    v94 = "unmatched .ifs or .elses";
    LOWORD(v96) = 259;
    v20 = sub_3909460(a1);
    v21 = sub_39092A0(v20);
    *(_BYTE *)(a1 + 17) = 1;
    v22 = *(__int64 **)(a1 + 344);
    v88 = 0;
    v89 = 0;
    sub_16D14E0(v22, v21, 0, (__int64)&v94, (unsigned __int64 *)&v88, 1, 0, 0, 1u);
    sub_38E35B0((_QWORD *)a1);
  }
  v23 = *(_QWORD *)(a1 + 320);
  if ( *(_QWORD *)(v23 + 1016) )
  {
    v24 = *(_QWORD *)(v23 + 1000);
    v25 = *(_QWORD *)(v24 + 160);
    v26 = v25 + 72LL * *(unsigned int *)(v24 + 168);
    if ( v25 != v26 )
    {
      v27 = v25 + 72;
      v28 = v26;
      v29 = 0;
      while ( 1 )
      {
        ++v29;
        if ( v28 == v27 )
          break;
        if ( !*(_QWORD *)(v27 + 8) && v29 )
        {
          LODWORD(v86[0]) = v29;
          v88 = "unassigned file number: ";
          v81 = v28;
          v89 = v86[0];
          v90[0] = 2307;
          v94 = (const char *)&v88;
          LOWORD(v96) = 770;
          v95 = " for .file directives";
          v30 = sub_3909460(a1);
          v31 = sub_39092A0(v30);
          *(_BYTE *)(a1 + 17) = 1;
          v32 = *(__int64 **)(a1 + 344);
          v84 = 0;
          v85 = 0;
          sub_16D14E0(v32, v31, 0, (__int64)&v94, &v84, 1, 0, 0, 1u);
          sub_38E35B0((_QWORD *)a1);
          v28 = v81;
        }
        v27 += 72;
      }
    }
  }
  if ( !a3 )
  {
    if ( *(_BYTE *)(*(_QWORD *)(a1 + 336) + 18LL) )
    {
      v55 = *(_QWORD *)(a1 + 320);
      v56 = *(_DWORD *)(v55 + 576);
      if ( v56 )
      {
        v57 = *(_QWORD **)(v55 + 568);
        if ( !*v57 || *v57 == -8 )
        {
          v68 = v57 + 1;
          do
          {
            do
            {
              v69 = *v68;
              v58 = v68++;
            }
            while ( v69 == -8 );
          }
          while ( !v69 );
        }
        else
        {
          v58 = *(__int64 **)(v55 + 568);
        }
        v59 = &v57[v56];
        if ( v59 != v58 )
        {
          v60 = &v84;
          do
          {
            while ( 1 )
            {
              v61 = *(_QWORD *)(*v58 + 8);
              if ( (*(_BYTE *)(v61 + 8) & 1) != 0 && (*(_BYTE *)(v61 + 9) & 0xC) != 8 )
              {
                v62 = *(_QWORD *)v61 & 0xFFFFFFFFFFFFFFF8LL;
                if ( !v62 )
                {
                  v70 = 0;
                  if ( (*(_QWORD *)v61 & 4) != 0 )
                  {
                    v74 = *(unsigned __int64 **)(v61 - 8);
                    v62 = *v74;
                    v70 = (unsigned __int64)(v74 + 2);
                  }
                  v89 = (unsigned __int64)v60;
                  v90[0] = 1283;
                  v83 = v60;
                  v95 = "' not defined";
                  v84 = v70;
                  v85 = v62;
                  LOWORD(v96) = 770;
                  v88 = "assembler local symbol '";
                  v94 = (const char *)&v88;
                  v71 = sub_3909460(a1);
                  v72 = sub_39092A0(v71);
                  *(_BYTE *)(a1 + 17) = 1;
                  v73 = *(__int64 **)(a1 + 344);
                  v86[0] = 0;
                  v86[1] = 0;
                  sub_16D14E0(v73, v72, 0, (__int64)&v94, v86, 1, 0, 0, 1u);
                  sub_38E35B0((_QWORD *)a1);
                  v60 = v83;
                }
              }
              v63 = v58[1];
              v64 = v58 + 1;
              if ( !v63 || v63 == -8 )
                break;
              ++v58;
              if ( v64 == v59 )
                goto LABEL_36;
            }
            v65 = v58 + 2;
            do
            {
              do
              {
                v66 = *v65;
                v58 = v65++;
              }
              while ( v66 == -8 );
            }
            while ( !v66 );
          }
          while ( v58 != v59 );
        }
      }
    }
LABEL_36:
    v33 = *(unsigned int *)(a1 + 608);
    v34 = *(_QWORD *)(a1 + 600);
    if ( v34 != v34 + 56 * v33 )
    {
      v82 = v34 + 56 * v33;
      while ( 1 )
      {
        v38 = *(unsigned __int64 **)v34;
        if ( (**(_QWORD **)v34 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
          goto LABEL_39;
        if ( (*((_BYTE *)v38 + 9) & 0xC) == 8
          && (*((_BYTE *)v38 + 8) |= 4u, v39 = (unsigned __int64)sub_38CE440(v38[3]), *v38 = v39 | *v38 & 7, v39) )
        {
          v34 += 56;
          if ( v82 == v34 )
            break;
        }
        else
        {
          v35 = *(__int64 **)(a1 + 344);
          *(__m128i *)(a1 + 560) = _mm_loadu_si128((const __m128i *)(v34 + 8));
          *(__m128i *)(a1 + 576) = _mm_loadu_si128((const __m128i *)(v34 + 24));
          v36 = *(_DWORD *)(v34 + 40);
          BYTE1(v96) = 1;
          *(_DWORD *)(a1 + 592) = v36;
          LOBYTE(v96) = 3;
          v94 = "directional label undefined";
          v37 = *(_QWORD *)(v34 + 48);
          *(_BYTE *)(a1 + 17) = 1;
          v88 = 0;
          v89 = 0;
          sub_16D14E0(v35, v37, 0, (__int64)&v94, (unsigned __int64 *)&v88, 1, 0, 0, 1u);
          sub_38E35B0((_QWORD *)a1);
LABEL_39:
          v34 += 56;
          if ( v82 == v34 )
            break;
        }
      }
    }
    if ( *(_BYTE *)(a1 + 17) )
      goto LABEL_45;
    sub_38DDA30(*(_QWORD **)(a1 + 328));
  }
  if ( *(_BYTE *)(a1 + 17) )
  {
LABEL_45:
    v40 = 1;
    goto LABEL_46;
  }
  v40 = *(unsigned __int8 *)(*(_QWORD *)(a1 + 320) + 1481LL);
LABEL_46:
  if ( (_BYTE *)v99[0] != v100 )
    _libc_free(v99[0]);
  return v40;
}
