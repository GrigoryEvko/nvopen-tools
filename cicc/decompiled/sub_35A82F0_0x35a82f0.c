// Function: sub_35A82F0
// Address: 0x35a82f0
//
void __fastcall sub_35A82F0(
        __int64 *a1,
        unsigned int a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        _QWORD *a8)
{
  __int64 v9; // rdi
  __int64 (*v10)(); // rax
  __int64 *v11; // rax
  __int64 v12; // r13
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // r13
  __int64 v16; // rax
  unsigned __int64 *v17; // rbx
  unsigned __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  __int64 v23; // rbx
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 v30; // rax
  __int64 v31; // r12
  __int64 v32; // rax
  __int64 v33; // r11
  unsigned int v34; // r13d
  __int64 v35; // r12
  __int64 v36; // r15
  __int64 v37; // rbx
  __int64 v38; // r12
  __int64 v39; // rsi
  __int64 v40; // rax
  unsigned int v41; // ecx
  __int64 *v42; // rdx
  __int64 v43; // r8
  char v44; // bl
  __int64 v45; // rdi
  void (__fastcall *v46)(__int64, __int64, __int64, __int64, _BYTE *, _QWORD, _BYTE **, _QWORD); // rax
  __int64 v47; // rdi
  __int64 v48; // rsi
  void (__fastcall *v49)(__int64, __int64, __int64, _QWORD, _BYTE *, _QWORD, __int64 **, _QWORD); // rax
  int v50; // edx
  __int64 v51; // rcx
  unsigned int v52; // esi
  __int64 *v53; // rdi
  int v54; // r11d
  _QWORD *v55; // rax
  unsigned int v56; // ecx
  _QWORD *v57; // rdx
  __int64 *v58; // r8
  __int64 *v59; // rax
  int v60; // r9d
  int v61; // edx
  unsigned int v62; // edx
  __int64 v63; // r11
  int v64; // esi
  __int64 v65; // rcx
  char v66; // [rsp-8h] [rbp-258h]
  __int64 v69; // [rsp+20h] [rbp-230h]
  __int64 v70; // [rsp+28h] [rbp-228h]
  __int64 v71; // [rsp+40h] [rbp-210h]
  __int64 *v72; // [rsp+48h] [rbp-208h]
  __int64 v73; // [rsp+50h] [rbp-200h]
  unsigned int v76; // [rsp+6Ch] [rbp-1E4h]
  __int64 *v77; // [rsp+70h] [rbp-1E0h]
  unsigned int v78; // [rsp+78h] [rbp-1D8h]
  __int64 v80; // [rsp+88h] [rbp-1C8h] BYREF
  __int64 v81; // [rsp+90h] [rbp-1C0h] BYREF
  __int64 *v82; // [rsp+98h] [rbp-1B8h] BYREF
  __int64 v83; // [rsp+A0h] [rbp-1B0h] BYREF
  __int64 v84; // [rsp+A8h] [rbp-1A8h]
  __int64 v85; // [rsp+B0h] [rbp-1A0h]
  unsigned int v86; // [rsp+B8h] [rbp-198h]
  _BYTE *v87; // [rsp+C0h] [rbp-190h] BYREF
  __int64 v88; // [rsp+C8h] [rbp-188h]
  _BYTE v89[160]; // [rsp+D0h] [rbp-180h] BYREF
  _BYTE *v90; // [rsp+170h] [rbp-E0h] BYREF
  __int64 v91; // [rsp+178h] [rbp-D8h]
  _BYTE v92[208]; // [rsp+180h] [rbp-D0h] BYREF

  v9 = a1[4];
  v87 = v89;
  v80 = 0;
  v81 = 0;
  v88 = 0x400000000LL;
  v10 = *(__int64 (**)())(*(_QWORD *)v9 + 344LL);
  if ( v10 != sub_2DB1AE0 )
  {
    if ( ((unsigned __int8 (__fastcall *)(__int64, __int64, __int64 *, __int64 *, _BYTE **, _QWORD))v10)(
           v9,
           a3,
           &v80,
           &v81,
           &v87,
           0) )
    {
LABEL_35:
      if ( v87 != v89 )
        _libc_free((unsigned __int64)v87);
      return;
    }
    v11 = *(__int64 **)(a3 + 112);
    v73 = *v11;
    if ( *v11 == a3 )
      v73 = v11[1];
    v83 = 0;
    v84 = 0;
    v85 = 0;
    v86 = 0;
    if ( !a2 )
    {
      v12 = a3;
      v71 = v73;
      goto LABEL_29;
    }
    v78 = a2;
    v12 = a3;
    v69 = 8LL * (a2 - 1);
    v71 = v73;
    while ( 1 )
    {
      LOBYTE(v91) = 0;
      v70 = v12;
      v15 = sub_2E7AAE0(a1[1], 0, (__int64)v90, 0);
      v16 = *(unsigned int *)(a7 + 8);
      if ( v16 + 1 > (unsigned __int64)*(unsigned int *)(a7 + 12) )
      {
        sub_C8D5F0(a7, (const void *)(a7 + 16), v16 + 1, 8u, v13, v14);
        v16 = *(unsigned int *)(a7 + 8);
      }
      *(_QWORD *)(*(_QWORD *)a7 + 8 * v16) = v15;
      ++*(_DWORD *)(a7 + 8);
      v17 = (unsigned __int64 *)a1[6];
      sub_2E33BD0(a1[1] + 320, v15);
      v18 = *v17;
      v19 = *(_QWORD *)v15;
      *(_QWORD *)(v15 + 8) = v17;
      v18 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)v15 = v18 | v19 & 7;
      *(_QWORD *)(v18 + 8) = v15;
      *v17 = v15 | *v17 & 7;
      sub_2E33690(v70, v73, v15);
      sub_2E33F80(v15, v73, -1, v20, v21, v22);
      v23 = a1[5];
      sub_2E34D50(*(_QWORD *)(v23 + 32), (__int64 *)v15, v24, v25, v26, v27);
      v30 = *(unsigned int *)(v23 + 352);
      v31 = *(unsigned int *)(v23 + 192);
      if ( v30 + 1 > (unsigned __int64)*(unsigned int *)(v23 + 356) )
      {
        sub_C8D5F0(v23 + 344, (const void *)(v23 + 360), v30 + 1, 8u, v28, v29);
        v30 = *(unsigned int *)(v23 + 352);
      }
      *(_QWORD *)(*(_QWORD *)(v23 + 344) + 8 * v30) = v31;
      v32 = v71;
      ++*(_DWORD *)(v23 + 352);
      if ( v73 == v71 )
        v32 = v15;
      v71 = v32;
      v76 = 2 * a2 + 1 - v78;
      v72 = (__int64 *)(v15 + 40);
      v33 = v15;
      v34 = v78;
      do
      {
        v35 = a1[6];
        v36 = v33;
        v37 = *(_QWORD *)(v35 + 56);
        v38 = v35 + 48;
        if ( v37 != v38 )
        {
          while ( 1 )
          {
            if ( *(_WORD *)(v37 + 68) == 68 || !*(_WORD *)(v37 + 68) )
              goto LABEL_24;
            v39 = *(_QWORD *)(*a1 + 72);
            v40 = *(unsigned int *)(*a1 + 88);
            if ( (_DWORD)v40 )
            {
              v41 = (v40 - 1) & (((unsigned int)v37 >> 9) ^ ((unsigned int)v37 >> 4));
              v42 = (__int64 *)(v39 + 16LL * v41);
              v43 = *v42;
              if ( v37 == *v42 )
              {
LABEL_22:
                if ( v42 != (__int64 *)(v39 + 16 * v40) )
                {
                  if ( v34 == *((_DWORD *)v42 + 2) )
                    goto LABEL_43;
                  goto LABEL_24;
                }
              }
              else
              {
                v50 = 1;
                while ( v43 != -4096 )
                {
                  v60 = v50 + 1;
                  v41 = (v40 - 1) & (v50 + v41);
                  v42 = (__int64 *)(v39 + 16LL * v41);
                  v43 = *v42;
                  if ( *v42 == v37 )
                    goto LABEL_22;
                  v50 = v60;
                }
              }
            }
            if ( v34 == -1 )
            {
LABEL_43:
              v82 = sub_3599350((__int64)a1, v37, -1, 0);
              sub_359F080(a1, (__int64)v82, v78 == 1, v76, 0, a5);
              v77 = v82;
              sub_2E31040(v72, (__int64)v82);
              v51 = *(_QWORD *)(v36 + 48);
              v77[1] = v36 + 48;
              v51 &= 0xFFFFFFFFFFFFFFF8LL;
              *v77 = v51 | *v77 & 7;
              *(_QWORD *)(v51 + 8) = v77;
              v52 = v86;
              *(_QWORD *)(v36 + 48) = *(_QWORD *)(v36 + 48) & 7LL | (unsigned __int64)v77;
              if ( v52 )
              {
                v53 = v82;
                v54 = 1;
                v55 = 0;
                v56 = (v52 - 1) & (((unsigned int)v82 >> 9) ^ ((unsigned int)v82 >> 4));
                v57 = (_QWORD *)(v84 + 16LL * v56);
                v58 = (__int64 *)*v57;
                if ( (__int64 *)*v57 == v82 )
                {
LABEL_45:
                  v59 = v57 + 1;
LABEL_46:
                  *v59 = v37;
                  goto LABEL_24;
                }
                while ( v58 != (__int64 *)-4096LL )
                {
                  if ( !v55 && v58 == (__int64 *)-8192LL )
                    v55 = v57;
                  v56 = (v52 - 1) & (v54 + v56);
                  v57 = (_QWORD *)(v84 + 16LL * v56);
                  v58 = (__int64 *)*v57;
                  if ( v82 == (__int64 *)*v57 )
                    goto LABEL_45;
                  ++v54;
                }
                if ( !v55 )
                  v55 = v57;
                ++v83;
                v61 = v85 + 1;
                v90 = v55;
                if ( 4 * ((int)v85 + 1) < 3 * v52 )
                {
                  if ( v52 - HIDWORD(v85) - v61 <= v52 >> 3 )
                  {
                    sub_2E48800((__int64)&v83, v52);
                    sub_3547B30((__int64)&v83, (__int64 *)&v82, &v90);
                    v53 = v82;
                    v61 = v85 + 1;
                    v55 = v90;
                  }
LABEL_60:
                  LODWORD(v85) = v61;
                  if ( *v55 != -4096 )
                    --HIDWORD(v85);
                  *v55 = v53;
                  v59 = v55 + 1;
                  *v59 = 0;
                  goto LABEL_46;
                }
              }
              else
              {
                ++v83;
                v90 = 0;
              }
              sub_2E48800((__int64)&v83, 2 * v52);
              if ( v86 )
              {
                v53 = v82;
                v62 = (v86 - 1) & (((unsigned int)v82 >> 9) ^ ((unsigned int)v82 >> 4));
                v55 = (_QWORD *)(v84 + 16LL * v62);
                v63 = *v55;
                if ( v82 == (__int64 *)*v55 )
                {
LABEL_66:
                  v90 = v55;
                  v61 = v85 + 1;
                }
                else
                {
                  v64 = 1;
                  v65 = 0;
                  while ( v63 != -4096 )
                  {
                    if ( !v65 && v63 == -8192 )
                      v65 = (__int64)v55;
                    v62 = (v86 - 1) & (v64 + v62);
                    v55 = (_QWORD *)(v84 + 16LL * v62);
                    v63 = *v55;
                    if ( v82 == (__int64 *)*v55 )
                      goto LABEL_66;
                    ++v64;
                  }
                  if ( !v65 )
                    v65 = (__int64)v55;
                  v90 = (_BYTE *)v65;
                  v61 = v85 + 1;
                  v55 = (_QWORD *)v65;
                }
              }
              else
              {
                v53 = v82;
                v90 = 0;
                v61 = v85 + 1;
                v55 = 0;
              }
              goto LABEL_60;
            }
LABEL_24:
            if ( (*(_BYTE *)v37 & 4) != 0 )
            {
              v37 = *(_QWORD *)(v37 + 8);
              if ( v38 == v37 )
                goto LABEL_26;
            }
            else
            {
              while ( (*(_BYTE *)(v37 + 44) & 8) != 0 )
                v37 = *(_QWORD *)(v37 + 8);
              v37 = *(_QWORD *)(v37 + 8);
              if ( v38 == v37 )
              {
LABEL_26:
                v33 = v36;
                break;
              }
            }
          }
        }
        ++v34;
      }
      while ( a2 >= v34 );
      v12 = v33;
      v44 = v78 == 1;
      v66 = v78-- == 1;
      sub_359D220(a1, v33, *(_QWORD *)(*a8 + v69), v70, a3, a5, (__int64)&v83, a2, v76, v66);
      sub_359E620(a1, v12, *(_QWORD *)(*a8 + v69), v70, a3, a5, a6, (__int64)&v83, a2, v76, v44);
      v69 -= 8;
      if ( !v78 )
      {
LABEL_29:
        sub_2E32770(v73, a1[6], v12);
        (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)a1[4] + 360LL))(a1[4], a3, 0);
        v45 = a1[4];
        v46 = *(void (__fastcall **)(__int64, __int64, __int64, __int64, _BYTE *, _QWORD, _BYTE **, _QWORD))(*(_QWORD *)v45 + 368LL);
        v90 = 0;
        if ( v80 == a4 )
          v46(v45, a3, a3, v71, v87, (unsigned int)v88, &v90, 0);
        else
          v46(v45, a3, v71, a3, v87, (unsigned int)v88, &v90, 0);
        sub_9C6650(&v90);
        if ( *(_DWORD *)(a7 + 8) )
        {
          v47 = a1[4];
          v48 = *(_QWORD *)(*(_QWORD *)a7 + 8LL * *(unsigned int *)(a7 + 8) - 8);
          v90 = v92;
          v91 = 0x400000000LL;
          v49 = *(void (__fastcall **)(__int64, __int64, __int64, _QWORD, _BYTE *, _QWORD, __int64 **, _QWORD))(*(_QWORD *)v47 + 368LL);
          v82 = 0;
          v49(v47, v48, v73, 0, v92, 0, &v82, 0);
          sub_9C6650(&v82);
          if ( v90 != v92 )
            _libc_free((unsigned __int64)v90);
        }
        sub_C7D6A0(v84, 16LL * v86, 8);
        goto LABEL_35;
      }
    }
  }
}
