// Function: sub_2F59030
// Address: 0x2f59030
//
__int64 __fastcall sub_2F59030(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r15
  __int64 v4; // rax
  __int64 *v5; // rdi
  __int64 (__fastcall *v6)(__int64); // rax
  __int64 v7; // rax
  __int64 (*v8)(); // r8
  __int64 (*v9)(); // r8
  __int64 **v10; // r12
  __int64 **v11; // r14
  int v12; // ebx
  __int64 v13; // rax
  __int64 v14; // r8
  __int64 v15; // rbx
  int v16; // r14d
  int v17; // r14d
  unsigned int v18; // r13d
  __int64 v19; // r14
  __int64 v20; // r12
  __int64 v21; // rax
  __int64 v22; // rcx
  __int64 v23; // rdx
  unsigned int *v24; // r12
  __int64 v25; // r15
  unsigned int v26; // eax
  char *v27; // rdi
  int v28; // esi
  char *v29; // rcx
  unsigned __int64 v30; // r14
  __int64 *v31; // rbx
  unsigned __int64 v32; // rsi
  bool v33; // al
  __int64 **v34; // rbx
  int v35; // r13d
  __int64 *v36; // r12
  __int64 v37; // rax
  double v38; // xmm0_8
  __int64 v39; // rax
  double v40; // xmm1_8
  float v41; // xmm0_4
  _DWORD *v43; // rbx
  _DWORD *v44; // rax
  signed int v45; // r12d
  signed int v46; // r13d
  int v47; // eax
  unsigned __int64 v48; // rdi
  __int64 v49; // rdx
  unsigned int *v50; // rdi
  __int64 v51; // rdi
  __int64 v52; // rcx
  __int64 v53; // rdx
  __int64 v54; // rdx
  int *v55; // rdi
  int *v56; // rax
  __int64 v57; // [rsp+8h] [rbp-218h]
  bool v59; // [rsp+33h] [rbp-1EDh]
  int v60; // [rsp+34h] [rbp-1ECh]
  int v61; // [rsp+38h] [rbp-1E8h]
  int v62; // [rsp+3Ch] [rbp-1E4h]
  int v63; // [rsp+40h] [rbp-1E0h]
  int v64; // [rsp+44h] [rbp-1DCh]
  int *v66; // [rsp+50h] [rbp-1D0h]
  __int64 v67; // [rsp+58h] [rbp-1C8h]
  __int64 v68; // [rsp+60h] [rbp-1C0h]
  bool v70; // [rsp+78h] [rbp-1A8h]
  int v71; // [rsp+88h] [rbp-198h] BYREF
  unsigned int v72; // [rsp+8Ch] [rbp-194h] BYREF
  _QWORD v73[2]; // [rsp+90h] [rbp-190h] BYREF
  char v74; // [rsp+A0h] [rbp-180h]
  _BYTE v75[32]; // [rsp+B0h] [rbp-170h] BYREF
  __int64 **v76; // [rsp+D0h] [rbp-150h] BYREF
  __int64 v77; // [rsp+D8h] [rbp-148h]
  _BYTE v78[16]; // [rsp+E0h] [rbp-140h] BYREF
  unsigned int *v79; // [rsp+F0h] [rbp-130h] BYREF
  __int64 v80; // [rsp+F8h] [rbp-128h]
  _BYTE v81[72]; // [rsp+100h] [rbp-120h] BYREF
  int v82; // [rsp+148h] [rbp-D8h] BYREF
  unsigned __int64 v83; // [rsp+150h] [rbp-D0h]
  int *v84; // [rsp+158h] [rbp-C8h]
  int *v85; // [rsp+160h] [rbp-C0h]
  __int64 v86; // [rsp+168h] [rbp-B8h]
  char *v87; // [rsp+170h] [rbp-B0h] BYREF
  __int64 v88; // [rsp+178h] [rbp-A8h]
  _BYTE v89[72]; // [rsp+180h] [rbp-A0h] BYREF
  __int64 v90; // [rsp+1C8h] [rbp-58h] BYREF
  unsigned __int64 v91; // [rsp+1D0h] [rbp-50h]
  __int64 *v92; // [rsp+1D8h] [rbp-48h]
  __int64 *v93; // [rsp+1E0h] [rbp-40h]
  __int64 v94; // [rsp+1E8h] [rbp-38h]

  v3 = *(_QWORD *)(a3 + 56);
  v4 = *(_QWORD *)(*(_QWORD *)(a2 + 768) + 48LL);
  *(_QWORD *)(a1 + 32) = 0;
  v68 = v4;
  *(_DWORD *)(a1 + 40) = 0;
  v67 = a3 + 48;
  *(_OWORD *)a1 = 0;
  *(_OWORD *)(a1 + 16) = 0;
  if ( v3 != a3 + 48 )
  {
    v60 = 0;
    v63 = 0;
    v61 = 0;
    v62 = 0;
    v64 = 0;
    while ( 1 )
    {
      if ( *(_WORD *)(v3 + 68) == 20 )
      {
        v43 = *(_DWORD **)(v3 + 32);
        v44 = v43 + 10;
      }
      else
      {
        v5 = *(__int64 **)(a2 + 776);
        v6 = *(__int64 (__fastcall **)(__int64))(*v5 + 520);
        if ( v6 == sub_2DCA430 )
          goto LABEL_5;
        ((void (__fastcall *)(_QWORD *, _QWORD, __int64))v6)(v73, *(_QWORD *)(a2 + 776), v3);
        if ( !v74 )
        {
          v5 = *(__int64 **)(a2 + 776);
LABEL_5:
          v76 = (__int64 **)v78;
          v77 = 0x200000000LL;
          v7 = *v5;
          v8 = *(__int64 (**)())(*v5 + 88);
          if ( v8 == sub_2E97330 )
          {
            v9 = *(__int64 (**)())(v7 + 120);
            if ( v9 == sub_2F4C0B0 )
            {
LABEL_7:
              if ( (*(unsigned __int8 (__fastcall **)(__int64 *, __int64, __int64 ***))(v7 + 112))(v5, v3, &v76) )
              {
                v10 = v76;
                v11 = &v76[(unsigned int)v77];
                v12 = v77;
                if ( v11 != sub_2F4F320(v76, (__int64)v11, v68) )
                {
                  v59 = *(_WORD *)(v3 + 68) == 32 || ((*(_WORD *)(v3 + 68) - 26) & 0xFFFD) == 0;
                  if ( !v59 )
                  {
                    v63 += v12;
                    *(_DWORD *)(a1 + 4) = v63;
                    goto LABEL_49;
                  }
                  v13 = (*(__int64 (__fastcall **)(_QWORD, __int64))(**(_QWORD **)(a2 + 776) + 592LL))(
                          *(_QWORD *)(a2 + 776),
                          v3);
                  LODWORD(v90) = 0;
                  v15 = v13;
                  v92 = &v90;
                  v79 = (unsigned int *)v81;
                  v84 = &v82;
                  v85 = &v82;
                  v82 = 0;
                  v83 = 0;
                  v86 = 0;
                  v91 = 0;
                  v93 = &v90;
                  v94 = 0;
                  v16 = *(_DWORD *)(v3 + 40);
                  v66 = &v82;
                  v80 = 0x1000000000LL;
                  v87 = v89;
                  v88 = 0x1000000000LL;
                  v17 = v16 & 0xFFFFFF;
                  if ( v17 )
                  {
                    v18 = v17;
                    v19 = v3;
                    v20 = 0;
                    do
                    {
                      v21 = *(_QWORD *)(v3 + 32) + 40 * v20;
                      if ( *(_BYTE *)v21 == 5 )
                      {
                        v22 = *(_QWORD *)(v68 + 8);
                        if ( *(_BYTE *)(v22 + 40LL * (unsigned int)(*(_DWORD *)(v68 + 32) + *(_DWORD *)(v21 + 24)) + 18) )
                        {
                          v72 = *(_DWORD *)(v21 + 24);
                          if ( (unsigned int)v20 >= (unsigned int)v15 && (unsigned int)v20 < HIDWORD(v15) )
                            sub_2F58E50((__int64)v75, (__int64)&v79, &v72, v22, v14);
                          else
                            sub_2F58E50((__int64)v75, (__int64)&v87, &v72, v22, v14);
                        }
                      }
                      ++v20;
                    }
                    while ( v18 > (unsigned int)v20 );
                    v23 = v94;
                    if ( v86 )
                    {
                      v70 = 0;
                      v24 = (unsigned int *)v84;
                      v25 = v94;
                      v57 = v19;
LABEL_20:
                      if ( !v70 )
                        goto LABEL_21;
                      while ( 2 )
                      {
                        if ( v66 == (int *)v24 )
                          goto LABEL_73;
                        v26 = *v24;
                        if ( !v25 )
                          goto LABEL_23;
LABEL_35:
                        if ( v91 )
                        {
                          v30 = v91;
                          v31 = &v90;
                          while ( 1 )
                          {
                            while ( v26 > *(_DWORD *)(v30 + 32) )
                            {
                              v30 = *(_QWORD *)(v30 + 24);
                              if ( !v30 )
                                goto LABEL_41;
                            }
                            v32 = *(_QWORD *)(v30 + 16);
                            if ( v26 >= *(_DWORD *)(v30 + 32) )
                              break;
                            v31 = (__int64 *)v30;
                            v30 = *(_QWORD *)(v30 + 16);
                            if ( !v32 )
                            {
LABEL_41:
                              v33 = v31 == &v90;
                              goto LABEL_42;
                            }
                          }
                          v51 = *(_QWORD *)(v30 + 24);
                          if ( v51 )
                          {
                            do
                            {
                              while ( 1 )
                              {
                                v52 = *(_QWORD *)(v51 + 16);
                                v53 = *(_QWORD *)(v51 + 24);
                                if ( v26 < *(_DWORD *)(v51 + 32) )
                                  break;
                                v51 = *(_QWORD *)(v51 + 24);
                                if ( !v53 )
                                  goto LABEL_99;
                              }
                              v31 = (__int64 *)v51;
                              v51 = *(_QWORD *)(v51 + 16);
                            }
                            while ( v52 );
                          }
LABEL_99:
                          while ( v32 )
                          {
                            while ( 1 )
                            {
                              v54 = *(_QWORD *)(v32 + 24);
                              if ( v26 <= *(_DWORD *)(v32 + 32) )
                                break;
                              v32 = *(_QWORD *)(v32 + 24);
                              if ( !v54 )
                                goto LABEL_102;
                            }
                            v30 = v32;
                            v32 = *(_QWORD *)(v32 + 16);
                          }
LABEL_102:
                          if ( v92 != (__int64 *)v30 || v31 != &v90 )
                          {
                            for ( ; v31 != (__int64 *)v30; --v94 )
                            {
                              v55 = (int *)v30;
                              v30 = sub_220EF30(v30);
                              v56 = sub_220F330(v55, &v90);
                              j_j___libc_free_0((unsigned __int64)v56);
                              v25 = v94 - 1;
                            }
LABEL_31:
                            if ( !v70 )
                              goto LABEL_45;
LABEL_32:
                            ++v24;
                            continue;
                          }
                        }
                        else
                        {
                          v33 = v59;
                          v31 = &v90;
LABEL_42:
                          if ( v92 != v31 || !v33 )
                            goto LABEL_31;
                        }
                        break;
                      }
                      v25 = 0;
                      sub_2F4DD80(v91);
                      v92 = &v90;
                      v91 = 0;
                      v93 = &v90;
                      v94 = 0;
                      if ( !v70 )
                      {
LABEL_45:
                        v24 = (unsigned int *)sub_220EF30((__int64)v24);
LABEL_21:
                        if ( v66 != (int *)v24 )
                        {
                          v26 = v24[8];
                          if ( v25 )
                            goto LABEL_35;
LABEL_23:
                          v27 = v87;
                          v28 = v88;
                          v29 = &v87[4 * (unsigned int)v88];
                          if ( v87 != v29 )
                          {
                            while ( v26 != *(_DWORD *)v27 )
                            {
                              v27 += 4;
                              if ( v29 == v27 )
                                goto LABEL_31;
                            }
                            if ( v29 != v27 )
                            {
                              if ( v29 != v27 + 4 )
                              {
                                memmove(v27, v27 + 4, v29 - (v27 + 4));
                                v28 = v88;
                                v25 = v94;
                              }
                              LODWORD(v88) = v28 - 1;
                            }
                          }
                          goto LABEL_31;
                        }
LABEL_73:
                        v47 = v86;
                        v48 = v91;
                        v49 = v25;
                        v3 = v57;
                        if ( !v86 )
                          v47 = v80;
                        v63 += v47;
                        if ( !v49 )
                          LODWORD(v49) = v88;
                        *(_DWORD *)(a1 + 4) = v63;
                        *(_DWORD *)(a1 + 8) += v49;
                        sub_2F4DD80(v48);
                        if ( v87 != v89 )
                          _libc_free((unsigned __int64)v87);
                        sub_2F4DD80(v83);
                        if ( v79 != (unsigned int *)v81 )
                          _libc_free((unsigned __int64)v79);
LABEL_81:
                        v10 = v76;
                        goto LABEL_49;
                      }
                      goto LABEL_32;
                    }
                    v50 = v79;
                    v66 = (int *)&v79[(unsigned int)v80];
                  }
                  else
                  {
                    v23 = 0;
                    v66 = (int *)v81;
                    v50 = (unsigned int *)v81;
                  }
                  v57 = v3;
                  v24 = v50;
                  v25 = v23;
                  v70 = v59;
                  goto LABEL_20;
                }
              }
              LODWORD(v77) = 0;
              if ( !(*(unsigned __int8 (__fastcall **)(_QWORD, __int64, __int64 ***))(**(_QWORD **)(a2 + 776) + 144LL))(
                      *(_QWORD *)(a2 + 776),
                      v3,
                      &v76) )
                goto LABEL_81;
              v10 = v76;
              v34 = &v76[(unsigned int)v77];
              v35 = v77;
              if ( v34 != sub_2F4F320(v76, (__int64)v34, v68) )
              {
                v60 += v35;
                *(_DWORD *)(a1 + 16) = v60;
              }
LABEL_49:
              if ( v10 != (__int64 **)v78 )
                _libc_free((unsigned __int64)v10);
              goto LABEL_51;
            }
          }
          else
          {
            if ( ((unsigned int (__fastcall *)(__int64 *, __int64, int *))v8)(v5, v3, &v71)
              && *(_BYTE *)(*(_QWORD *)(v68 + 8) + 40LL * (unsigned int)(*(_DWORD *)(v68 + 32) + v71) + 18) )
            {
              ++v62;
              v10 = v76;
              *(_DWORD *)a1 = v62;
              goto LABEL_49;
            }
            v5 = *(__int64 **)(a2 + 776);
            v7 = *v5;
            v9 = *(__int64 (**)())(*v5 + 120);
            if ( v9 == sub_2F4C0B0 )
              goto LABEL_7;
          }
          if ( ((unsigned int (__fastcall *)(__int64 *, __int64, int *))v9)(v5, v3, &v71)
            && *(_BYTE *)(*(_QWORD *)(v68 + 8) + 40LL * (unsigned int)(*(_DWORD *)(v68 + 32) + v71) + 18) )
          {
            ++v61;
            v10 = v76;
            *(_DWORD *)(a1 + 12) = v61;
            goto LABEL_49;
          }
          v5 = *(__int64 **)(a2 + 776);
          v7 = *v5;
          goto LABEL_7;
        }
        v43 = (_DWORD *)v73[0];
        v44 = (_DWORD *)v73[1];
      }
      v45 = v44[2];
      v46 = v43[2];
      if ( v45 < 0 )
      {
        v45 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 24) + 32LL) + 4LL * (v45 & 0x7FFFFFFF));
        if ( v45 && ((*v44 >> 8) & 0xFFF) != 0 )
        {
          v45 = sub_E91CF0(*(_QWORD **)(a2 + 8), v45, (*v44 >> 8) & 0xFFF);
          if ( v46 < 0 )
          {
LABEL_64:
            v46 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 24) + 32LL) + 4LL * (v46 & 0x7FFFFFFF));
            if ( v46 && ((*v43 >> 8) & 0xFFF) != 0 )
              v46 = sub_E91CF0(*(_QWORD **)(a2 + 8), v46, (*v43 >> 8) & 0xFFF);
          }
        }
        else if ( v46 < 0 )
        {
          goto LABEL_64;
        }
        if ( v46 != v45 )
          *(_DWORD *)(a1 + 20) = ++v64;
        goto LABEL_51;
      }
      if ( v46 < 0 )
        goto LABEL_64;
LABEL_51:
      if ( (*(_BYTE *)v3 & 4) == 0 )
      {
        while ( (*(_BYTE *)(v3 + 44) & 8) != 0 )
          v3 = *(_QWORD *)(v3 + 8);
      }
      v3 = *(_QWORD *)(v3 + 8);
      if ( v3 == v67 )
        goto LABEL_53;
    }
  }
  v64 = 0;
  v60 = 0;
  v61 = 0;
  v63 = 0;
  v62 = 0;
LABEL_53:
  v36 = *(__int64 **)(a2 + 792);
  v37 = sub_2E39EA0(v36, a3);
  if ( v37 < 0 )
    v38 = (double)(int)(v37 & 1 | ((unsigned __int64)v37 >> 1)) + (double)(int)(v37 & 1 | ((unsigned __int64)v37 >> 1));
  else
    v38 = (double)(int)v37;
  v39 = sub_2E3A080((__int64)v36);
  if ( v39 < 0 )
    v40 = (double)(int)(v39 & 1 | ((unsigned __int64)v39 >> 1)) + (double)(int)(v39 & 1 | ((unsigned __int64)v39 >> 1));
  else
    v40 = (double)(int)v39;
  v41 = v38 / v40;
  *(float *)(a1 + 24) = (float)v62 * v41;
  *(float *)(a1 + 28) = (float)v63 * v41;
  *(float *)(a1 + 32) = (float)v61 * v41;
  *(float *)(a1 + 36) = (float)v60 * v41;
  *(float *)(a1 + 40) = v41 * (float)v64;
  return a1;
}
