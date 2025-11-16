// Function: sub_1F7A810
// Address: 0x1f7a810
//
_QWORD *__fastcall sub_1F7A810(unsigned __int8 *a1, __int64 a2, double a3, double a4, __m128i a5)
{
  __int64 v7; // rax
  unsigned int v8; // esi
  __int64 v9; // r10
  __int64 v10; // r12
  __int64 v11; // r13
  char *v12; // rax
  unsigned __int8 v13; // r11
  int v14; // eax
  unsigned int v15; // r15d
  __int64 v16; // rsi
  __int64 *v17; // r14
  __int64 v18; // r15
  char v20; // al
  __int64 v21; // r10
  unsigned __int8 v22; // r11
  __int64 v23; // rax
  __int64 v24; // rdi
  __int64 (*v25)(); // rax
  __int64 v26; // rdx
  int v27; // eax
  __int64 *v28; // rsi
  void *v29; // r12
  __int64 v30; // r10
  unsigned __int8 v31; // r11
  __int64 v32; // r10
  __int64 v33; // r11
  __int64 v34; // rdi
  __int64 (*v35)(); // rax
  __int64 v36; // rax
  char v37; // al
  __int64 v38; // rsi
  __int64 *v39; // r14
  __int64 v40; // rcx
  __int64 v41; // rax
  __int64 v42; // rdx
  __int64 v43; // rsi
  __int64 v44; // r8
  __int64 v45; // r9
  __int64 v46; // rcx
  __int64 v47; // rbx
  __int64 v48; // r13
  __int64 v49; // rax
  __int64 v50; // r15
  __int64 v51; // r14
  __int64 v52; // rdx
  __int64 v53; // r14
  __int64 v54; // rbx
  __int64 v55; // r15
  __int64 v56; // r13
  __int64 v57; // r14
  __int64 v58; // rax
  char v59; // al
  __int128 v60; // [rsp-10h] [rbp-C0h]
  __int128 v61; // [rsp-10h] [rbp-C0h]
  __int64 v62; // [rsp+0h] [rbp-B0h]
  __int64 v63; // [rsp+8h] [rbp-A8h]
  __int64 v64; // [rsp+18h] [rbp-98h]
  __int64 v65; // [rsp+18h] [rbp-98h]
  __int64 v66; // [rsp+20h] [rbp-90h]
  __int64 v67; // [rsp+20h] [rbp-90h]
  __int64 v68; // [rsp+20h] [rbp-90h]
  unsigned __int8 v69; // [rsp+28h] [rbp-88h]
  unsigned __int8 v70; // [rsp+28h] [rbp-88h]
  unsigned __int16 v71; // [rsp+28h] [rbp-88h]
  __int64 v72; // [rsp+28h] [rbp-88h]
  unsigned __int8 v73; // [rsp+28h] [rbp-88h]
  unsigned __int8 v74; // [rsp+30h] [rbp-80h]
  __int64 v75; // [rsp+30h] [rbp-80h]
  __int64 v76; // [rsp+30h] [rbp-80h]
  __int64 v77; // [rsp+30h] [rbp-80h]
  __int64 v78; // [rsp+30h] [rbp-80h]
  __int64 v79; // [rsp+30h] [rbp-80h]
  const void **v80; // [rsp+38h] [rbp-78h]
  __int64 v81; // [rsp+38h] [rbp-78h]
  __int64 v82; // [rsp+38h] [rbp-78h]
  __int64 v83; // [rsp+40h] [rbp-70h] BYREF
  int v84; // [rsp+48h] [rbp-68h]
  __int64 v85; // [rsp+50h] [rbp-60h] BYREF
  int v86; // [rsp+58h] [rbp-58h]
  __int64 v87; // [rsp+60h] [rbp-50h] BYREF
  void *v88; // [rsp+68h] [rbp-48h] BYREF
  __int64 v89; // [rsp+70h] [rbp-40h]

  v7 = *(_QWORD *)(a2 + 32);
  v8 = *(_DWORD *)(v7 + 8);
  v9 = *(_QWORD *)v7;
  v10 = *(_QWORD *)v7;
  v11 = *(_QWORD *)(v7 + 8);
  v12 = *(char **)(a2 + 40);
  v13 = *v12;
  v80 = (const void **)*((_QWORD *)v12 + 1);
  v14 = *(unsigned __int16 *)(v9 + 24);
  v15 = v13;
  v74 = v13;
  if ( v14 == 11 || v14 == 33 || (v66 = v9, (unsigned __int8)sub_1D16930(v9)) )
  {
    v16 = *(_QWORD *)(a2 + 72);
    v17 = *(__int64 **)a1;
    v87 = v16;
    if ( v16 )
      sub_1623A60((__int64)&v87, v16, 2);
    *((_QWORD *)&v60 + 1) = v11;
    *(_QWORD *)&v60 = v10;
    LODWORD(v88) = *(_DWORD *)(a2 + 64);
    v18 = sub_1D309E0(v17, 162, (__int64)&v87, v15, v80, 0, a3, a4, *(double *)a5.m128i_i64, v60);
    if ( v87 )
      sub_161E7C0((__int64)&v87, v87);
    return (_QWORD *)v18;
  }
  v20 = sub_1F79A30(
          v66,
          v8,
          a1[24],
          *(_QWORD *)(*(_QWORD *)a1 + 16LL),
          (_BYTE *)(**(_QWORD **)a1 + 792LL),
          0,
          a3,
          a4,
          *(double *)a5.m128i_i64);
  v21 = v66;
  v22 = v74;
  if ( !v20 )
  {
    if ( *(_WORD *)(v66 + 24) == 78 )
    {
      if ( (v23 = *(_QWORD *)(v66 + 48)) != 0 && !*(_QWORD *)(v23 + 32)
        || (v24 = *((_QWORD *)a1 + 1), v25 = *(__int64 (**)())(*(_QWORD *)v24 + 904LL), v25 == sub_1F3CBD0)
        || (v37 = ((__int64 (__fastcall *)(__int64, _QWORD, const void **))v25)(v24, v15, v80),
            v21 = v66,
            v22 = v74,
            !v37) )
      {
        v26 = *(_QWORD *)(*(_QWORD *)(v21 + 32) + 40LL);
        v27 = *(unsigned __int16 *)(v26 + 24);
        if ( v27 == 11 || v27 == 33 )
        {
          v69 = v22;
          v75 = v21;
          v28 = (__int64 *)(*(_QWORD *)(v26 + 88) + 32LL);
          v29 = sub_16982C0();
          if ( (void *)*v28 == v29 )
          {
            sub_169C6E0(&v88, (__int64)v28);
            v31 = v69;
            v30 = v75;
          }
          else
          {
            sub_16986C0(&v88, v28);
            v30 = v75;
            v31 = v69;
          }
          v70 = v31;
          v76 = v30;
          if ( v88 == v29 )
          {
            sub_169C8D0((__int64)&v88, a3, a4, *(double *)a5.m128i_i64);
            v33 = v70;
            v32 = v76;
          }
          else
          {
            sub_1699490((__int64)&v88);
            v32 = v76;
            v33 = v70;
          }
          if ( *((int *)a1 + 4) > 2 )
          {
            v34 = *((_QWORD *)a1 + 1);
            v35 = *(__int64 (**)())(*(_QWORD *)v34 + 328LL);
            if ( v35 != sub_1F3CA70 )
            {
              v79 = v32;
              v73 = v33;
              v59 = ((__int64 (__fastcall *)(__int64, __int64 *, _QWORD, const void **))v35)(v34, &v87, v15, v80);
              v32 = v79;
              v33 = v73;
              if ( v59 )
              {
LABEL_35:
                v38 = *(_QWORD *)(a2 + 72);
                v39 = *(__int64 **)a1;
                v40 = *(_QWORD *)(v32 + 32);
                v71 = *(_WORD *)(v32 + 80);
                v85 = v38;
                if ( v38 )
                {
                  v64 = v40;
                  v67 = v32;
                  sub_1623A60((__int64)&v85, v38, 2);
                  v40 = v64;
                  v32 = v67;
                }
                v68 = v32;
                v86 = *(_DWORD *)(a2 + 64);
                v41 = sub_1D309E0(
                        v39,
                        162,
                        (__int64)&v85,
                        v15,
                        v80,
                        0,
                        a3,
                        a4,
                        *(double *)a5.m128i_i64,
                        *(_OWORD *)(v40 + 40));
                v43 = *(_QWORD *)(a2 + 72);
                v44 = v41;
                v45 = v42;
                v46 = *(_QWORD *)(v68 + 32);
                v83 = v43;
                if ( v43 )
                {
                  v62 = v41;
                  v63 = v42;
                  v65 = v46;
                  sub_1623A60((__int64)&v83, v43, 2);
                  v44 = v62;
                  v45 = v63;
                  v46 = v65;
                }
                *((_QWORD *)&v61 + 1) = v45;
                *(_QWORD *)&v61 = v44;
                v84 = *(_DWORD *)(a2 + 64);
                v18 = (__int64)sub_1D332F0(
                                 v39,
                                 78,
                                 (__int64)&v83,
                                 v15,
                                 v80,
                                 v71,
                                 a3,
                                 a4,
                                 a5,
                                 *(_QWORD *)v46,
                                 *(_QWORD *)(v46 + 8),
                                 v61);
                if ( v83 )
                  sub_161E7C0((__int64)&v83, v83);
                if ( v85 )
                  sub_161E7C0((__int64)&v85, v85);
                if ( v29 == v88 )
                {
                  v81 = v89;
                  if ( v89 )
                  {
                    v47 = v89 + 32LL * *(_QWORD *)(v89 - 8);
                    if ( v89 != v47 )
                    {
                      v72 = v18;
                      do
                      {
                        v47 -= 32;
                        if ( v29 == *(void **)(v47 + 8) )
                        {
                          v48 = *(_QWORD *)(v47 + 16);
                          if ( v48 )
                          {
                            v49 = 32LL * *(_QWORD *)(v48 - 8);
                            v50 = v48 + v49;
                            while ( v48 != v50 )
                            {
                              v50 -= 32;
                              if ( v29 == *(void **)(v50 + 8) )
                              {
                                v51 = *(_QWORD *)(v50 + 16);
                                if ( v51 )
                                {
                                  v52 = v51 + 32LL * *(_QWORD *)(v51 - 8);
                                  if ( v51 != v52 )
                                  {
                                    do
                                    {
                                      v77 = v52 - 32;
                                      sub_127D120((_QWORD *)(v52 - 24));
                                      v52 = v77;
                                    }
                                    while ( v51 != v77 );
                                  }
                                  j_j_j___libc_free_0_0(v51 - 8);
                                }
                              }
                              else
                              {
                                sub_1698460(v50 + 8);
                              }
                            }
                            j_j_j___libc_free_0_0(v48 - 8);
                          }
                        }
                        else
                        {
                          sub_1698460(v47 + 8);
                        }
                      }
                      while ( v81 != v47 );
                      v18 = v72;
                    }
                    j_j_j___libc_free_0_0(v81 - 8);
                  }
                }
                else
                {
                  sub_1698460((__int64)&v88);
                }
                return (_QWORD *)v18;
              }
              v34 = *((_QWORD *)a1 + 1);
            }
            v36 = 1;
            if ( (_BYTE)v33 == 1 || (_BYTE)v33 && (v36 = (unsigned __int8)v33, *(_QWORD *)(v34 + 8 * v33 + 120)) )
            {
              if ( !*(_BYTE *)(v34 + 259 * v36 + 2433) )
                goto LABEL_35;
            }
          }
          if ( v29 == v88 )
          {
            v53 = v89;
            if ( v89 )
            {
              v54 = v89 + 32LL * *(_QWORD *)(v89 - 8);
              if ( v89 != v54 )
              {
                v78 = v89;
                do
                {
                  v54 -= 32;
                  if ( v29 == *(void **)(v54 + 8) )
                  {
                    v55 = *(_QWORD *)(v54 + 16);
                    if ( v55 )
                    {
                      v56 = v55 + 32LL * *(_QWORD *)(v55 - 8);
                      while ( v55 != v56 )
                      {
                        v56 -= 32;
                        if ( v29 == *(void **)(v56 + 8) )
                        {
                          v57 = *(_QWORD *)(v56 + 16);
                          if ( v57 )
                          {
                            v58 = v57 + 32LL * *(_QWORD *)(v57 - 8);
                            if ( v57 != v58 )
                            {
                              do
                              {
                                v82 = v58 - 32;
                                sub_127D120((_QWORD *)(v58 - 24));
                                v58 = v82;
                              }
                              while ( v57 != v82 );
                            }
                            j_j_j___libc_free_0_0(v57 - 8);
                          }
                        }
                        else
                        {
                          sub_1698460(v56 + 8);
                        }
                      }
                      j_j_j___libc_free_0_0(v55 - 8);
                    }
                  }
                  else
                  {
                    sub_1698460(v54 + 8);
                  }
                }
                while ( v78 != v54 );
                v53 = v78;
              }
              j_j_j___libc_free_0_0(v53 - 8);
            }
          }
          else
          {
            sub_1698460((__int64)&v88);
          }
        }
      }
    }
    return 0;
  }
  return sub_1F7A040(v10, v11, *(__int64 **)a1, a1[24], 0, a3, a4, a5);
}
