// Function: sub_2EDA920
// Address: 0x2eda920
//
__int64 __fastcall sub_2EDA920(__int64 *a1, __int64 a2, __int64 a3, _BYTE *a4, __int64 a5)
{
  __int64 v5; // rbx
  __int64 v6; // r13
  unsigned int v8; // r12d
  char v9; // dl
  __int64 v10; // rdi
  __int64 (*v11)(); // rax
  __int64 v13; // rdi
  __int64 (*v14)(); // rax
  __int64 v15; // rdi
  __int64 (*v16)(); // rax
  char v17; // cl
  __int64 v18; // rsi
  __int64 v19; // rax
  int v20; // edx
  int v21; // edi
  _QWORD *v22; // rax
  __int64 v23; // r8
  __int64 v24; // rdx
  __int64 **v25; // rax
  __int64 *v26; // r15
  __int64 v27; // r13
  __int64 v28; // r11
  unsigned __int64 v29; // r8
  void *v30; // r9
  size_t v31; // r15
  __int64 v32; // rcx
  __int64 v33; // rax
  __int64 v34; // rsi
  unsigned int v35; // edx
  __int64 v36; // rdx
  __int64 **v37; // r15
  __int64 v38; // rdx
  __int64 **v39; // rbx
  __int64 *v40; // r13
  __int64 v41; // r8
  __int64 v42; // r9
  __int64 v43; // rax
  __int64 v44; // r10
  unsigned __int64 v45; // rdx
  __int64 *v46; // r11
  __int64 v47; // rax
  __int64 *v48; // r9
  __int64 v49; // r15
  __int64 *v50; // r14
  __int64 *v51; // rax
  __int64 *v52; // r11
  unsigned __int64 v53; // r10
  __int64 v54; // r8
  __int64 v55; // rax
  int v56; // eax
  char *v57; // rdi
  _BYTE *v58; // rdi
  size_t v59; // rdx
  int v60; // r9d
  __int64 v61; // rax
  __int64 v62; // [rsp+0h] [rbp-140h]
  __int64 v63; // [rsp+8h] [rbp-138h]
  __int64 *v64; // [rsp+8h] [rbp-138h]
  __int64 *src; // [rsp+10h] [rbp-130h]
  void *srca; // [rsp+10h] [rbp-130h]
  __int64 v67; // [rsp+18h] [rbp-128h]
  __int64 *v68; // [rsp+18h] [rbp-128h]
  int v69; // [rsp+18h] [rbp-128h]
  int v70; // [rsp+18h] [rbp-128h]
  unsigned int v71; // [rsp+18h] [rbp-128h]
  __int64 v72; // [rsp+20h] [rbp-120h]
  __int64 *v73; // [rsp+20h] [rbp-120h]
  unsigned int v74; // [rsp+20h] [rbp-120h]
  unsigned int v75; // [rsp+2Ch] [rbp-114h]
  __int64 v78; // [rsp+40h] [rbp-100h]
  __int64 **v79; // [rsp+40h] [rbp-100h]
  __int64 v82; // [rsp+58h] [rbp-E8h]
  __int64 v83; // [rsp+68h] [rbp-D8h] BYREF
  char v84[16]; // [rsp+70h] [rbp-D0h] BYREF
  __int64 v85; // [rsp+80h] [rbp-C0h]
  void *v86; // [rsp+A0h] [rbp-A0h] BYREF
  __int64 v87; // [rsp+A8h] [rbp-98h]
  _BYTE v88[32]; // [rsp+B0h] [rbp-90h] BYREF
  __int64 v89; // [rsp+D0h] [rbp-70h] BYREF
  _BYTE *v90; // [rsp+D8h] [rbp-68h] BYREF
  __int64 v91; // [rsp+E0h] [rbp-60h]
  _BYTE dest[88]; // [rsp+E8h] [rbp-58h] BYREF

  v5 = *(_QWORD *)(a2 + 32);
  v6 = v5 + 40LL * (*(_DWORD *)(a2 + 40) & 0xFFFFFF);
  if ( v5 == v6 )
    return 0;
  v82 = 0;
  v75 = ((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4);
  do
  {
    while ( 1 )
    {
      if ( *(_BYTE *)v5 )
        goto LABEL_5;
      v8 = *(_DWORD *)(v5 + 8);
      if ( !v8 )
        goto LABEL_5;
      v9 = *(_BYTE *)(v5 + 3) & 0x10;
      if ( v8 - 1 > 0x3FFFFFFE )
      {
        if ( v9 )
        {
          v10 = a1[1];
          v11 = *(__int64 (**)())(*(_QWORD *)v10 + 1008LL);
          if ( v11 == sub_2ED1200 )
          {
            if ( !v82 )
              goto LABEL_30;
          }
          else
          {
            if ( !((unsigned __int8 (__fastcall *)(__int64, unsigned __int64))v11)(
                    v10,
                    *(_QWORD *)(*(_QWORD *)(a1[3] + 56) + 16LL * (v8 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL) )
              return 0;
            if ( !v82 )
            {
LABEL_30:
              v83 = a3;
              v17 = *(_BYTE *)(a5 + 8) & 1;
              if ( !v17 )
              {
                v18 = *(_QWORD *)(a5 + 16);
                v19 = *(unsigned int *)(a5 + 24);
                if ( (_DWORD)v19 )
                {
                  v20 = v19 - 1;
                  goto LABEL_33;
                }
LABEL_71:
                v55 = 56 * v19;
                goto LABEL_72;
              }
              v18 = a5 + 16;
              v20 = 3;
LABEL_33:
              v21 = v20 & v75;
              v22 = (_QWORD *)(v18 + 56LL * (v20 & v75));
              v23 = *v22;
              if ( a3 != *v22 )
              {
                v56 = 1;
                while ( v23 != -4096 )
                {
                  v60 = v56 + 1;
                  v61 = v20 & (unsigned int)(v21 + v56);
                  v21 = v61;
                  v22 = (_QWORD *)(v18 + 56 * v61);
                  v23 = *v22;
                  if ( a3 == *v22 )
                    goto LABEL_34;
                  v56 = v60;
                }
                if ( !v17 )
                {
                  v19 = *(unsigned int *)(a5 + 24);
                  goto LABEL_71;
                }
                v55 = 224;
LABEL_72:
                v22 = (_QWORD *)(v18 + v55);
              }
LABEL_34:
              v24 = 224;
              if ( !v17 )
                v24 = 56LL * *(unsigned int *)(a5 + 24);
              if ( v22 != (_QWORD *)(v18 + v24) )
              {
                v25 = (__int64 **)(v22 + 1);
                goto LABEL_38;
              }
              v29 = *(unsigned int *)(a3 + 120);
              v30 = *(void **)(a3 + 112);
              v31 = 8 * v29;
              v86 = v88;
              v87 = 0x400000000LL;
              if ( v29 > 4 )
              {
                srca = v30;
                v69 = v29;
                sub_C8D5F0((__int64)&v86, v88, v29, 8u, v29, (__int64)v30);
                LODWORD(v29) = v69;
                v30 = srca;
                v57 = (char *)v86 + 8 * (unsigned int)v87;
              }
              else
              {
                if ( !v31 )
                {
LABEL_49:
                  v32 = a1[4];
                  LODWORD(v87) = v29 + v31;
                  v33 = (unsigned int)(v29 + v31);
                  if ( v83 )
                  {
                    v34 = (unsigned int)(*(_DWORD *)(v83 + 24) + 1);
                    v35 = *(_DWORD *)(v83 + 24) + 1;
                  }
                  else
                  {
                    v34 = 0;
                    v35 = 0;
                  }
                  if ( v35 >= *(_DWORD *)(v32 + 32) )
                    BUG();
                  v36 = *(_QWORD *)(*(_QWORD *)(v32 + 24) + 8 * v34);
                  v37 = *(__int64 ***)(v36 + 24);
                  v38 = *(unsigned int *)(v36 + 32);
                  if ( v37 != &v37[v38] )
                  {
                    v63 = v5;
                    v39 = &v37[v38];
                    v67 = v6;
                    do
                    {
                      v40 = *v37;
                      if ( *(_QWORD *)(*v37)[1] == *(_QWORD *)(a2 + 24) && !sub_2E322C0(v83, *v40) )
                      {
                        v43 = (unsigned int)v87;
                        v44 = *v40;
                        v45 = (unsigned int)v87 + 1LL;
                        if ( v45 > HIDWORD(v87) )
                        {
                          v62 = *v40;
                          sub_C8D5F0((__int64)&v86, v88, v45, 8u, v41, v42);
                          v43 = (unsigned int)v87;
                          v44 = v62;
                        }
                        *((_QWORD *)v86 + v43) = v44;
                        LODWORD(v87) = v87 + 1;
                      }
                      ++v37;
                    }
                    while ( v39 != v37 );
                    v6 = v67;
                    v5 = v63;
                    v33 = (unsigned int)v87;
                  }
                  v46 = (__int64 *)v86;
                  v47 = 8 * v33;
                  v48 = &v83;
                  v73 = (__int64 *)((char *)v86 + v47);
                  if ( v47 )
                  {
                    v49 = v47 >> 3;
                    src = a1;
                    v50 = (__int64 *)v86;
                    do
                    {
                      v64 = v48;
                      v51 = (__int64 *)sub_2207800(8 * v49);
                      v48 = v64;
                      if ( v51 )
                      {
                        v52 = v50;
                        a1 = src;
                        v68 = v51;
                        sub_2ED40C0(v52, v73, v51, v49, src, v64);
                        v53 = (unsigned __int64)v68;
                        goto LABEL_65;
                      }
                      v49 >>= 1;
                    }
                    while ( v49 );
                    v46 = v50;
                    a1 = src;
                  }
                  sub_2ED34B0(v46, v73, (__int64)a1, v48);
                  v53 = 0;
LABEL_65:
                  j_j___libc_free_0(v53);
                  v54 = (unsigned int)v87;
                  v90 = dest;
                  v89 = v83;
                  v91 = 0x400000000LL;
                  if ( (_DWORD)v87 )
                  {
                    v58 = dest;
                    v59 = 8LL * (unsigned int)v87;
                    if ( (unsigned int)v87 <= 4
                      || (v71 = v87,
                          sub_C8D5F0((__int64)&v90, dest, (unsigned int)v87, 8u, (unsigned int)v87, (unsigned int)v87),
                          v58 = v90,
                          v54 = v71,
                          (v59 = 8LL * (unsigned int)v87) != 0) )
                    {
                      v74 = v54;
                      memcpy(v58, v86, v59);
                      v54 = v74;
                    }
                    LODWORD(v91) = v54;
                  }
                  sub_2EDA4D0((__int64)v84, a5, &v89, (__int64)&v90, v54, (__int64)&v89);
                  if ( v90 != dest )
                    _libc_free((unsigned __int64)v90);
                  v25 = (__int64 **)(v85 + 8);
                  if ( v86 != v88 )
                  {
                    v79 = (__int64 **)(v85 + 8);
                    _libc_free((unsigned __int64)v86);
                    v25 = v79;
                  }
LABEL_38:
                  v26 = *v25;
                  v78 = (__int64)&(*v25)[*((unsigned int *)v25 + 2)];
                  if ( *v25 == (__int64 *)v78 )
                    return 0;
                  v72 = v6;
                  while ( 1 )
                  {
                    v27 = *v26;
                    LOBYTE(v89) = 0;
                    if ( (unsigned __int8)sub_2ED2890((__int64)a1, v8, v27, a3, a4, &v89) )
                      break;
                    if ( (_BYTE)v89 )
                      return v82;
                    if ( (__int64 *)v78 == ++v26 )
                      return 0;
                  }
                  v28 = v27;
                  v6 = v72;
                  if ( !v28 )
                    return 0;
                  v82 = v28;
                  if ( !(unsigned __int8)sub_2EDB1A0(a1, v8, a2, a3, v28, a5) )
                    return 0;
                  goto LABEL_5;
                }
                v57 = v88;
              }
              v70 = v29;
              memcpy(v57, v30, v31);
              LODWORD(v31) = v87;
              LODWORD(v29) = v70;
              goto LABEL_49;
            }
          }
          LOBYTE(v89) = 0;
          if ( !(unsigned __int8)sub_2ED2890((__int64)a1, v8, v82, a3, a4, &v89) )
            return 0;
        }
        goto LABEL_5;
      }
      if ( !v9 )
      {
        if ( !(unsigned __int8)sub_2EBF3A0((_QWORD *)a1[3], v8) )
        {
          v13 = a1[1];
          v14 = *(__int64 (**)())(*(_QWORD *)v13 + 32LL);
          if ( v14 == sub_2E4EE60 || !((unsigned __int8 (__fastcall *)(__int64, __int64))v14)(v13, v5) )
            return 0;
        }
        goto LABEL_5;
      }
      if ( (((*(_BYTE *)(v5 + 3) & 0x10) != 0) & (*(_BYTE *)(v5 + 3) >> 6)) == 0 )
        break;
LABEL_5:
      v5 += 40;
      if ( v6 == v5 )
        goto LABEL_21;
    }
    if ( !(unsigned __int8)sub_2EBF3A0((_QWORD *)a1[3], v8) )
      return 0;
    v5 += 40;
  }
  while ( v6 != v5 );
LABEL_21:
  if ( a3 == v82 )
    return 0;
  if ( !v82 )
    return 0;
  if ( *(_BYTE *)(v82 + 216) )
    return 0;
  if ( *(_BYTE *)(v82 + 262) )
    return 0;
  v15 = a1[1];
  v16 = *(__int64 (**)())(*(_QWORD *)v15 + 40LL);
  if ( v16 != sub_2ED11A0
    && !((unsigned __int8 (__fastcall *)(__int64, __int64, __int64, __int64))v16)(v15, a2, v82, a1[6]) )
  {
    return 0;
  }
  return v82;
}
