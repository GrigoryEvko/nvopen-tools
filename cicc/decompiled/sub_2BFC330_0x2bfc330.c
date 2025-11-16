// Function: sub_2BFC330
// Address: 0x2bfc330
//
void __fastcall sub_2BFC330(__int64 **a1, __int64 a2)
{
  __int64 v2; // r15
  __int64 v5; // rax
  __int64 v6; // r8
  __int64 v7; // r9
  const char *v8; // rdx
  unsigned __int64 v9; // rax
  __int64 v10; // rcx
  __int64 v11; // rax
  __int64 *v12; // rdi
  size_t v13; // rsi
  __int64 *v14; // rdx
  __int64 *v15; // r8
  const char *v16; // r14
  __int64 v17; // rax
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  _QWORD **v21; // rsi
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // r9
  const char *v26; // r12
  const char *v27; // r14
  __int64 v28; // rdi
  __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // r8
  __int64 v32; // r9
  __int64 v33; // rax
  __int64 v34; // r12
  bool v35; // zf
  __int64 *v36; // rax
  unsigned int v37; // edx
  int v38; // edx
  unsigned int v39; // esi
  unsigned int v40; // ecx
  __int64 v41; // rdx
  __int64 v42; // r13
  __int64 v43; // r14
  __int64 v44; // rax
  __int64 v45; // r14
  __int64 v46; // rax
  __int64 v47; // rdx
  _QWORD *v48; // rax
  __int64 v49; // r12
  __int64 v50; // rsi
  __int64 v51; // rdx
  __int64 v52; // rax
  __int64 v53; // rsi
  __int64 v54; // rsi
  unsigned __int8 v55; // al
  char v56; // dl
  __int64 v57; // rax
  char v58; // dl
  __int64 v59; // r8
  __int64 (__fastcall *v60)(__int64); // rax
  __int64 v61; // rsi
  __int64 v62; // rax
  __int64 v63; // r8
  __int64 v64; // rdx
  int v65; // eax
  int v66; // eax
  unsigned int v67; // esi
  __int64 v68; // rax
  __int64 v69; // rsi
  __int64 v70; // rsi
  _QWORD **v71; // rsi
  __int64 v72; // rax
  size_t v73; // rdx
  __int64 v74; // [rsp+8h] [rbp-D8h]
  char v75; // [rsp+8h] [rbp-D8h]
  char v76; // [rsp+20h] [rbp-C0h]
  __int64 v77; // [rsp+20h] [rbp-C0h]
  __int64 v78; // [rsp+20h] [rbp-C0h]
  __int64 v79; // [rsp+28h] [rbp-B8h]
  __int64 v80; // [rsp+28h] [rbp-B8h]
  __int64 v81; // [rsp+30h] [rbp-B0h] BYREF
  __int64 *v82; // [rsp+38h] [rbp-A8h] BYREF
  __int64 *v83; // [rsp+40h] [rbp-A0h] BYREF
  size_t n; // [rsp+48h] [rbp-98h]
  _QWORD src[2]; // [rsp+50h] [rbp-90h] BYREF
  const char *v86; // [rsp+60h] [rbp-80h] BYREF
  __int64 v87; // [rsp+68h] [rbp-78h]
  _BYTE v88[112]; // [rsp+70h] [rbp-70h] BYREF

  *(_QWORD *)(a2 + 96) = 0;
  v5 = sub_AA56F0(*(_QWORD *)(a2 + 104));
  *(_QWORD *)(a2 + 1152) = a1;
  *(_QWORD *)(a2 + 112) = v5;
  sub_2C06B20(a2 + 1024);
  v8 = *(const char **)(a2 + 104);
  v9 = *((_QWORD *)v8 + 6) & 0xFFFFFFFFFFFFFFF8LL;
  if ( (const char *)v9 == v8 + 48 )
    goto LABEL_82;
  if ( !v9 )
    BUG();
  v10 = (unsigned int)*(unsigned __int8 *)(v9 - 24) - 30;
  if ( (unsigned int)v10 > 0xA )
LABEL_82:
    BUG();
  if ( *(_QWORD *)(v9 - 56) )
  {
    v10 = *(_QWORD *)(v9 - 48);
    **(_QWORD **)(v9 - 40) = v10;
    if ( v10 )
      *(_QWORD *)(v10 + 16) = *(_QWORD *)(v9 - 40);
  }
  *(_QWORD *)(v9 - 56) = 0;
  v11 = *(_QWORD *)(a2 + 112);
  v86 = v8;
  v79 = a2 + 200;
  v87 = v11 | 4;
  sub_FFB3D0(a2 + 200, (unsigned __int64 *)&v86, 1, v10, v6, v7);
  v88[17] = 1;
  v86 = "Final VPlan";
  v88[16] = 3;
  sub_CA0F50((__int64 *)&v83, (void **)&v86);
  v12 = a1[21];
  if ( v83 == src )
  {
    v73 = n;
    if ( n )
    {
      if ( n == 1 )
        *(_BYTE *)v12 = src[0];
      else
        memcpy(v12, src, n);
      v73 = n;
      v12 = a1[21];
    }
    a1[22] = (__int64 *)v73;
    *((_BYTE *)v12 + v73) = 0;
    v12 = v83;
  }
  else
  {
    v13 = n;
    v14 = (__int64 *)src[0];
    if ( v12 == (__int64 *)(a1 + 23) )
    {
      a1[21] = v83;
      a1[22] = (__int64 *)v13;
      a1[23] = v14;
    }
    else
    {
      v15 = a1[23];
      a1[21] = v83;
      a1[22] = (__int64 *)v13;
      a1[23] = v14;
      if ( v12 )
      {
        v83 = v12;
        src[0] = v15;
        goto LABEL_11;
      }
    }
    v83 = src;
    v12 = src;
  }
LABEL_11:
  n = 0;
  *(_BYTE *)v12 = 0;
  if ( v83 != src )
    j_j___libc_free_0((unsigned __int64)v83);
  v16 = *(const char **)(a2 + 112);
  v17 = sub_AA56F0((__int64)v16);
  v86 = v16;
  v87 = v17 | 4;
  sub_FFB3D0(v79, (unsigned __int64 *)&v86, 1, v18, v19, v20);
  v21 = &v83;
  v83 = *a1;
  v86 = v88;
  v87 = 0x800000000LL;
  sub_2BF66C0((__int64)&v86, (__int64 *)&v83);
  v26 = v86;
  v27 = &v86[8 * (unsigned int)v87];
  if ( v86 != v27 )
  {
    do
    {
      v28 = *((_QWORD *)v27 - 1);
      v27 -= 8;
      v21 = (_QWORD **)a2;
      (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v28 + 16LL))(v28, a2);
    }
    while ( v26 != v27 );
  }
  sub_FFCE90(v79, (__int64)v21, v22, v23, v24, v25);
  sub_FFD870(v79, (__int64)v21, v29, v30, v31, v32);
  sub_FFBC40(v79, (__int64)v21);
  v33 = sub_2BF3F10(a1);
  v34 = v33;
  if ( v33 )
  {
    v81 = sub_2BF0520(v33);
    v35 = (unsigned __int8)sub_2ABFB80(a2 + 120, &v81, &v82) == 0;
    v36 = v82;
    if ( !v35 )
      goto LABEL_24;
    v37 = *(_DWORD *)(a2 + 128);
    ++*(_QWORD *)(a2 + 120);
    v83 = v36;
    v38 = (v37 >> 1) + 1;
    if ( (*(_BYTE *)(a2 + 128) & 1) != 0 )
    {
      v40 = 12;
      v39 = 4;
    }
    else
    {
      v39 = *(_DWORD *)(a2 + 144);
      v40 = 3 * v39;
    }
    if ( 4 * v38 >= v40 )
    {
      v39 *= 2;
    }
    else if ( v39 - (v38 + *(_DWORD *)(a2 + 132)) > v39 >> 3 )
    {
LABEL_21:
      *(_DWORD *)(a2 + 128) = *(_DWORD *)(a2 + 128) & 1 | (2 * v38);
      if ( *v36 != -4096 )
        --*(_DWORD *)(a2 + 132);
      v41 = v81;
      v36[1] = 0;
      *v36 = v41;
LABEL_24:
      v42 = v36[1];
      v43 = sub_2BF04D0(v34);
      v44 = sub_2BF05A0(v43);
      v45 = *(_QWORD *)(v43 + 120);
      v80 = v44;
      if ( v45 == v44 )
        goto LABEL_62;
      while ( 1 )
      {
        if ( !v45 )
          BUG();
        v55 = *(_BYTE *)(v45 - 16);
        if ( v55 == 27 )
          goto LABEL_43;
        if ( (unsigned int)v55 - 33 <= 1 )
        {
          if ( v55 == 33 )
          {
            v71 = (_QWORD **)(*(_QWORD *)(v45 - 8) & 0xFFFFFFFFFFFFFFF8LL);
            if ( (*(_QWORD *)(v45 - 8) & 4) != 0 )
              v71 = (_QWORD **)**v71;
            v47 = sub_2BFB640(a2, (__int64)v71, 0);
          }
          else
          {
            v46 = sub_2BFB640(a2, v45 + 72, 0);
            v47 = *(_QWORD *)(v46 - 32LL * (*(_DWORD *)(v46 + 4) & 0x7FFFFFF));
          }
          *(_QWORD *)(*(_QWORD *)(v47 - 8) + 32LL * *(unsigned int *)(v47 + 72) + 8) = v42;
          v48 = (_QWORD *)(*(_QWORD *)(v42 + 48) & 0xFFFFFFFFFFFFFFF8LL);
          if ( v48 == (_QWORD *)(v42 + 48) )
            goto LABEL_83;
          if ( !v48 )
            BUG();
          if ( (unsigned int)*((unsigned __int8 *)v48 - 24) - 30 > 0xA )
LABEL_83:
            BUG();
          LOWORD(v2) = 0;
          v49 = *(_QWORD *)(*(_QWORD *)(v47 - 8) + 32LL);
          sub_B444E0((_QWORD *)v49, *v48 & 0xFFFFFFFFFFFFFFF8LL, v2);
          if ( *(_BYTE *)(v45 - 16) == 33 )
          {
            v50 = v45 + 72;
            if ( *(_DWORD *)(v45 + 32) == 5 )
              v50 = *(_QWORD *)(*(_QWORD *)(v45 + 24) + 32LL);
            v51 = sub_2BFB640(a2, v50, 0);
            if ( (*(_BYTE *)(v49 + 7) & 0x40) != 0 )
              v52 = *(_QWORD *)(v49 - 8);
            else
              v52 = v49 - 32LL * (*(_DWORD *)(v49 + 4) & 0x7FFFFFF);
            if ( *(_QWORD *)v52 )
            {
              v53 = *(_QWORD *)(v52 + 8);
              **(_QWORD **)(v52 + 16) = v53;
              if ( v53 )
                *(_QWORD *)(v53 + 16) = *(_QWORD *)(v52 + 16);
            }
            *(_QWORD *)v52 = v51;
            if ( v51 )
            {
              v54 = *(_QWORD *)(v51 + 16);
              *(_QWORD *)(v52 + 8) = v54;
              if ( v54 )
                *(_QWORD *)(v54 + 16) = v52 + 8;
              *(_QWORD *)(v52 + 16) = v51 + 16;
              *(_QWORD *)(v51 + 16) = v52;
            }
          }
LABEL_43:
          v45 = *(_QWORD *)(v45 + 8);
          if ( v45 == v80 )
            goto LABEL_62;
        }
        else
        {
          v56 = 1;
          if ( v55 != 35 )
          {
            v56 = 0;
            if ( v55 == 36 )
              v56 = *(_BYTE *)(v45 + 136);
          }
          v76 = v56;
          v57 = sub_2BFB640(a2, v45 + 72, v56);
          v58 = v76;
          v59 = v57;
          v60 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)(v45 - 24) + 40LL);
          if ( v60 == sub_2AA7530 )
          {
            v61 = *(_QWORD *)(*(_QWORD *)(v45 + 24) + 8LL);
          }
          else
          {
            v75 = v76;
            v78 = v59;
            v72 = v60(v45 - 24);
            v58 = v75;
            v59 = v78;
            v61 = v72;
          }
          v77 = v59;
          v62 = sub_2BFB640(a2, v61, v58);
          v63 = v77;
          v64 = v62;
          v65 = *(_DWORD *)(v77 + 4) & 0x7FFFFFF;
          if ( v65 == *(_DWORD *)(v77 + 72) )
          {
            v74 = v64;
            sub_B48D90(v77);
            v63 = v77;
            v64 = v74;
            v65 = *(_DWORD *)(v77 + 4) & 0x7FFFFFF;
          }
          v66 = (v65 + 1) & 0x7FFFFFF;
          v67 = v66 | *(_DWORD *)(v63 + 4) & 0xF8000000;
          v68 = *(_QWORD *)(v63 - 8) + 32LL * (unsigned int)(v66 - 1);
          *(_DWORD *)(v63 + 4) = v67;
          if ( *(_QWORD *)v68 )
          {
            v69 = *(_QWORD *)(v68 + 8);
            **(_QWORD **)(v68 + 16) = v69;
            if ( v69 )
              *(_QWORD *)(v69 + 16) = *(_QWORD *)(v68 + 16);
          }
          *(_QWORD *)v68 = v64;
          if ( v64 )
          {
            v70 = *(_QWORD *)(v64 + 16);
            *(_QWORD *)(v68 + 8) = v70;
            if ( v70 )
              *(_QWORD *)(v70 + 16) = v68 + 8;
            *(_QWORD *)(v68 + 16) = v64 + 16;
            *(_QWORD *)(v64 + 16) = v68;
          }
          *(_QWORD *)(*(_QWORD *)(v63 - 8)
                    + 32LL * *(unsigned int *)(v63 + 72)
                    + 8LL * ((*(_DWORD *)(v63 + 4) & 0x7FFFFFFu) - 1)) = v42;
          v45 = *(_QWORD *)(v45 + 8);
          if ( v45 == v80 )
            goto LABEL_62;
        }
      }
    }
    sub_2ACA3E0(a2 + 120, v39);
    sub_2ABFB80(a2 + 120, &v81, &v83);
    v36 = v83;
    v38 = (*(_DWORD *)(a2 + 128) >> 1) + 1;
    goto LABEL_21;
  }
LABEL_62:
  if ( v86 != v88 )
    _libc_free((unsigned __int64)v86);
}
