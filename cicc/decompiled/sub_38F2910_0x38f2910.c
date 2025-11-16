// Function: sub_38F2910
// Address: 0x38f2910
//
__int64 __fastcall sub_38F2910(__int64 a1, unsigned __int64 a2)
{
  _DWORD *v2; // rax
  unsigned int v3; // r12d
  _DWORD *v5; // rax
  _DWORD *v6; // rax
  __int64 v7; // rdx
  unsigned int v8; // ecx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rax
  bool v12; // cc
  _QWORD *v13; // rax
  __int64 v14; // rbx
  __int64 v15; // r8
  __int64 v16; // r9
  _DWORD *v17; // rax
  __int64 v19; // r10
  _BYTE *v20; // r10
  char v21; // dl
  __int64 v22; // rdi
  void (__fastcall *v23)(__int64, unsigned __int64, __int64, void **, __int64, _BYTE *, const char **); // rax
  __int64 v24; // rax
  __int64 v25; // rcx
  __int64 v26; // rax
  __int64 v27; // rdx
  const char *v28; // rax
  __int64 v29; // rax
  __int64 v30; // r14
  __int64 v31; // rsi
  __int64 v32; // rax
  __int64 v33; // rdx
  char v34; // al
  _BYTE *v35; // rax
  __int64 v36; // rsi
  void (__fastcall *v37)(__int64 *, __int64, _QWORD, unsigned __int64, __int64, _BYTE *, void **, __int64, const char **, _QWORD); // rax
  __int64 v38; // rax
  __int64 v39; // rsi
  __int64 v40; // rdx
  __int64 v41; // rcx
  __int64 v42; // r8
  _BYTE *v43; // rax
  unsigned __int64 v44; // rsi
  unsigned __int64 v45; // rdx
  int v46; // ecx
  unsigned __int64 v47; // rdi
  unsigned __int64 v48; // rdi
  unsigned __int64 v49; // rdi
  __int64 v50; // rax
  unsigned __int64 v51; // rdi
  __int64 v52; // rbx
  __int64 v53; // r10
  unsigned __int64 v54; // r11
  __int64 v55; // rax
  unsigned __int64 *v56; // rbx
  unsigned int v57; // r13d
  unsigned __int64 *v58; // r12
  unsigned __int64 v59; // r11
  __int64 v60; // rax
  unsigned __int64 *v61; // rbx
  unsigned int v62; // r13d
  unsigned __int64 *v63; // r12
  _QWORD *v64; // [rsp+0h] [rbp-190h]
  __int64 v65; // [rsp+8h] [rbp-188h]
  __int64 v66; // [rsp+18h] [rbp-178h]
  char *v67; // [rsp+28h] [rbp-168h]
  size_t v68; // [rsp+30h] [rbp-160h]
  char v69; // [rsp+46h] [rbp-14Ah]
  char v70; // [rsp+47h] [rbp-149h]
  __int64 v71; // [rsp+48h] [rbp-148h]
  unsigned __int64 v72; // [rsp+50h] [rbp-140h]
  void **v73; // [rsp+60h] [rbp-130h]
  __int64 v74; // [rsp+68h] [rbp-128h]
  __int64 v76; // [rsp+80h] [rbp-110h]
  __int64 v77; // [rsp+88h] [rbp-108h]
  _BYTE *v78; // [rsp+88h] [rbp-108h]
  const char *v79; // [rsp+98h] [rbp-F8h] BYREF
  unsigned __int64 v80; // [rsp+A0h] [rbp-F0h] BYREF
  __int64 v81; // [rsp+A8h] [rbp-E8h] BYREF
  __int64 v82; // [rsp+B0h] [rbp-E0h] BYREF
  __int64 v83; // [rsp+B8h] [rbp-D8h]
  _QWORD v84[2]; // [rsp+C0h] [rbp-D0h] BYREF
  __int16 v85; // [rsp+D0h] [rbp-C0h]
  void **v86; // [rsp+E0h] [rbp-B0h] BYREF
  __int64 v87; // [rsp+E8h] [rbp-A8h]
  _BYTE v88[16]; // [rsp+F0h] [rbp-A0h] BYREF
  void **v89; // [rsp+100h] [rbp-90h] BYREF
  __int64 v90; // [rsp+108h] [rbp-88h]
  _BYTE v91[16]; // [rsp+110h] [rbp-80h] BYREF
  void *src; // [rsp+120h] [rbp-70h] BYREF
  size_t n; // [rsp+128h] [rbp-68h]
  _BYTE v94[16]; // [rsp+130h] [rbp-60h] BYREF
  const char *v95; // [rsp+140h] [rbp-50h] BYREF
  size_t v96; // [rsp+148h] [rbp-48h]
  __int64 v97; // [rsp+150h] [rbp-40h] BYREF

  v76 = -1;
  if ( **(_DWORD **)(a1 + 152) == 4 )
  {
    v11 = sub_3909460(a1);
    v12 = *(_DWORD *)(v11 + 32) <= 0x40u;
    v13 = *(_QWORD **)(v11 + 24);
    if ( !v12 )
      v13 = (_QWORD *)*v13;
    v14 = (__int64)v13;
    v76 = (__int64)v13;
    sub_38EB180(a1);
    if ( v14 < 0 )
    {
      v95 = "negative file number";
      LOWORD(v97) = 259;
      return (unsigned int)sub_3909CF0(a1, &v95, 0, 0, v15, v16);
    }
  }
  v95 = "unexpected token in '.file' directive";
  v86 = (void **)v88;
  v87 = 0;
  v88[0] = 0;
  LOWORD(v97) = 259;
  v2 = (_DWORD *)sub_3909460(a1);
  if ( !(unsigned __int8)sub_3909CB0(a1, *v2 != 3, &v95) )
  {
    v3 = sub_38ECF20(a1, (unsigned __int64 *)&v86);
    if ( !(_BYTE)v3 )
    {
      v91[0] = 0;
      v89 = (void **)v91;
      v5 = *(_DWORD **)(a1 + 152);
      v90 = 0;
      v72 = 0;
      v71 = 0;
      if ( *v5 == 3 )
      {
        v95 = "explicit path specified, but no file number";
        LOWORD(v97) = 259;
        if ( (unsigned __int8)sub_3909CB0(a1, v76 == -1, &v95)
          || (unsigned __int8)sub_38ECF20(a1, (unsigned __int64 *)&v89) )
        {
          v3 = 1;
LABEL_19:
          if ( v89 != (void **)v91 )
            j_j___libc_free_0((unsigned __int64)v89);
          goto LABEL_4;
        }
        v73 = v89;
        v74 = v90;
        v72 = (unsigned __int64)v86;
        v71 = v87;
      }
      else
      {
        v73 = v86;
        v74 = v87;
      }
      v94[0] = 0;
      n = 0;
      v69 = 0;
      v70 = 0;
      src = v94;
      while ( !(unsigned __int8)sub_3909EB0(a1, 9) )
      {
        v82 = 0;
        v83 = 0;
        v95 = "unexpected token in '.file' directive";
        LOWORD(v97) = 259;
        v6 = (_DWORD *)sub_3909460(a1);
        if ( (unsigned __int8)sub_3909CB0(a1, *v6 != 2, &v95) || (unsigned __int8)sub_38F0EE0(a1, &v82, v7, v8) )
          goto LABEL_40;
        if ( v83 == 3 )
        {
          if ( *(_WORD *)v82 != 25709 || *(_BYTE *)(v82 + 2) != 53 )
          {
LABEL_16:
            v95 = "unexpected token in '.file' directive";
            LOWORD(v97) = 259;
            v3 = sub_3909CF0(a1, &v95, 0, 0, v9, v10);
            goto LABEL_17;
          }
          LOWORD(v97) = 259;
          v95 = "MD5 checksum specified, but no file number";
          if ( (unsigned __int8)sub_3909CB0(a1, v76 == -1, &v95) || (unsigned __int8)sub_38EE320(a1, &v79, &v80) )
          {
LABEL_40:
            v3 = 1;
            goto LABEL_17;
          }
          v70 = 1;
        }
        else
        {
          if ( v83 != 6 || *(_DWORD *)v82 != 1920298867 || *(_WORD *)(v82 + 4) != 25955 )
            goto LABEL_16;
          v84[0] = "source specified, but no file number";
          v85 = 259;
          if ( (unsigned __int8)sub_3909CB0(a1, v76 == -1, v84) )
            goto LABEL_40;
          v95 = "unexpected token in '.file' directive";
          LOWORD(v97) = 259;
          v17 = (_DWORD *)sub_3909460(a1);
          if ( (unsigned __int8)sub_3909CB0(a1, *v17 != 3, &v95)
            || (unsigned __int8)sub_38ECF20(a1, (unsigned __int64 *)&src) )
          {
            goto LABEL_40;
          }
          v69 = 1;
        }
      }
      if ( v76 == -1 )
      {
        (*(void (__fastcall **)(_QWORD, void **, __int64))(**(_QWORD **)(a1 + 328) + 552LL))(
          *(_QWORD *)(a1 + 328),
          v73,
          v74);
LABEL_17:
        if ( src != v94 )
          j_j___libc_free_0((unsigned __int64)src);
        goto LABEL_19;
      }
      v19 = *(_QWORD *)(a1 + 320);
      if ( !*(_BYTE *)(v19 + 1041) )
      {
LABEL_43:
        v20 = 0;
        if ( v70 )
        {
          v43 = (_BYTE *)sub_145CBF0((__int64 *)(*(_QWORD *)(a1 + 320) + 48LL), 16, 1);
          v44 = (unsigned __int64)v79;
          v45 = v80;
          v46 = 56;
          v20 = v43;
          do
          {
            *v43++ = v44 >> v46;
            v47 = v45 >> v46;
            v46 -= 8;
            v43[7] = v47;
          }
          while ( v46 != -8 );
        }
        if ( v69 )
        {
          v78 = v20;
          v67 = (char *)sub_145CBF0((__int64 *)(*(_QWORD *)(a1 + 320) + 48LL), (unsigned int)n, 8);
          v68 = n;
          memcpy(v67, src, n);
          v20 = v78;
          if ( !v76 )
          {
            v21 = 1;
            goto LABEL_46;
          }
          v36 = *(_QWORD *)(a1 + 328);
          v37 = *(void (__fastcall **)(__int64 *, __int64, _QWORD, unsigned __int64, __int64, _BYTE *, void **, __int64, const char **, _QWORD))(*(_QWORD *)v36 + 568LL);
          v95 = v67;
          LOBYTE(v97) = 1;
          v96 = v68;
        }
        else
        {
          v21 = 0;
          if ( !v76 )
          {
LABEL_46:
            if ( *(_WORD *)(*(_QWORD *)(a1 + 320) + 1160LL) <= 4u )
            {
              BYTE1(v97) = 1;
              v28 = "file 0 not supported prior to DWARF-5";
LABEL_59:
              v95 = v28;
              LOBYTE(v97) = 3;
              v3 = sub_38E4170((_QWORD *)a1, a2, (__int64)&v95, 0, 0);
              goto LABEL_17;
            }
            v22 = *(_QWORD *)(a1 + 328);
            v23 = *(void (__fastcall **)(__int64, unsigned __int64, __int64, void **, __int64, _BYTE *, const char **))(*(_QWORD *)v22 + 576LL);
            LOBYTE(v97) = v21;
            if ( v21 )
            {
              v95 = v67;
              v96 = v68;
            }
            v23(v22, v72, v71, v73, v74, v20, &v95);
LABEL_50:
            if ( *(_BYTE *)(a1 + 846) )
              goto LABEL_17;
            v24 = *(_QWORD *)(a1 + 320);
            v25 = v24 + 984;
            v26 = *(_QWORD *)(v24 + 992);
            if ( v26 )
            {
              do
              {
                v27 = v26;
                v26 = *(_QWORD *)(v26 + 16);
              }
              while ( v26 );
              if ( v27 != v25 && !*(_DWORD *)(v27 + 32) )
                v25 = v27;
            }
            v3 = 0;
            if ( !*(_DWORD *)(v25 + 168) || *(_BYTE *)(v25 + 529) == *(_BYTE *)(v25 + 530) )
              goto LABEL_17;
            *(_BYTE *)(a1 + 846) = 1;
            v28 = "inconsistent use of MD5 checksums";
            BYTE1(v97) = 1;
            goto LABEL_59;
          }
          v36 = *(_QWORD *)(a1 + 328);
          v37 = *(void (__fastcall **)(__int64 *, __int64, _QWORD, unsigned __int64, __int64, _BYTE *, void **, __int64, const char **, _QWORD))(*(_QWORD *)v36 + 568LL);
          LOBYTE(v97) = 0;
        }
        v37(&v82, v36, (unsigned int)v76, v72, v71, v20, v73, v74, &v95, 0);
        if ( (v83 & 1) != 0 )
        {
          LOBYTE(v83) = v83 & 0xFD;
          v38 = v82;
          v82 = 0;
          v81 = v38 | 1;
          sub_12BF440((__int64)&v95, &v81);
          v39 = a2;
          v85 = 260;
          v84[0] = &v95;
          v3 = sub_3909790(a1, a2, v84, 0, 0);
          if ( v95 != (const char *)&v97 )
          {
            v39 = v97 + 1;
            j_j___libc_free_0((unsigned __int64)v95);
          }
          if ( (v81 & 1) != 0 || (v81 & 0xFFFFFFFFFFFFFFFELL) != 0 )
            sub_16BCAE0(&v81, v39, v40);
          if ( (v83 & 2) != 0 )
            sub_14F4240(&v82, v39, v40);
          if ( (v83 & 1) != 0 && v82 )
            (*(void (__fastcall **)(__int64, __int64, __int64, __int64, __int64, __int64 *))(*(_QWORD *)v82 + 8LL))(
              v82,
              v39,
              v40,
              v41,
              v42,
              &v81);
          goto LABEL_17;
        }
        goto LABEL_50;
      }
      v29 = *(_QWORD *)(v19 + 992);
      if ( v29 )
      {
        do
        {
          v30 = v29;
          v29 = *(_QWORD *)(v29 + 16);
        }
        while ( v29 );
        if ( v19 + 984 != v30 && !*(_DWORD *)(v30 + 32) )
        {
LABEL_69:
          v35 = *(_BYTE **)(v30 + 456);
          *(_QWORD *)(v30 + 464) = 0;
          *v35 = 0;
          *(_WORD *)(v30 + 528) = 256;
          *(_BYTE *)(v30 + 530) = 0;
          *(_BYTE *)(*(_QWORD *)(a1 + 320) + 1041LL) = 0;
          goto LABEL_43;
        }
      }
      else
      {
        v30 = v19 + 984;
      }
      v64 = (_QWORD *)(v19 + 984);
      v65 = *(_QWORD *)(a1 + 320);
      v31 = v30;
      v30 = sub_22077B0(0x250u);
      *(_DWORD *)(v30 + 32) = 0;
      memset((void *)(v30 + 40), 0, 0x228u);
      *(_BYTE *)(v30 + 529) = 1;
      *(_QWORD *)(v30 + 48) = v30 + 64;
      *(_QWORD *)(v30 + 160) = v30 + 176;
      *(_QWORD *)(v30 + 56) = 0x300000000LL;
      *(_QWORD *)(v30 + 168) = 0x300000000LL;
      *(_QWORD *)(v30 + 408) = 0x1000000000LL;
      *(_QWORD *)(v30 + 424) = v30 + 440;
      *(_QWORD *)(v30 + 456) = v30 + 472;
      v32 = sub_38C3D00((_QWORD *)(v65 + 976), v31, (unsigned int *)(v30 + 32));
      v77 = v32;
      if ( v33 )
      {
        v34 = v32 != 0 || v64 == (_QWORD *)v33;
        if ( v77 == 0 && v64 != (_QWORD *)v33 )
          v34 = *(_DWORD *)(v33 + 32) != 0;
        sub_220F040(v34, v30, (_QWORD *)v33, v64);
        ++*(_QWORD *)(v65 + 1016);
      }
      else
      {
        j___libc_free_0(0);
        v48 = *(_QWORD *)(v30 + 456);
        if ( v30 + 472 != v48 )
          j_j___libc_free_0(v48);
        v49 = *(_QWORD *)(v30 + 424);
        if ( v30 + 440 != v49 )
          j_j___libc_free_0(v49);
        if ( *(_DWORD *)(v30 + 404) && (v50 = *(unsigned int *)(v30 + 400), (_DWORD)v50) )
        {
          v51 = *(_QWORD *)(v30 + 392);
          v66 = 8 * v50;
          v52 = 0;
          do
          {
            v53 = *(_QWORD *)(v51 + v52);
            if ( v53 != -8 && v53 )
            {
              _libc_free(*(_QWORD *)(v51 + v52));
              v51 = *(_QWORD *)(v30 + 392);
            }
            v52 += 8;
          }
          while ( v66 != v52 );
        }
        else
        {
          v51 = *(_QWORD *)(v30 + 392);
        }
        _libc_free(v51);
        v54 = *(_QWORD *)(v30 + 160);
        v55 = 72LL * *(unsigned int *)(v30 + 168);
        if ( v54 != v54 + v55 )
        {
          v56 = (unsigned __int64 *)(v54 + v55);
          v57 = v3;
          v58 = *(unsigned __int64 **)(v30 + 160);
          do
          {
            v56 -= 9;
            if ( (unsigned __int64 *)*v56 != v56 + 2 )
              j_j___libc_free_0(*v56);
          }
          while ( v58 != v56 );
          v3 = v57;
          v54 = *(_QWORD *)(v30 + 160);
        }
        if ( v30 + 176 != v54 )
          _libc_free(v54);
        v59 = *(_QWORD *)(v30 + 48);
        v60 = 32LL * *(unsigned int *)(v30 + 56);
        if ( v59 != v59 + v60 )
        {
          v61 = (unsigned __int64 *)(v59 + v60);
          v62 = v3;
          v63 = *(unsigned __int64 **)(v30 + 48);
          do
          {
            v61 -= 4;
            if ( (unsigned __int64 *)*v61 != v61 + 2 )
              j_j___libc_free_0(*v61);
          }
          while ( v63 != v61 );
          v3 = v62;
          v59 = *(_QWORD *)(v30 + 48);
        }
        if ( v30 + 64 != v59 )
          _libc_free(v59);
        j_j___libc_free_0(v30);
        v30 = v77;
      }
      goto LABEL_69;
    }
  }
  v3 = 1;
LABEL_4:
  if ( v86 != (void **)v88 )
    j_j___libc_free_0((unsigned __int64)v86);
  return v3;
}
