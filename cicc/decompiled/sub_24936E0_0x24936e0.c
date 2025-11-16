// Function: sub_24936E0
// Address: 0x24936e0
//
__int64 __fastcall sub_24936E0(_QWORD *a1, unsigned __int64 a2, __int64 **a3, __int64 *a4, __int64 a5, __int64 a6)
{
  __int64 v8; // r14
  __int64 v9; // r14
  __int64 v10; // r15
  __int64 v11; // rax
  char v12; // al
  _QWORD *v13; // rax
  __int64 v14; // rbx
  __int64 v15; // r14
  __int64 v16; // r15
  __int64 v17; // rdx
  unsigned int v18; // esi
  unsigned __int64 v19; // rdx
  __int64 **v20; // rcx
  _BYTE *v21; // rax
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rsi
  __int64 v25; // r14
  __int64 v26; // rax
  char v27; // al
  char v28; // bl
  _QWORD *v29; // r15
  __int64 v30; // r14
  __int64 v31; // rbx
  __int64 v32; // rdx
  unsigned int v33; // esi
  bool v34; // zf
  __int64 v35; // rcx
  __int64 result; // rax
  char *v37; // rax
  size_t v38; // rdx
  char **v39; // rax
  __int64 v40; // r9
  __int64 v41; // rax
  __int64 v42; // rbx
  unsigned __int64 v43; // rax
  __int64 v44; // rdx
  unsigned __int64 v45; // r8
  __int64 v46; // r8
  __int64 v47; // rsi
  __int64 **v48; // r15
  __int64 v49; // rax
  __int64 v50; // r8
  __int64 **v51; // rdx
  unsigned __int8 *v52; // rsi
  __int64 v53; // r10
  __int64 v54; // rdi
  unsigned int v55; // esi
  unsigned __int8 **v56; // rax
  unsigned __int8 *v57; // r10
  __int64 v58; // rax
  unsigned __int64 v59; // rdx
  int v60; // r8d
  int v61; // r9d
  __int64 v62; // rax
  unsigned __int64 v63; // rdx
  char *v64; // rax
  int i; // edx
  void *v66; // r14
  size_t v67; // rsi
  int v68; // eax
  int v69; // ecx
  unsigned int v70; // [rsp+18h] [rbp-1A8h]
  unsigned __int8 v71; // [rsp+1Ch] [rbp-1A4h]
  unsigned __int8 v72; // [rsp+1Eh] [rbp-1A2h]
  _QWORD *v73; // [rsp+28h] [rbp-198h]
  __int64 v74; // [rsp+38h] [rbp-188h]
  __int64 v75; // [rsp+40h] [rbp-180h]
  char v77; // [rsp+58h] [rbp-168h]
  __int64 v78; // [rsp+58h] [rbp-168h]
  unsigned __int8 *v79; // [rsp+58h] [rbp-168h]
  __int64 **v80; // [rsp+58h] [rbp-168h]
  unsigned __int64 v81; // [rsp+58h] [rbp-168h]
  __int64 v82; // [rsp+58h] [rbp-168h]
  __int64 v83; // [rsp+58h] [rbp-168h]
  __int64 v85; // [rsp+60h] [rbp-160h]
  __int64 v86; // [rsp+60h] [rbp-160h]
  __int64 v87; // [rsp+60h] [rbp-160h]
  unsigned int v89; // [rsp+70h] [rbp-150h]
  unsigned int v90; // [rsp+78h] [rbp-148h]
  int *v91[2]; // [rsp+80h] [rbp-140h] BYREF
  int v92[8]; // [rsp+90h] [rbp-130h] BYREF
  __int16 v93; // [rsp+B0h] [rbp-110h]
  unsigned __int64 v94; // [rsp+C0h] [rbp-100h] BYREF
  __int64 v95; // [rsp+C8h] [rbp-F8h]
  _BYTE v96[16]; // [rsp+D0h] [rbp-F0h] BYREF
  __int16 v97; // [rsp+E0h] [rbp-E0h]
  unsigned __int64 v98; // [rsp+F0h] [rbp-D0h] BYREF
  __int64 v99; // [rsp+F8h] [rbp-C8h]
  _BYTE v100[16]; // [rsp+100h] [rbp-C0h] BYREF
  __int16 v101; // [rsp+110h] [rbp-B0h]
  int *v102; // [rsp+120h] [rbp-A0h] BYREF
  __int64 v103; // [rsp+128h] [rbp-98h]
  _BYTE v104[16]; // [rsp+130h] [rbp-90h] BYREF
  __int16 v105; // [rsp+140h] [rbp-80h]

  v8 = *(_QWORD *)(a2 - 32);
  if ( *(_BYTE *)v8 == 25 )
  {
    HIDWORD(v94) = 0;
    v34 = *(_BYTE *)(a6 + 108) == 0;
    v105 = 257;
    v98 = (unsigned int)v94;
    if ( v34 )
      return sub_24932B0((__int64 *)a6, 0x2Eu, a2, a3, (__int64)&v102, 0, v94, 0);
    else
      return sub_B358C0(a6, 0x6Eu, a2, (__int64)a3, (unsigned int)v94, (__int64)&v102, 0, 0, 0);
  }
  if ( *(_BYTE *)v8 || *(_QWORD *)(v8 + 24) != *(_QWORD *)(a2 + 80) )
    goto LABEL_4;
  v70 = *(_DWORD *)(v8 + 36);
  if ( !v70 )
  {
    if ( !sub_981210(*a4, v8, (unsigned int *)&v102) )
      goto LABEL_4;
    v64 = (char *)&unk_49D2DE0;
    for ( i = 449; i != (_DWORD)v102; i = *(_DWORD *)v64 )
    {
      v64 += 16;
      if ( &unk_49D3170 == (_UNKNOWN *)v64 )
        goto LABEL_4;
    }
    v66 = (void *)*((_QWORD *)v64 + 1);
    if ( !v66 )
      goto LABEL_4;
    v67 = strlen(*((const char **)v64 + 1));
    v39 = sub_2491530(v66, v67);
    goto LABEL_20;
  }
  v37 = (char *)sub_BD5D20(v8);
  v39 = sub_2491530(v37, v38);
  if ( v39 )
  {
LABEL_20:
    v70 = *((_DWORD *)v39 + 2);
    v75 = ((__int64 (__fastcall *)(_QWORD))v39[2])(a1[1]);
    goto LABEL_21;
  }
  v75 = *(_QWORD *)(v8 + 24);
LABEL_21:
  v102 = (int *)v104;
  v103 = 0x800000000LL;
  sub_B6DAB0(v70, (__int64)&v102);
  v94 = (unsigned __int64)v96;
  v91[0] = v102;
  v95 = 0x400000000LL;
  v91[1] = (int *)(unsigned int)v103;
  sub_B6B020(v75, v91, (__int64)&v94);
  v99 = 0x400000000LL;
  v98 = (unsigned __int64)v100;
  v41 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  if ( (_DWORD)v41 != 1 )
  {
    v74 = (unsigned int)(v41 - 2);
    v42 = 0;
    v73 = a1 + 2;
    while ( 1 )
    {
      v46 = *(_QWORD *)(a2 + 32 * (v42 - v41));
      v47 = *(_QWORD *)(v46 + 8);
      v48 = *(__int64 ***)(*(_QWORD *)(v75 + 16) + 8 * (v42 + 1));
      if ( (__int64 **)v47 == v48 )
      {
        v62 = (unsigned int)v99;
        v63 = (unsigned int)v99 + 1LL;
        if ( v63 > HIDWORD(v99) )
        {
          v82 = v46;
          sub_C8D5F0((__int64)&v98, v100, v63, 8u, v46, v40);
          v62 = (unsigned int)v99;
          v46 = v82;
        }
        *(_QWORD *)(v98 + 8 * v62) = v46;
        LODWORD(v99) = v99 + 1;
LABEL_27:
        if ( v74 == v42 )
          goto LABEL_40;
        goto LABEL_28;
      }
      v79 = *(unsigned __int8 **)(a2 + 32 * (v42 - v41));
      v49 = sub_2491640(v73, v47);
      v50 = (__int64)v79;
      v51 = (__int64 **)v49;
      if ( *v79 <= 0x15u )
      {
        v52 = v79;
        v80 = (__int64 **)v49;
        v53 = sub_2492FB0((_QWORD **)a5, v52);
        if ( v48 != v80 )
          goto LABEL_32;
        goto LABEL_37;
      }
      v54 = *(unsigned int *)(a5 + 32);
      v40 = *(_QWORD *)(a5 + 16);
      if ( (_DWORD)v54 )
      {
        v55 = (v54 - 1) & (((unsigned int)v79 >> 9) ^ ((unsigned int)v79 >> 4));
        v56 = (unsigned __int8 **)(v40 + 16LL * v55);
        v57 = *v56;
        if ( v79 == *v56 )
          goto LABEL_36;
        v68 = 1;
        while ( v57 != (unsigned __int8 *)-4096LL )
        {
          v69 = v68 + 1;
          v55 = (v54 - 1) & (v68 + v55);
          v56 = (unsigned __int8 **)(v40 + 16LL * v55);
          v57 = *v56;
          if ( v79 == *v56 )
            goto LABEL_36;
          v68 = v69;
        }
      }
      v56 = (unsigned __int8 **)(v40 + 16 * v54);
LABEL_36:
      v53 = (__int64)v56[1];
      if ( v48 != v51 )
      {
LABEL_32:
        v34 = *(_BYTE *)(a6 + 108) == 0;
        v93 = 257;
        v90 = v89;
        if ( v34 )
          v43 = sub_24932B0((__int64 *)a6, 0x2Du, v53, v48, (__int64)v92, 0, v89, 0);
        else
          v43 = sub_B358C0(a6, 0x71u, v53, (__int64)v48, v89, (__int64)v92, 0, v71, v72);
        v44 = (unsigned int)v99;
        v45 = (unsigned int)v99 + 1LL;
        if ( v45 > HIDWORD(v99) )
        {
          v81 = v43;
          sub_C8D5F0((__int64)&v98, v100, (unsigned int)v99 + 1LL, 8u, v45, v40);
          v44 = (unsigned int)v99;
          v43 = v81;
        }
        *(_QWORD *)(v98 + 8 * v44) = v43;
        LODWORD(v99) = v99 + 1;
        goto LABEL_27;
      }
LABEL_37:
      v58 = (unsigned int)v99;
      v59 = (unsigned int)v99 + 1LL;
      if ( v59 > HIDWORD(v99) )
      {
        v83 = v53;
        sub_C8D5F0((__int64)&v98, v100, v59, 8u, v50, v40);
        v58 = (unsigned int)v99;
        v53 = v83;
      }
      *(_QWORD *)(v98 + 8 * v58) = v53;
      LODWORD(v99) = v99 + 1;
      if ( v74 == v42 )
      {
LABEL_40:
        v60 = v98;
        v61 = v99;
        goto LABEL_41;
      }
LABEL_28:
      ++v42;
      v41 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
    }
  }
  v60 = (unsigned int)v100;
  v61 = 0;
LABEL_41:
  v93 = 257;
  result = sub_B33D10(a6, v70, v94, (unsigned int)v95, v60, v61, v90, (__int64)v92);
  if ( a3 != **(__int64 ****)(v75 + 16) )
  {
    v34 = *(_BYTE *)(a6 + 108) == 0;
    v93 = 257;
    if ( v34 )
      result = sub_24932B0((__int64 *)a6, 0x2Eu, result, a3, (__int64)v92, 0, v89, 0);
    else
      result = sub_B358C0(a6, 0x6Eu, result, (__int64)a3, v89, (__int64)v92, 0, 0, 0);
  }
  if ( (_BYTE *)v98 != v100 )
  {
    v85 = result;
    _libc_free(v98);
    result = v85;
  }
  if ( (_BYTE *)v94 != v96 )
  {
    v86 = result;
    _libc_free(v94);
    result = v86;
  }
  if ( v102 != (int *)v104 )
  {
    v87 = result;
    _libc_free((unsigned __int64)v102);
    result = v87;
  }
  if ( !result )
  {
LABEL_4:
    v101 = 257;
    v9 = a1[6];
    v10 = a1[53];
    v11 = sub_AA4E30(*(_QWORD *)(a6 + 48));
    v12 = sub_AE5020(v11, v9);
    v105 = 257;
    v77 = v12;
    v13 = sub_BD2C40(80, unk_3F10A14);
    v14 = (__int64)v13;
    if ( v13 )
      sub_B4D190((__int64)v13, v9, v10, (__int64)&v102, 0, v77, 0, 0);
    (*(void (__fastcall **)(_QWORD, __int64, unsigned __int64 *, _QWORD, _QWORD))(**(_QWORD **)(a6 + 88) + 16LL))(
      *(_QWORD *)(a6 + 88),
      v14,
      &v98,
      *(_QWORD *)(a6 + 56),
      *(_QWORD *)(a6 + 64));
    v15 = *(_QWORD *)a6;
    v16 = *(_QWORD *)a6 + 16LL * *(unsigned int *)(a6 + 8);
    if ( *(_QWORD *)a6 != v16 )
    {
      do
      {
        v17 = *(_QWORD *)(v15 + 8);
        v18 = *(_DWORD *)v15;
        v15 += 16;
        sub_B99FD0(v14, v18, v17);
      }
      while ( v16 != v15 );
    }
    v19 = *(_QWORD *)(a2 - 32);
    v105 = 257;
    v20 = (__int64 **)a1[6];
    v101 = 257;
    v21 = (_BYTE *)sub_24932B0((__int64 *)a6, 0x2Fu, v19, v20, (__int64)&v98, 0, v94, 0);
    v22 = sub_92B530((unsigned int **)a6, 0x20u, v14, v21, (__int64)&v102);
    v23 = a1[55];
    v24 = a1[54];
    v101 = 257;
    v97 = 257;
    v78 = v22;
    v25 = sub_2493450((__int64 *)a6, v24, v23, 0, 0, (__int64)&v94);
    v26 = sub_AA4E30(*(_QWORD *)(a6 + 48));
    v27 = sub_AE5020(v26, (__int64)a3);
    v105 = 257;
    v28 = v27;
    v29 = sub_BD2C40(80, unk_3F10A14);
    if ( v29 )
      sub_B4D190((__int64)v29, (__int64)a3, v25, (__int64)&v102, 0, v28, 0, 0);
    (*(void (__fastcall **)(_QWORD, _QWORD *, unsigned __int64 *, _QWORD, _QWORD))(**(_QWORD **)(a6 + 88) + 16LL))(
      *(_QWORD *)(a6 + 88),
      v29,
      &v98,
      *(_QWORD *)(a6 + 56),
      *(_QWORD *)(a6 + 64));
    v30 = *(_QWORD *)a6;
    v31 = *(_QWORD *)a6 + 16LL * *(unsigned int *)(a6 + 8);
    if ( *(_QWORD *)a6 != v31 )
    {
      do
      {
        v32 = *(_QWORD *)(v30 + 8);
        v33 = *(_DWORD *)v30;
        v30 += 16;
        sub_B99FD0((__int64)v29, v33, v32);
      }
      while ( v31 != v30 );
    }
    v92[1] = 0;
    v34 = *(_BYTE *)(a6 + 108) == 0;
    v105 = 257;
    v101 = 257;
    v94 = (unsigned int)v92[0];
    if ( v34 )
      v35 = sub_24932B0((__int64 *)a6, 0x2Eu, a2, a3, (__int64)&v98, 0, v92[0], 0);
    else
      v35 = sub_B358C0(a6, 0x6Eu, a2, (__int64)a3, (unsigned int)v92[0], (__int64)&v98, 0, 0, 0);
    return sub_B36550((unsigned int **)a6, v78, (__int64)v29, v35, (__int64)&v102, 0);
  }
  return result;
}
