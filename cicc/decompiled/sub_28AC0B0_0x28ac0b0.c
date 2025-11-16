// Function: sub_28AC0B0
// Address: 0x28ac0b0
//
__int64 __fastcall sub_28AC0B0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        unsigned __int64 a7,
        char a8)
{
  __int64 v8; // rax
  unsigned int v9; // edx
  __int64 v10; // rax
  unsigned int v11; // r13d
  __int64 v15; // r15
  __int64 v16; // r15
  __int64 v17; // r15
  char v18; // al
  __int64 v19; // r9
  char v20; // al
  unsigned __int64 v21; // rcx
  unsigned __int8 v22; // di
  __int16 v23; // cx
  unsigned __int64 v24; // rax
  unsigned __int8 v25; // dl
  char *v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // r8
  __int64 v30; // r9
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // r8
  __int64 v34; // r9
  _QWORD **v35; // r14
  _QWORD **v36; // r12
  _QWORD *v37; // rsi
  __int64 *v38; // rax
  __int64 *v39; // rbx
  __int64 v40; // rdi
  __int64 *v41; // r12
  __int64 *v42; // rax
  _QWORD *v43; // r14
  unsigned __int8 v44; // dl
  char v45; // dh
  unsigned __int64 *v46; // r8
  char v47; // al
  __int64 v48; // rcx
  __int64 v50; // [rsp+28h] [rbp-258h] BYREF
  __int64 v51; // [rsp+30h] [rbp-250h] BYREF
  __int64 v52; // [rsp+38h] [rbp-248h] BYREF
  char v53; // [rsp+4Bh] [rbp-235h] BYREF
  char v54; // [rsp+4Ch] [rbp-234h] BYREF
  char v55; // [rsp+4Dh] [rbp-233h] BYREF
  unsigned __int8 v56; // [rsp+4Eh] [rbp-232h] BYREF
  unsigned __int8 v57; // [rsp+4Fh] [rbp-231h] BYREF
  __int64 v58; // [rsp+50h] [rbp-230h] BYREF
  char v59; // [rsp+58h] [rbp-228h]
  char v60; // [rsp+60h] [rbp-220h]
  __int64 v61; // [rsp+70h] [rbp-210h] BYREF
  char v62; // [rsp+78h] [rbp-208h]
  char v63; // [rsp+80h] [rbp-200h]
  _QWORD v64[6]; // [rsp+90h] [rbp-1F0h] BYREF
  _BYTE *v65; // [rsp+C0h] [rbp-1C0h] BYREF
  __int64 v66; // [rsp+C8h] [rbp-1B8h]
  _BYTE v67[32]; // [rsp+D0h] [rbp-1B0h] BYREF
  _QWORD v68[6]; // [rsp+F0h] [rbp-190h] BYREF
  _QWORD v69[6]; // [rsp+120h] [rbp-160h] BYREF
  _QWORD v70[6]; // [rsp+150h] [rbp-130h] BYREF
  _QWORD v71[8]; // [rsp+180h] [rbp-100h] BYREF
  __int64 v72; // [rsp+1C0h] [rbp-C0h] BYREF
  __int64 *v73; // [rsp+1C8h] [rbp-B8h]
  __int64 v74; // [rsp+1D0h] [rbp-B0h]
  int v75; // [rsp+1D8h] [rbp-A8h]
  char v76; // [rsp+1DCh] [rbp-A4h]
  char v77; // [rsp+1E0h] [rbp-A0h] BYREF
  _BYTE *v78; // [rsp+200h] [rbp-80h] BYREF
  __int64 v79; // [rsp+208h] [rbp-78h]
  _BYTE v80[112]; // [rsp+210h] [rbp-70h] BYREF

  v8 = *(_QWORD *)(a5 + 8);
  v51 = a3;
  v52 = a2;
  v9 = *(_DWORD *)(v8 + 8);
  v10 = *(_QWORD *)(a4 + 8);
  v50 = a5;
  if ( v9 >> 8 == *(_DWORD *)(v10 + 8) >> 8
    && (v15 = sub_B43CC0(a4), sub_B4CED0((__int64)&v58, v50, v15), v60)
    && a7 == v58
    && v59 == a8
    && (sub_B4CED0((__int64)&v61, a4, v15), v63)
    && v61 == a7
    && v62 == a8
    && sub_B4D040(v50)
    && sub_B4D040(a4) )
  {
    v72 = 0;
    v16 = 0xBFFFFFFFFFFFFFFELL;
    v65 = v67;
    v73 = (__int64 *)&v77;
    v66 = 0x400000000LL;
    v71[1] = &v50;
    v71[6] = &v72;
    v71[2] = &v53;
    v74 = 4;
    v75 = 0;
    v76 = 1;
    v53 = 0;
    v71[0] = a1;
    v71[3] = &v54;
    v71[4] = &v61;
    v71[5] = &v65;
    v55 = 0;
    if ( a7 <= 0x3FFFFFFFFFFFFFFBLL )
    {
      v17 = 0x4000000000000000LL;
      if ( !a8 )
        v17 = 0;
      v16 = a7 | v17;
    }
    v64[1] = a6;
    v78 = v80;
    v79 = 0x800000000LL;
    v64[0] = &v51;
    v64[4] = &v78;
    v68[0] = a4;
    v68[1] = v16;
    memset(&v68[2], 0, 32);
    v64[2] = v68;
    v64[3] = &v55;
    v18 = sub_28AA520((__int64)v71, a4, (unsigned __int8 (__fastcall *)(__int64, __int64))sub_28A9A60, (__int64)v64);
    v19 = a6;
    if ( !v18 )
      goto LABEL_42;
    if ( (_DWORD)v79 )
    {
      v20 = sub_D0E9A0((__int64)&v78, *(_QWORD *)(v51 + 40), 0, *(_QWORD *)(a1 + 24), 0, a6);
      v19 = a6;
      if ( v20 )
        goto LABEL_42;
    }
    v70[1] = &v52;
    v69[0] = v50;
    v70[2] = &v51;
    v69[1] = v16;
    memset(&v69[2], 0, 32);
    v70[0] = a1;
    v70[3] = v19;
    v70[4] = v69;
    v70[5] = &v55;
    v11 = sub_28AA520((__int64)v71, v50, (unsigned __int8 (__fastcall *)(__int64, __int64))sub_28A9440, (__int64)v70);
    if ( !(_BYTE)v11 )
    {
LABEL_42:
      v11 = 0;
    }
    else
    {
      if ( v53 )
      {
        v43 = (_QWORD *)v50;
        v46 = (unsigned __int64 *)sub_AA5190(*(_QWORD *)(v50 + 40));
        if ( v46 )
        {
          v47 = v45;
        }
        else
        {
          v47 = 0;
          v44 = 0;
        }
        v48 = v44;
        BYTE1(v48) = v47;
        sub_B44550(v43, *(_QWORD *)(v50 + 40), v46, v48);
      }
      _BitScanReverse64(&v21, 1LL << *(_WORD *)(a4 + 2));
      v22 = 63 - (v21 ^ 0x3F);
      v23 = *(_WORD *)(v50 + 2);
      v57 = v22;
      _BitScanReverse64(&v24, 1LL << v23);
      v25 = 63 - (v24 ^ 0x3F);
      v26 = (char *)&v56;
      v56 = v25;
      if ( v25 < v22 )
        v26 = (char *)&v57;
      *(_WORD *)(v50 + 2) = (unsigned __int8)*v26 | v23 & 0xFFC0;
      sub_BD84D0(a4, v50);
      sub_28AAD10(a1, (_QWORD *)a4, v27, v28, v29, v30);
      sub_B9ADA0(v50, 0, 0);
      if ( (_DWORD)v66 )
      {
        v35 = (_QWORD **)v65;
        v36 = (_QWORD **)&v65[8 * (unsigned int)v66];
        do
        {
          v37 = *v35++;
          sub_28AAD10(a1, v37, v31, v32, v33, v34);
        }
        while ( v36 != v35 );
      }
      v38 = v73;
      if ( v76 )
        v39 = &v73[HIDWORD(v74)];
      else
        v39 = &v73[(unsigned int)v74];
      if ( v73 != v39 )
      {
        while ( 1 )
        {
          v40 = *v38;
          v41 = v38;
          if ( (unsigned __int64)*v38 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v39 == ++v38 )
            goto LABEL_31;
        }
        while ( v39 != v41 )
        {
          sub_B99FD0(v40, 8u, 0);
          v42 = v41 + 1;
          if ( v41 + 1 == v39 )
            break;
          while ( 1 )
          {
            v40 = *v42;
            v41 = v42;
            if ( (unsigned __int64)*v42 < 0xFFFFFFFFFFFFFFFELL )
              break;
            if ( v39 == ++v42 )
              goto LABEL_31;
          }
        }
      }
    }
LABEL_31:
    if ( v78 != v80 )
      _libc_free((unsigned __int64)v78);
    if ( !v76 )
      _libc_free((unsigned __int64)v73);
    if ( v65 != v67 )
      _libc_free((unsigned __int64)v65);
  }
  else
  {
    return 0;
  }
  return v11;
}
