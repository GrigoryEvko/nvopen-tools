// Function: sub_2F5A640
// Address: 0x2f5a640
//
__int64 __fastcall sub_2F5A640(_QWORD *a1, __int64 a2)
{
  __int64 (*v3)(void); // rdx
  __int64 v4; // rax
  unsigned int v5; // r13d
  __int64 v7; // r14
  __int64 (*v8)(); // rdx
  int v9; // eax
  unsigned int v10; // edx
  _QWORD *v11; // r15
  _QWORD *v12; // r14
  unsigned __int64 v13; // rsi
  _QWORD *v14; // rax
  _QWORD *v15; // rdi
  __int64 v16; // rcx
  __int64 v17; // rdx
  __int64 v18; // rcx
  _QWORD *v19; // r8
  __int64 v20; // rax
  _QWORD *v21; // rdi
  __int64 v22; // rdx
  char v23; // al
  _QWORD *v24; // r15
  _QWORD *v25; // r14
  unsigned __int64 v26; // rsi
  _QWORD *v27; // rax
  _QWORD *v28; // rdi
  __int64 v29; // rcx
  __int64 v30; // rdx
  __int64 v31; // rcx
  _QWORD *v32; // r8
  __int64 v33; // rax
  _QWORD *v34; // rdi
  __int64 v35; // rdx
  char v36; // al
  _QWORD *v37; // r14
  unsigned __int64 v38; // rdi
  __int64 v39; // rsi
  __int64 v40; // rdx
  __int64 v41; // r9
  __int64 v42; // r8
  unsigned __int64 v43; // rax
  __int64 v44; // rdi
  unsigned __int64 v45; // rax
  __int64 v46; // rdi
  __int64 v47; // r15
  __int64 v48; // rbx
  _QWORD *v49; // rax
  _QWORD *v50; // rcx
  unsigned __int64 v51; // rdi
  void (*v52)(void); // rax
  __int64 v53; // rdx
  __int64 v54; // rsi
  __int64 v55; // rax
  __int64 v56; // rdi
  __int64 v57; // rax
  __int64 v58; // r15
  _QWORD *v59; // rbx
  unsigned __int64 v60; // rdi
  unsigned __int64 v61; // rdi
  unsigned __int64 v62; // rdi
  unsigned __int64 v63; // rdi
  __int64 v64; // rax
  __int64 v65; // rbx
  unsigned __int64 v66; // rdi
  __int64 v67; // rsi
  __int64 v68; // r9
  unsigned __int64 v69; // rax
  __int64 v70; // r14
  unsigned __int64 v71; // rdx
  __int64 v72; // rdx
  __int64 v73; // rcx
  unsigned __int64 v74; // r8
  __int64 v75; // r9
  __int64 v76; // rdi
  __int64 (*v77)(); // rdx
  __int64 v78; // rdi
  __int64 (*v79)(); // rdx
  const char *v80; // r13
  __int64 *v81; // rax
  const char *v82; // r14
  __int64 *v83; // rax
  unsigned __int64 v84; // rdi
  int v85; // eax
  __int64 v86; // r15
  _QWORD *v87; // r14
  _QWORD *v88; // r15
  unsigned __int64 v89; // rdi
  unsigned __int64 v90; // rdi
  __int64 v91; // rax
  __int64 v92; // [rsp+8h] [rbp-68h]
  __int64 v93; // [rsp+10h] [rbp-60h]
  __int64 v94; // [rsp+18h] [rbp-58h]
  __int64 v95; // [rsp+18h] [rbp-58h]
  int v96; // [rsp+18h] [rbp-58h]
  unsigned __int64 v97[10]; // [rsp+20h] [rbp-50h] BYREF

  a1[96] = a2;
  v3 = *(__int64 (**)(void))(**(_QWORD **)(a2 + 16) + 128LL);
  v4 = 0;
  if ( v3 != sub_2DAC790 )
    v4 = v3();
  a1[97] = v4;
  if ( unk_503FCFD )
  {
    v80 = (const char *)a1[96];
    v81 = (__int64 *)sub_CB72A0();
    sub_2F06090(v80, a1[4], a1[98], (__int64)"Before greedy register allocator", v81, 1);
  }
  sub_35B4B20(a1, a1[3], a1[4], a1[5]);
  v5 = sub_2F55040((__int64)a1);
  if ( (_BYTE)v5 )
  {
    sub_2FAD5E0(a1[98]);
    sub_2F54D60((__int64)a1);
    v7 = a1[1];
    v8 = *(__int64 (**)())(*(_QWORD *)v7 + 328LL);
    v9 = 0;
    if ( v8 != sub_2F3F790 )
      v9 = ((__int64 (__fastcall *)(_QWORD, _QWORD))v8)(a1[1], a1[96]);
    v10 = *(_DWORD *)(v7 + 16);
    a1[3633] = **(_QWORD **)(v7 + 248) + v10 * v9;
    a1[3634] = v10;
    v11 = sub_C52410();
    v12 = v11 + 1;
    v13 = sub_C959E0();
    v14 = (_QWORD *)v11[2];
    if ( v14 )
    {
      v15 = v11 + 1;
      do
      {
        while ( 1 )
        {
          v16 = v14[2];
          v17 = v14[3];
          if ( v13 <= v14[4] )
            break;
          v14 = (_QWORD *)v14[3];
          if ( !v17 )
            goto LABEL_14;
        }
        v15 = v14;
        v14 = (_QWORD *)v14[2];
      }
      while ( v16 );
LABEL_14:
      if ( v12 != v15 && v13 >= v15[4] )
        v12 = v15;
    }
    if ( v12 == (_QWORD *)((char *)sub_C52410() + 8) )
      goto LABEL_88;
    v20 = v12[7];
    v19 = v12 + 6;
    if ( !v20 )
      goto LABEL_88;
    v21 = v12 + 6;
    do
    {
      while ( 1 )
      {
        v18 = *(_QWORD *)(v20 + 16);
        v22 = *(_QWORD *)(v20 + 24);
        if ( *(_DWORD *)(v20 + 32) >= dword_5023BA8 )
          break;
        v20 = *(_QWORD *)(v20 + 24);
        if ( !v22 )
          goto LABEL_23;
      }
      v21 = (_QWORD *)v20;
      v20 = *(_QWORD *)(v20 + 16);
    }
    while ( v18 );
LABEL_23:
    if ( v19 == v21 || dword_5023BA8 < *((_DWORD *)v21 + 8) || (v23 = qword_5023C28, !*((_DWORD *)v21 + 9)) )
    {
LABEL_88:
      v76 = a1[1];
      v77 = *(__int64 (**)())(*(_QWORD *)v76 + 672LL);
      v23 = 0;
      if ( v77 != sub_2F4C0A0 )
        v23 = ((__int64 (__fastcall *)(__int64, _QWORD, __int64 (*)(), __int64, _QWORD *))v77)(
                v76,
                a1[96],
                v77,
                v18,
                v19);
    }
    *((_BYTE *)a1 + 29080) = v23;
    v24 = sub_C52410();
    v25 = v24 + 1;
    v26 = sub_C959E0();
    v27 = (_QWORD *)v24[2];
    if ( v27 )
    {
      v28 = v24 + 1;
      do
      {
        while ( 1 )
        {
          v29 = v27[2];
          v30 = v27[3];
          if ( v26 <= v27[4] )
            break;
          v27 = (_QWORD *)v27[3];
          if ( !v30 )
            goto LABEL_31;
        }
        v28 = v27;
        v27 = (_QWORD *)v27[2];
      }
      while ( v29 );
LABEL_31:
      if ( v25 != v28 && v26 >= v28[4] )
        v25 = v28;
    }
    if ( v25 == (_QWORD *)((char *)sub_C52410() + 8) )
      goto LABEL_90;
    v33 = v25[7];
    v32 = v25 + 6;
    if ( !v33 )
      goto LABEL_90;
    v26 = (unsigned int)dword_5023AC8;
    v34 = v25 + 6;
    do
    {
      while ( 1 )
      {
        v31 = *(_QWORD *)(v33 + 16);
        v35 = *(_QWORD *)(v33 + 24);
        if ( *(_DWORD *)(v33 + 32) >= dword_5023AC8 )
          break;
        v33 = *(_QWORD *)(v33 + 24);
        if ( !v35 )
          goto LABEL_40;
      }
      v34 = (_QWORD *)v33;
      v33 = *(_QWORD *)(v33 + 16);
    }
    while ( v31 );
LABEL_40:
    if ( v34 == v32
      || dword_5023AC8 < *((_DWORD *)v34 + 8)
      || (v31 = *((unsigned int *)v34 + 9), v36 = qword_5023B48, !(_DWORD)v31) )
    {
LABEL_90:
      v78 = a1[1];
      v79 = *(__int64 (**)())(*(_QWORD *)v78 + 464LL);
      v36 = 0;
      if ( v79 != sub_2F4C060 )
        v36 = ((__int64 (__fastcall *)(__int64, unsigned __int64, __int64 (*)(), __int64, _QWORD *))v79)(
                v78,
                v26,
                v79,
                v31,
                v32);
    }
    *((_BYTE *)a1 + 29081) = v36;
    v37 = a1 + 117;
    if ( *((_BYTE *)a1 + 960) )
    {
      v38 = a1[115];
      if ( (_QWORD *)v38 != v37 )
        _libc_free(v38);
    }
    v39 = a1[107];
    a1[115] = v37;
    a1[116] = 0;
    v40 = a1[96];
    v41 = a1[101];
    v42 = a1[99];
    a1[117] = 0;
    *((_DWORD *)a1 + 238) = 1;
    *((_BYTE *)a1 + 960) = 1;
    (*(void (__fastcall **)(unsigned __int64 *, __int64, __int64, _QWORD *, __int64, __int64))(*(_QWORD *)v39 + 24LL))(
      v97,
      v39,
      v40,
      a1,
      v42,
      v41);
    v43 = v97[0];
    v44 = a1[121];
    v97[0] = 0;
    a1[121] = v43;
    if ( v44 )
    {
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v44 + 8LL))(v44);
      if ( v97[0] )
        (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v97[0] + 8LL))(v97[0]);
    }
    (*(void (__fastcall **)(unsigned __int64 *, _QWORD, _QWORD, _QWORD *, _QWORD))(*(_QWORD *)a1[108] + 24LL))(
      v97,
      a1[108],
      a1[96],
      a1,
      a1[98]);
    v45 = v97[0];
    v46 = a1[122];
    v97[0] = 0;
    a1[122] = v45;
    if ( v46 )
    {
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v46 + 8LL))(v46);
      if ( v97[0] )
        (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v97[0] + 8LL))(v97[0]);
    }
    v47 = a1[99];
    v92 = a1[3];
    v48 = a1[101];
    v93 = a1[4];
    v94 = a1[96];
    v49 = (_QWORD *)sub_22077B0(0x38u);
    v50 = v49;
    if ( v49 )
    {
      v49[4] = v48;
      v49[5] = 0;
      v49[1] = v94;
      *v49 = &unk_4A2AFC8;
      v49[2] = v93;
      v49[3] = v92;
      v49[6] = v47;
    }
    v51 = a1[114];
    a1[114] = v49;
    if ( v51 )
    {
      v52 = *(void (**)(void))(*(_QWORD *)v51 + 8LL);
      if ( (char *)v52 == (char *)sub_2F4C270 )
        j_j___libc_free_0(v51);
      else
        v52();
      v50 = (_QWORD *)a1[114];
    }
    v53 = a1[3];
    v54 = a1[96];
    v97[0] = a1[4];
    v97[1] = a1[106];
    v97[2] = a1[100];
    v97[3] = a1[99];
    v55 = sub_34F6050(v97, v54, v53, v50);
    v56 = a1[109];
    a1[109] = v55;
    if ( v56 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v56 + 16LL))(v56);
    sub_34C97A0(a1[114]);
    v57 = sub_22077B0(0x2C0u);
    v58 = v57;
    if ( v57 )
      sub_2FB0E40(v57, a1[3], a1[4], a1[101]);
    v59 = (_QWORD *)a1[124];
    a1[124] = v58;
    if ( v59 )
    {
      v60 = v59[78];
      if ( (_QWORD *)v60 != v59 + 80 )
        _libc_free(v60);
      v61 = v59[35];
      if ( (_QWORD *)v61 != v59 + 37 )
        _libc_free(v61);
      v62 = v59[25];
      if ( (_QWORD *)v62 != v59 + 27 )
        _libc_free(v62);
      v63 = v59[7];
      if ( (_QWORD *)v63 != v59 + 9 )
        _libc_free(v63);
      j_j___libc_free_0((unsigned __int64)v59);
      v58 = a1[124];
    }
    v95 = a1[114];
    v64 = sub_22077B0(0x738u);
    v65 = v64;
    if ( v64 )
      sub_2FB1ED0(v64, v58, a1[4], a1[3], a1[100], a1[99], v95);
    v66 = a1[125];
    a1[125] = v65;
    if ( v66 )
      sub_2F4CF00(v66);
    v67 = a1[96];
    sub_3501A90(a1 + 126, v67, *(_QWORD *)(a1[5] + 48LL), a1[98], a1[4], a1[1]);
    v69 = *((unsigned int *)a1 + 6046);
    if ( v69 != 32 )
    {
      if ( v69 > 0x20 )
      {
        v86 = a1[3022];
        v87 = (_QWORD *)(v86 + 144 * v69);
        v88 = (_QWORD *)(v86 + 4608);
        while ( v88 != v87 )
        {
          v87 -= 18;
          v89 = v87[12];
          if ( (_QWORD *)v89 != v87 + 14 )
            _libc_free(v89);
          v90 = v87[3];
          if ( (_QWORD *)v90 != v87 + 5 )
            _libc_free(v90);
          v91 = v87[1];
          v87[2] = 0;
          if ( v91 )
            --*(_DWORD *)(v91 + 8);
        }
      }
      else
      {
        if ( *((_DWORD *)a1 + 6047) <= 0x1Fu )
        {
          v70 = sub_C8D7D0((__int64)(a1 + 3022), (__int64)(a1 + 3024), 0x20u, 0x90u, v97, v68);
          sub_2F57150((__int64)(a1 + 3022), v70);
          v84 = a1[3022];
          v85 = v97[0];
          if ( a1 + 3024 != (_QWORD *)v84 )
          {
            v96 = v97[0];
            _libc_free(v84);
            v85 = v96;
          }
          *((_DWORD *)a1 + 6047) = v85;
          v69 = *((unsigned int *)a1 + 6046);
          a1[3022] = v70;
        }
        else
        {
          v70 = a1[3022];
        }
        v67 = v70 + 4608;
        v71 = v70 + 144 * v69;
        if ( v71 != v70 + 4608 )
        {
          do
          {
            if ( v71 )
            {
              memset((void *)v71, 0, 0x90u);
              *(_DWORD *)(v71 + 36) = 6;
              *(_QWORD *)(v71 + 24) = v71 + 40;
              *(_QWORD *)(v71 + 96) = v71 + 112;
              *(_DWORD *)(v71 + 108) = 8;
            }
            v71 += 144LL;
          }
          while ( v67 != v71 );
        }
      }
      *((_DWORD *)a1 + 6046) = 32;
    }
    sub_2F55730((__int64)(a1 + 3619));
    *((_DWORD *)a1 + 7248) = 0;
    sub_35B5380(a1);
    sub_2F58C00((__int64)a1, v67, v72, v73, v74, v75);
    if ( unk_503FCFD )
    {
      v82 = (const char *)a1[96];
      v83 = (__int64 *)sub_CB72A0();
      sub_2F06090(v82, a1[4], a1[98], (__int64)"Before post optimization", v83, 1);
    }
    (*(void (__fastcall **)(_QWORD *))(*a1 + 24LL))(a1);
    sub_2F5A580(a1);
    sub_2F50510((__int64)a1);
  }
  return v5;
}
