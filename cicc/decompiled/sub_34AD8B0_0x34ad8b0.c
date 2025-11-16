// Function: sub_34AD8B0
// Address: 0x34ad8b0
//
void __fastcall sub_34AD8B0(__int64 *a1, __int64 a2, __int64 a3, _QWORD *a4, __int64 a5)
{
  __int64 v8; // r15
  __int64 v9; // rax
  __int64 (__fastcall *v10)(__int64); // rcx
  __int64 (__fastcall *v11)(__int64); // rax
  __int64 v12; // r14
  char *v13; // rax
  __int64 v14; // rdx
  char *v15; // rdi
  __int64 v16; // r9
  __int64 v17; // r8
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // rdx
  unsigned __int64 v22; // rdi
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // r8
  __int64 v30; // r9
  _QWORD *v31; // rax
  unsigned int v32; // ebx
  __int64 v33; // r14
  __int64 v34; // r10
  __int64 v35; // rax
  int v36; // edx
  int v37; // esi
  __int64 v38; // rax
  const __m128i *v39; // r10
  __int64 v40; // r11
  __int64 v41; // r14
  __int64 v42; // rdx
  __int64 v43; // r9
  __int64 v44; // rdx
  __int64 v45; // r9
  __int64 v46; // rcx
  unsigned int v47; // r8d
  unsigned int v48; // [rsp+20h] [rbp-420h]
  unsigned int v49; // [rsp+24h] [rbp-41Ch]
  unsigned int v50; // [rsp+28h] [rbp-418h]
  const __m128i *v51; // [rsp+28h] [rbp-418h]
  __int64 v52; // [rsp+30h] [rbp-410h]
  __int64 v54; // [rsp+40h] [rbp-400h]
  _QWORD v56[2]; // [rsp+50h] [rbp-3F0h] BYREF
  char v57; // [rsp+60h] [rbp-3E0h]
  _QWORD v58[2]; // [rsp+70h] [rbp-3D0h] BYREF
  char v59; // [rsp+80h] [rbp-3C0h]
  char *v60; // [rsp+90h] [rbp-3B0h] BYREF
  unsigned int v61; // [rsp+98h] [rbp-3A8h]
  char v62; // [rsp+A0h] [rbp-3A0h] BYREF
  unsigned __int64 v63[2]; // [rsp+B0h] [rbp-390h] BYREF
  _BYTE v64[16]; // [rsp+C0h] [rbp-380h] BYREF
  __int64 v65; // [rsp+D0h] [rbp-370h] BYREF
  unsigned __int64 v66; // [rsp+D8h] [rbp-368h] BYREF
  __int128 v67; // [rsp+E0h] [rbp-360h] BYREF
  int v68; // [rsp+128h] [rbp-318h]
  __int64 v69; // [rsp+130h] [rbp-310h]
  unsigned __int64 v70; // [rsp+138h] [rbp-308h]
  __int64 v71; // [rsp+140h] [rbp-300h] BYREF
  unsigned __int64 v72[2]; // [rsp+148h] [rbp-2F8h] BYREF
  _BYTE v73[64]; // [rsp+158h] [rbp-2E8h] BYREF
  unsigned int v74; // [rsp+198h] [rbp-2A8h]
  __int64 v75; // [rsp+1A0h] [rbp-2A0h]
  __int64 v76; // [rsp+1A8h] [rbp-298h]
  __int64 v77; // [rsp+1B0h] [rbp-290h] BYREF
  _QWORD v78[2]; // [rsp+1B8h] [rbp-288h] BYREF
  _BYTE v79[64]; // [rsp+1C8h] [rbp-278h] BYREF
  int v80; // [rsp+208h] [rbp-238h]
  __int64 v81; // [rsp+210h] [rbp-230h]
  __int64 v82; // [rsp+218h] [rbp-228h]
  char v83[112]; // [rsp+220h] [rbp-220h] BYREF
  __m128i v84; // [rsp+290h] [rbp-1B0h] BYREF
  unsigned int v85; // [rsp+2A0h] [rbp-1A0h]
  __int64 v86; // [rsp+2B8h] [rbp-188h]
  int v87; // [rsp+2C8h] [rbp-178h]
  unsigned __int64 v88; // [rsp+2D0h] [rbp-170h]
  char v89; // [rsp+2E0h] [rbp-160h] BYREF
  unsigned int v90; // [rsp+2E8h] [rbp-158h]
  __int64 v91; // [rsp+2F0h] [rbp-150h]
  __int64 v92; // [rsp+2F8h] [rbp-148h]
  __int64 v93; // [rsp+300h] [rbp-140h]
  char v94[8]; // [rsp+308h] [rbp-138h] BYREF
  int v95; // [rsp+310h] [rbp-130h]
  int v96; // [rsp+358h] [rbp-E8h]
  __int64 v97; // [rsp+360h] [rbp-E0h]
  __int64 v98; // [rsp+368h] [rbp-D8h]
  char *v99; // [rsp+3E0h] [rbp-60h]
  char v100; // [rsp+3F0h] [rbp-50h] BYREF

  if ( *(_WORD *)(a2 + 68) == 20 )
  {
    v12 = *(_QWORD *)(a2 + 32);
    v54 = v12 + 40;
  }
  else
  {
    v8 = a1[2];
    v9 = *(_QWORD *)v8;
    v10 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v8 + 520LL);
    if ( v10 == sub_2DCA430 )
    {
LABEL_3:
      v11 = *(__int64 (__fastcall **)(__int64))(v9 + 528);
      if ( v11 == sub_2E77FE0 )
        return;
      ((void (__fastcall *)(_QWORD *, __int64, __int64))v11)(v56, v8, a2);
      v12 = v56[0];
      v54 = v56[1];
      if ( !v57 )
        return;
      goto LABEL_6;
    }
    ((void (__fastcall *)(_QWORD *, __int64, __int64))v10)(v58, v8, a2);
    v12 = v58[0];
    v54 = v58[1];
    if ( !v59 )
    {
      v9 = *(_QWORD *)v8;
      goto LABEL_3;
    }
  }
LABEL_6:
  if ( (*(_BYTE *)(v12 + 3) & 0x10) == 0 )
    return;
  v49 = *(_DWORD *)(v54 + 8);
  v48 = *(_DWORD *)(v12 + 8);
  v13 = sub_E922F0((_QWORD *)a1[1], v48);
  v15 = &v13[2 * v14];
  if ( v13 == v15 )
    return;
  v16 = a1[5];
  v17 = 1;
  while ( (*(_QWORD *)(v16 + 8 * ((unsigned __int64)*(unsigned __int16 *)v13 >> 6)) & (1LL << *(_WORD *)v13)) == 0 )
  {
    v13 += 2;
    if ( v15 == v13 )
      return;
  }
  if ( *(_BYTE *)v12 || !sub_349DFE0(v12, *(_QWORD *)(a2 + 24), a1[1]) )
    goto LABEL_13;
  sub_34A41A0((__int64)&v77, a3 + 8, (_QWORD *)0x4000000100000000LL, 0x4000000200000000uLL, v17, v16);
  sub_34A19F0((__int64)&v65, (__int64)&v77, v23, v24, v25, v26);
  sub_34A19F0((__int64)&v71, (__int64)v83, v27, v28, v29, v30);
  v50 = v74;
  v52 = v69;
  v31 = a4;
  v32 = v68;
  v33 = (__int64)v31;
  while ( v32 != v50 )
  {
    while ( 1 )
    {
LABEL_30:
      v34 = sub_349D6E0(v33, __ROL8__(v52 + v32, 32));
      if ( *(_DWORD *)(v34 + 56) == 2 )
      {
        v84.m128i_i32[0] = 1;
        v84.m128i_i64[1] = v49;
        v38 = sub_349EDA0(
                *(_QWORD *)(v34 + 64),
                *(_QWORD *)(v34 + 64) + 32LL * *(unsigned int *)(v34 + 72),
                (__int64)&v84);
        if ( v40 != v38 )
        {
          a4 = (_QWORD *)v33;
          v51 = v39;
          v41 = v39[2].m128i_i64[1];
          sub_349F140((__int64)&v84, v39[3].m128i_i64[0]);
          v87 = 3;
          v86 = v41;
          *(_QWORD *)(v88 + 8) = v48;
          sub_34A9810(a3, v51, v42, v48, (__int64)v51, v43);
          sub_34A0610(&v60, a4, (__int64)&v84);
          v46 = v61;
          v63[0] = (unsigned __int64)v64;
          v63[1] = 0x200000000LL;
          if ( v61 )
            sub_349DD80((__int64)v63, (__int64)&v60, v44, v61, (__int64)v63, v45);
          sub_34AADC0(a3, (__int64)v63, &v84, v46, (__int64)v63, v45);
          if ( (_BYTE *)v63[0] != v64 )
            _libc_free(v63[0]);
          if ( v60 != &v62 )
            _libc_free((unsigned __int64)v60);
          if ( v99 != &v100 )
            _libc_free((unsigned __int64)v99);
          if ( (char *)v88 != &v89 )
            _libc_free(v88);
          goto LABEL_36;
        }
      }
      if ( v52 + (unsigned __int64)v32 >= v70 )
        break;
      v68 = ++v32;
      if ( v32 == v50 )
        goto LABEL_33;
    }
    v35 = v66 + 16LL * (unsigned int)v67 - 16;
    v36 = *(_DWORD *)(v35 + 12) + 1;
    *(_DWORD *)(v35 + 12) = v36;
    v37 = v67;
    if ( v36 == *(_DWORD *)(v66 + 16LL * (unsigned int)v67 - 8) )
    {
      v47 = *(_DWORD *)(v65 + 192);
      if ( v47 )
      {
        sub_F03D40((__int64 *)&v66, v47);
        v37 = v67;
      }
    }
    if ( v37 && *(_DWORD *)(v66 + 12) < *(_DWORD *)(v66 + 8) )
    {
      v68 = 0;
      v32 = 0;
      v52 = *(_QWORD *)sub_34A2590((__int64)&v65);
      v69 = v52;
      v70 = *(_QWORD *)sub_34A25B0((__int64)&v65);
      v50 = v74;
    }
    else
    {
      v32 = -1;
      v68 = -1;
      v69 = 0;
      v70 = 0;
      v50 = v74;
      v52 = 0;
    }
  }
LABEL_33:
  if ( v75 != v52 || v70 != v76 )
    goto LABEL_30;
  a4 = (_QWORD *)v33;
LABEL_36:
  if ( (_BYTE *)v72[0] != v73 )
    _libc_free(v72[0]);
  if ( (__int128 *)v66 != (__int128 *)((char *)&v67 + 8) )
    _libc_free(v66);
  sub_34A03D0((__int64)&v77);
LABEL_13:
  if ( (((*(_BYTE *)(v54 + 3) & 0x40) != 0) & ((*(_BYTE *)(v54 + 3) >> 4) ^ 1)) != 0 )
  {
    sub_34A41A0(
      (__int64)&v84,
      a3 + 8,
      (_QWORD *)((unsigned __int64)v49 << 32),
      (unsigned __int64)(v49 + 1) << 32,
      v17,
      v16);
    v21 = v85;
    v71 = v84.m128i_i64[0];
    v72[0] = (unsigned __int64)v73;
    v72[1] = 0x400000000LL;
    if ( v85 )
      sub_349DB40((__int64)v72, (__int64)&v84.m128i_i64[1], v85, v18, v19, v20);
    v78[0] = v79;
    v74 = v90;
    v75 = v91;
    v76 = v92;
    v77 = v93;
    v78[1] = 0x400000000LL;
    if ( v95 )
      sub_349DB40((__int64)v78, (__int64)v94, v21, v18, v19, v20);
    v80 = v96;
    v81 = v97;
    v82 = v98;
    if ( sub_34A1A50((__int64)&v71, (__int64)&v77) )
    {
      v22 = v78[0];
      if ( (_BYTE *)v78[0] != v79 )
        goto LABEL_20;
    }
    else
    {
      v67 = 0;
      v65 = 1;
      v66 = v49;
      sub_34AC3C0((__int64)a1, a2, a3, a5, (__int64)a4, __ROL8__(v75 + v74, 32), 0, (__int64)&v65, v48);
      v22 = v78[0];
      if ( (_BYTE *)v78[0] != v79 )
LABEL_20:
        _libc_free(v22);
    }
    if ( (_BYTE *)v72[0] != v73 )
      _libc_free(v72[0]);
    sub_34A03D0((__int64)&v84);
  }
}
