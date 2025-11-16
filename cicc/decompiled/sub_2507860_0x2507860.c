// Function: sub_2507860
// Address: 0x2507860
//
__int64 __fastcall sub_2507860(__int64 *a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rcx
  __int64 *v6; // r14
  __int64 v8; // r13
  __int64 v9; // rax
  unsigned int v10; // r15d
  __int64 v11; // rdx
  __int64 v12; // r9
  __int64 *v13; // r13
  __int64 v14; // r12
  __int64 v15; // rdx
  __int64 v16; // r8
  __int64 v17; // rsi
  __int64 v18; // r12
  __int64 v19; // rax
  int v20; // r13d
  int v21; // edx
  _QWORD *v22; // rax
  int v23; // r9d
  __int64 v24; // r13
  __int64 v25; // rax
  __int64 v26; // rax
  unsigned __int64 v27; // rdx
  __int64 v28; // rax
  __int64 v29; // rdx
  unsigned __int64 v30; // r9
  __int64 v31; // r15
  __int64 v32; // rax
  unsigned __int64 *v33; // rdi
  unsigned __int64 *v34; // rdx
  int v35; // esi
  __int64 v36; // rax
  __int64 v37; // rsi
  unsigned __int8 *v38; // rax
  __int64 v39; // r12
  unsigned __int64 v40; // r15
  unsigned __int64 v41; // r14
  unsigned __int64 v42; // rax
  unsigned __int64 v43; // r14
  __int64 v44; // rax
  __int64 v45; // r8
  __int64 v46; // r9
  __int64 v47; // rbx
  __int64 v48; // rax
  __int64 *v49; // rax
  unsigned __int64 *v50; // rbx
  unsigned __int64 *v51; // r12
  unsigned __int64 v52; // rdi
  __int64 v54; // rdx
  bool v55; // al
  __int64 v56; // r11
  __int64 v57; // rax
  unsigned __int64 *v58; // rdi
  unsigned __int64 *v59; // rdx
  int v60; // esi
  __int64 v61; // rax
  __int64 v62; // rsi
  unsigned __int8 *v63; // rax
  unsigned int v64; // ecx
  __int64 v65; // [rsp+0h] [rbp-2F0h]
  __int64 v66; // [rsp+0h] [rbp-2F0h]
  __int64 v67; // [rsp+8h] [rbp-2E8h]
  __int64 v68; // [rsp+8h] [rbp-2E8h]
  __int64 v69; // [rsp+10h] [rbp-2E0h]
  __int64 v70; // [rsp+18h] [rbp-2D8h]
  __int64 v71; // [rsp+20h] [rbp-2D0h]
  __int64 v72; // [rsp+28h] [rbp-2C8h]
  __int64 v73; // [rsp+28h] [rbp-2C8h]
  __int64 v74; // [rsp+30h] [rbp-2C0h]
  __int64 *v75; // [rsp+30h] [rbp-2C0h]
  __int64 v76; // [rsp+38h] [rbp-2B8h]
  __int64 v77; // [rsp+40h] [rbp-2B0h]
  __int64 v78; // [rsp+40h] [rbp-2B0h]
  __int64 *v79; // [rsp+50h] [rbp-2A0h]
  __int64 v80; // [rsp+50h] [rbp-2A0h]
  unsigned __int8 *v81; // [rsp+50h] [rbp-2A0h]
  _QWORD *v82; // [rsp+50h] [rbp-2A0h]
  __int64 *v83; // [rsp+60h] [rbp-290h]
  __int64 v84; // [rsp+60h] [rbp-290h]
  __int64 v85; // [rsp+60h] [rbp-290h]
  int v86; // [rsp+60h] [rbp-290h]
  __int64 v87; // [rsp+78h] [rbp-278h] BYREF
  int v88[8]; // [rsp+80h] [rbp-270h] BYREF
  __int16 v89; // [rsp+A0h] [rbp-250h]
  __int64 *v90; // [rsp+B0h] [rbp-240h] BYREF
  __int64 v91; // [rsp+B8h] [rbp-238h]
  _BYTE v92[128]; // [rsp+C0h] [rbp-230h] BYREF
  _BYTE *v93; // [rsp+140h] [rbp-1B0h] BYREF
  __int64 v94; // [rsp+148h] [rbp-1A8h]
  _BYTE v95[128]; // [rsp+150h] [rbp-1A0h] BYREF
  unsigned __int64 *v96; // [rsp+1D0h] [rbp-120h] BYREF
  unsigned __int64 v97; // [rsp+1D8h] [rbp-118h] BYREF
  __int64 v98; // [rsp+1E0h] [rbp-110h] BYREF
  _BYTE v99[264]; // [rsp+1E8h] [rbp-108h] BYREF

  v5 = (__int64)v95;
  v6 = a2;
  v8 = *a2;
  v93 = v95;
  v79 = &v87;
  v87 = *(_QWORD *)(v8 + 72);
  v90 = (__int64 *)v92;
  v91 = 0x1000000000LL;
  v94 = 0x1000000000LL;
  v9 = *a1;
  if ( *(_DWORD *)(*a1 + 8) )
  {
    v77 = v8;
    v10 = 1;
    v11 = 0;
    while ( 1 )
    {
      v12 = v10 - 1;
      v13 = (__int64 *)(*(_QWORD *)v9 + 8 * v11);
      v14 = *v13;
      if ( *v13 )
      {
        if ( *(_QWORD *)(v14 + 152) )
        {
          v15 = *v6;
          v16 = *((unsigned int *)v6 + 4);
          v98 = 0;
          v97 = (unsigned __int64)v99;
          v96 = (unsigned __int64 *)v15;
          if ( (_DWORD)v16 )
          {
            sub_2506A60((__int64)&v97, (__int64)(v6 + 1), v15, v5, v16, (__int64)v99);
            v17 = *v13;
            if ( !*(_QWORD *)(v14 + 152) )
              sub_4263D6(&v97, v17, v54);
          }
          else
          {
            v17 = *v13;
          }
          (*(void (__fastcall **)(__int64, __int64, unsigned __int64 **, __int64 **))(v14 + 160))(
            v14 + 136,
            v17,
            &v96,
            &v90);
          v12 = (__int64)v99;
          if ( (_BYTE *)v97 != v99 )
            _libc_free(v97);
          v14 = *v13;
        }
        v18 = *(unsigned int *)(v14 + 32);
        v19 = (unsigned int)v94;
        v20 = v18;
        v21 = v94;
        if ( v18 + (unsigned __int64)(unsigned int)v94 > HIDWORD(v94) )
        {
          sub_C8D5F0((__int64)&v93, v95, v18 + (unsigned int)v94, 8u, a5, v12);
          v19 = (unsigned int)v94;
          v21 = v94;
        }
        v22 = &v93[8 * v19];
        if ( v18 )
        {
          do
          {
            if ( v22 )
              *v22 = 0;
            ++v22;
            --v18;
          }
          while ( v18 );
          v21 = v94;
        }
        v9 = *a1;
        v23 = v20 + v21;
        v5 = v10 + 1;
        v11 = v10;
        LODWORD(v94) = v23;
        if ( *(_DWORD *)(v9 + 8) <= v10 )
          goto LABEL_27;
      }
      else
      {
        if ( *((_DWORD *)v6 + 4) || (v76 = v11, v55 = sub_B491E0(*v6), v12 = v10 - 1, v55) )
        {
          v24 = 0;
          v25 = *(int *)(v6[1] + 4LL * v10);
          if ( (int)v25 >= 0 )
            v24 = *(_QWORD *)(*v6 + 32 * (v25 - (*(_DWORD *)(*v6 + 4) & 0x7FFFFFF)));
        }
        else
        {
          v24 = *(_QWORD *)(*v6 + 32 * (v76 - (*(_DWORD *)(*v6 + 4) & 0x7FFFFFF)));
        }
        v26 = (unsigned int)v91;
        v27 = (unsigned int)v91 + 1LL;
        if ( v27 > HIDWORD(v91) )
        {
          v86 = v12;
          sub_C8D5F0((__int64)&v90, v92, v27, 8u, a5, v12);
          v26 = (unsigned int)v91;
          LODWORD(v12) = v86;
        }
        v90[v26] = v24;
        LODWORD(v91) = v91 + 1;
        v28 = sub_A744E0(&v87, v12);
        v29 = (unsigned int)v94;
        v30 = (unsigned int)v94 + 1LL;
        if ( v30 > HIDWORD(v94) )
        {
          v85 = v28;
          sub_C8D5F0((__int64)&v93, v95, (unsigned int)v94 + 1LL, 8u, a5, v30);
          v29 = (unsigned int)v94;
          v28 = v85;
        }
        *(_QWORD *)&v93[8 * v29] = v28;
        v9 = *a1;
        v11 = v10;
        v5 = v10 + 1;
        LODWORD(v94) = v94 + 1;
        if ( *(_DWORD *)(v9 + 8) <= v10 )
        {
LABEL_27:
          v8 = v77;
          break;
        }
      }
      v10 = v5;
    }
  }
  v96 = (unsigned __int64 *)&v98;
  v97 = 0x400000000LL;
  sub_B56970(v8, (__int64)&v96);
  v78 = v8 + 24;
  if ( *(_BYTE *)v8 == 34 )
  {
    v31 = 0;
    v89 = 257;
    v83 = v90;
    v80 = (unsigned int)v91;
    v71 = *(_QWORD *)(v8 - 64);
    v72 = *(_QWORD *)(v8 - 96);
    v32 = *(_QWORD *)a1[1];
    v74 = v32;
    if ( v32 )
      v31 = *(_QWORD *)(v32 + 24);
    v33 = &v96[7 * (unsigned int)v97];
    if ( v96 == v33 )
    {
      v35 = 0;
    }
    else
    {
      v34 = v96;
      v35 = 0;
      do
      {
        v36 = v34[5] - v34[4];
        v34 += 7;
        v35 += v36 >> 3;
      }
      while ( v33 != v34 );
    }
    v37 = (unsigned int)(v91 + v35 + 3);
    v65 = (unsigned int)v97;
    LOBYTE(v6) = 16 * (_DWORD)v97 != 0;
    v67 = (__int64)v96;
    v38 = (unsigned __int8 *)sub_BD2CC0(88, ((unsigned __int64)(unsigned int)(16 * v97) << 32) | v37);
    v39 = (__int64)v38;
    if ( v38 )
    {
      v70 = v80;
      v81 = v38;
      sub_B44260((__int64)v38, **(_QWORD **)(v31 + 16), 5, v37 & 0x7FFFFFF | ((_DWORD)v6 << 28), v78, 0);
      *(_QWORD *)(v39 + 72) = 0;
      sub_B4A9C0(v39, v31, v74, v72, v71, (__int64)v88, v83, v70, v67, v65);
    }
    else
    {
      v81 = 0;
    }
  }
  else
  {
    v56 = 0;
    v89 = 257;
    v75 = v90;
    v73 = (unsigned int)v91;
    v57 = *(_QWORD *)a1[1];
    v84 = v57;
    if ( v57 )
      v56 = *(_QWORD *)(v57 + 24);
    v58 = &v96[7 * (unsigned int)v97];
    if ( v96 == v58 )
    {
      v60 = 0;
    }
    else
    {
      v59 = v96;
      v60 = 0;
      do
      {
        v61 = v59[5] - v59[4];
        v59 += 7;
        v60 += v61 >> 3;
      }
      while ( v58 != v59 );
    }
    v62 = (unsigned int)(v91 + v60 + 1);
    v66 = (unsigned int)v97;
    LOBYTE(v79) = 16 * (_DWORD)v97 != 0;
    v68 = (__int64)v96;
    v69 = v56;
    v63 = (unsigned __int8 *)sub_BD2CC0(88, ((unsigned __int64)(unsigned int)(16 * v97) << 32) | v62);
    v39 = (__int64)v63;
    if ( v63 )
    {
      v64 = v62 & 0x7FFFFFF | ((_DWORD)v79 << 28);
      v81 = v63;
      sub_B44260((__int64)v63, **(_QWORD **)(v69 + 16), 56, v64, v78, 0);
      *(_QWORD *)(v39 + 72) = 0;
      sub_B4A290(v39, v69, v84, v75, v73, (__int64)v88, v68, v66);
    }
    else
    {
      v81 = 0;
    }
    *(_WORD *)(v39 + 2) = *(_WORD *)(v8 + 2) & 3 | *(_WORD *)(v39 + 2) & 0xFFFC;
  }
  *(_QWORD *)v88 = 2;
  sub_B47C00((__int64)v81, v8, v88, 2);
  *(_WORD *)(v39 + 2) = *(_WORD *)(v8 + 2) & 0xFFC | *(_WORD *)(v39 + 2) & 0xF003;
  sub_BD6B90(v81, (unsigned __int8 *)v8);
  v40 = (unsigned int)v94;
  v82 = v93;
  v41 = sub_A74610(&v87);
  v42 = sub_A74680(&v87);
  *(_QWORD *)(v39 + 72) = sub_A78180((_QWORD *)a1[2], v42, v41, v82, v40);
  v43 = *(_QWORD *)a1[3];
  v44 = sub_B491C0(v39);
  sub_A75730(v44, v43);
  v47 = a1[4];
  v48 = *(unsigned int *)(v47 + 8);
  if ( v48 + 1 > (unsigned __int64)*(unsigned int *)(v47 + 12) )
  {
    sub_C8D5F0(v47, (const void *)(v47 + 16), v48 + 1, 0x10u, v45, v46);
    v48 = *(unsigned int *)(v47 + 8);
  }
  v49 = (__int64 *)(*(_QWORD *)v47 + 16 * v48);
  v49[1] = v39;
  *v49 = v8;
  ++*(_DWORD *)(v47 + 8);
  v50 = v96;
  v51 = &v96[7 * (unsigned int)v97];
  if ( v96 != v51 )
  {
    do
    {
      v52 = *(v51 - 3);
      v51 -= 7;
      if ( v52 )
        j_j___libc_free_0(v52);
      if ( (unsigned __int64 *)*v51 != v51 + 2 )
        j_j___libc_free_0(*v51);
    }
    while ( v50 != v51 );
    v51 = v96;
  }
  if ( v51 != (unsigned __int64 *)&v98 )
    _libc_free((unsigned __int64)v51);
  if ( v93 != v95 )
    _libc_free((unsigned __int64)v93);
  if ( v90 != (__int64 *)v92 )
    _libc_free((unsigned __int64)v90);
  return 1;
}
