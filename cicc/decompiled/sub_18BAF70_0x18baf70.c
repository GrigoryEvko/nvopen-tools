// Function: sub_18BAF70
// Address: 0x18baf70
//
__int64 __fastcall sub_18BAF70(
        __int64 *a1,
        __int64 *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __m128 a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        double a12,
        __m128 a13)
{
  __int64 *v13; // r14
  __int64 v16; // rax
  __int64 v17; // r15
  __int64 *v18; // r13
  __int64 v19; // rax
  __int64 v20; // rsi
  __int64 v21; // rax
  int v22; // r8d
  int v23; // r9d
  __int64 v24; // r15
  double v25; // xmm4_8
  double v26; // xmm5_8
  __int64 v27; // rax
  __int64 v28; // rcx
  int v29; // r15d
  __int64 v30; // rdx
  __int64 v31; // rdi
  __int64 v32; // r8
  int v33; // r9d
  __int64 v34; // rax
  _QWORD *v35; // rax
  _BYTE *v36; // rdi
  __int64 v38; // [rsp+0h] [rbp-320h]
  __int64 *v39; // [rsp+8h] [rbp-318h]
  __int64 v41; // [rsp+28h] [rbp-2F8h]
  __int64 v42; // [rsp+38h] [rbp-2E8h] BYREF
  _BYTE *v43; // [rsp+40h] [rbp-2E0h] BYREF
  __int64 v44; // [rsp+48h] [rbp-2D8h]
  _BYTE v45[16]; // [rsp+50h] [rbp-2D0h] BYREF
  __int64 v46; // [rsp+60h] [rbp-2C0h] BYREF
  __int64 v47; // [rsp+68h] [rbp-2B8h]
  __int64 v48; // [rsp+70h] [rbp-2B0h]
  __int64 v49; // [rsp+78h] [rbp-2A8h]
  __int64 v50; // [rsp+80h] [rbp-2A0h]
  __int64 *v51; // [rsp+88h] [rbp-298h]
  __int64 v52; // [rsp+90h] [rbp-290h]
  __int64 v53; // [rsp+98h] [rbp-288h]
  __int64 v54; // [rsp+A0h] [rbp-280h]
  __int64 *v55; // [rsp+A8h] [rbp-278h]
  char *v56; // [rsp+B0h] [rbp-270h]
  __int64 v57; // [rsp+B8h] [rbp-268h]
  char v58; // [rsp+C0h] [rbp-260h] BYREF
  __int64 v59; // [rsp+E0h] [rbp-240h]
  __int64 v60; // [rsp+E8h] [rbp-238h]
  __int64 v61; // [rsp+F0h] [rbp-230h]
  int v62; // [rsp+F8h] [rbp-228h]
  char *v63; // [rsp+100h] [rbp-220h]
  __int64 v64; // [rsp+108h] [rbp-218h]
  char v65; // [rsp+110h] [rbp-210h] BYREF
  __int64 v66; // [rsp+210h] [rbp-110h]
  _BYTE *v67; // [rsp+218h] [rbp-108h]
  _BYTE *v68; // [rsp+220h] [rbp-100h]
  __int64 v69; // [rsp+228h] [rbp-F8h]
  int v70; // [rsp+230h] [rbp-F0h]
  _BYTE v71[64]; // [rsp+238h] [rbp-E8h] BYREF
  __int64 v72; // [rsp+278h] [rbp-A8h]
  _BYTE *v73; // [rsp+280h] [rbp-A0h]
  _BYTE *v74; // [rsp+288h] [rbp-98h]
  __int64 v75; // [rsp+290h] [rbp-90h]
  int v76; // [rsp+298h] [rbp-88h]
  _BYTE v77[64]; // [rsp+2A0h] [rbp-80h] BYREF
  __int64 v78; // [rsp+2E0h] [rbp-40h]
  __int64 v79; // [rsp+2E8h] [rbp-38h]

  v13 = a2;
  v39 = &a2[4 * a3];
  v41 = a5 + 1;
  if ( a2 == v39 )
    return 1;
  while ( *(_QWORD *)(*v13 + 96) == v41 )
  {
    v16 = sub_1632FA0(*a1);
    v46 = 0;
    v48 = 0;
    v17 = v16;
    v49 = 0;
    v50 = 0;
    v51 = 0;
    v52 = 0;
    v53 = 0;
    v54 = 0;
    v55 = 0;
    v47 = 8;
    v46 = sub_22077B0(64);
    v18 = (__int64 *)(v46 + ((4 * v47 - 4) & 0xFFFFFFFFFFFFFFF8LL));
    v19 = sub_22077B0(512);
    v51 = v18;
    *v18 = v19;
    v50 = v19 + 512;
    v54 = v19 + 512;
    v63 = &v65;
    v56 = &v58;
    v20 = 0x2000000000LL;
    v67 = v71;
    v68 = v71;
    v49 = v19;
    v55 = v18;
    v53 = v19;
    v48 = v19;
    v57 = 0x400000000LL;
    v59 = 0;
    v60 = 0;
    v61 = 0;
    v62 = 0;
    v64 = 0x2000000000LL;
    v66 = 0;
    v69 = 8;
    v70 = 0;
    v72 = 0;
    v73 = v77;
    v74 = v77;
    v75 = 8;
    v76 = 0;
    v78 = v17;
    v79 = 0;
    if ( v19 )
    {
      *(_QWORD *)v19 = 0;
      *(_QWORD *)(v19 + 8) = 0;
      *(_QWORD *)(v19 + 16) = 0;
      *(_DWORD *)(v19 + 24) = 0;
    }
    v52 = v19 + 32;
    v44 = 0x200000000LL;
    v21 = *v13;
    v43 = v45;
    v24 = sub_15A06D0(
            *(__int64 ***)(*(_QWORD *)(*(_QWORD *)(v21 + 24) + 16LL) + 8LL),
            0x2000000000LL,
            (__int64)v77,
            0x400000000LL);
    v27 = (unsigned int)v44;
    if ( (unsigned int)v44 >= HIDWORD(v44) )
    {
      v20 = (__int64)v45;
      sub_16CD150((__int64)&v43, v45, 0, 8, v22, v23);
      v27 = (unsigned int)v44;
    }
    v28 = 0;
    *(_QWORD *)&v43[8 * v27] = v24;
    v29 = 0;
    LODWORD(v44) = v44 + 1;
    if ( a5 )
    {
      while ( 1 )
      {
        v30 = *(_QWORD *)(*(_QWORD *)(*v13 + 24) + 16LL);
        v31 = *(_QWORD *)(v30 + 8LL * (unsigned int)(v29 + 2));
        if ( *(_BYTE *)(v31 + 8) != 11 )
          break;
        v20 = *(_QWORD *)(a4 + 8 * v28);
        v32 = sub_159C470(v31, v20, 0);
        v34 = (unsigned int)v44;
        if ( (unsigned int)v44 >= HIDWORD(v44) )
        {
          v20 = (__int64)v45;
          v38 = v32;
          sub_16CD150((__int64)&v43, v45, 0, 8, v32, v33);
          v34 = (unsigned int)v44;
          v32 = v38;
        }
        v28 = (unsigned int)++v29;
        *(_QWORD *)&v43[8 * v34] = v32;
        LODWORD(v44) = v44 + 1;
        if ( v29 == a5 )
          goto LABEL_12;
      }
LABEL_20:
      if ( v43 != v45 )
        _libc_free((unsigned __int64)v43);
      sub_185D1A0((__int64)&v46, v20, v30, v28, a6, a7, a8, a9, v25, v26, a12, a13);
      return 0;
    }
LABEL_12:
    v20 = *v13;
    if ( !(unsigned __int8)sub_1AC8A90(&v46, *v13, &v42, &v43) )
      goto LABEL_20;
    v30 = v42;
    if ( *(_BYTE *)(v42 + 16) != 13 )
      goto LABEL_20;
    v35 = *(_QWORD **)(v42 + 24);
    if ( *(_DWORD *)(v42 + 32) > 0x40u )
      v35 = (_QWORD *)*v35;
    v36 = v43;
    v13[2] = (__int64)v35;
    if ( v36 != v45 )
      _libc_free((unsigned __int64)v36);
    v13 += 4;
    sub_185D1A0((__int64)&v46, v20, v30, v28, a6, a7, a8, a9, v25, v26, a12, a13);
    if ( v39 == v13 )
      return 1;
  }
  return 0;
}
