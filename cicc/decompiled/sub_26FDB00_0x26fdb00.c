// Function: sub_26FDB00
// Address: 0x26fdb00
//
__int64 __fastcall sub_26FDB00(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rdx
  __int64 v6; // rbx
  _BYTE *v8; // r13
  __int64 v9; // rax
  __int64 v10; // r15
  __int64 *v11; // r14
  __int64 v12; // rax
  __int64 v13; // rsi
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // r8
  __int64 v17; // rdx
  unsigned __int64 v18; // r9
  unsigned int v19; // r15d
  __int64 v20; // rcx
  __int64 v21; // rdi
  __int64 v22; // rax
  __int64 v23; // r9
  __int64 v24; // rdx
  unsigned __int64 v25; // r8
  _QWORD *v26; // rax
  _BYTE *v27; // rdi
  __int64 v29; // [rsp+0h] [rbp-310h]
  __int64 v30; // [rsp+0h] [rbp-310h]
  __int64 v31; // [rsp+8h] [rbp-308h]
  __int64 v32; // [rsp+18h] [rbp-2F8h]
  __int64 v35; // [rsp+38h] [rbp-2D8h] BYREF
  _BYTE *v36; // [rsp+40h] [rbp-2D0h] BYREF
  __int64 v37; // [rsp+48h] [rbp-2C8h]
  _BYTE v38[16]; // [rsp+50h] [rbp-2C0h] BYREF
  __int64 v39; // [rsp+60h] [rbp-2B0h] BYREF
  __int64 v40; // [rsp+68h] [rbp-2A8h]
  __int64 v41; // [rsp+70h] [rbp-2A0h]
  __int64 v42; // [rsp+78h] [rbp-298h]
  __int64 v43; // [rsp+80h] [rbp-290h]
  __int64 *v44; // [rsp+88h] [rbp-288h]
  __int64 v45; // [rsp+90h] [rbp-280h]
  __int64 v46; // [rsp+98h] [rbp-278h]
  __int64 v47; // [rsp+A0h] [rbp-270h]
  __int64 *v48; // [rsp+A8h] [rbp-268h]
  char *v49; // [rsp+B0h] [rbp-260h]
  __int64 v50; // [rsp+B8h] [rbp-258h]
  char v51; // [rsp+C0h] [rbp-250h] BYREF
  __int64 v52; // [rsp+E0h] [rbp-230h]
  __int64 v53; // [rsp+E8h] [rbp-228h]
  __int64 v54; // [rsp+F0h] [rbp-220h]
  int v55; // [rsp+F8h] [rbp-218h]
  char *v56; // [rsp+100h] [rbp-210h]
  __int64 v57; // [rsp+108h] [rbp-208h]
  char v58; // [rsp+110h] [rbp-200h] BYREF
  __int64 v59; // [rsp+210h] [rbp-100h]
  char *v60; // [rsp+218h] [rbp-F8h]
  __int64 v61; // [rsp+220h] [rbp-F0h]
  int v62; // [rsp+228h] [rbp-E8h]
  char v63; // [rsp+22Ch] [rbp-E4h]
  char v64; // [rsp+230h] [rbp-E0h] BYREF
  __int64 v65; // [rsp+270h] [rbp-A0h]
  char *v66; // [rsp+278h] [rbp-98h]
  __int64 v67; // [rsp+280h] [rbp-90h]
  int v68; // [rsp+288h] [rbp-88h]
  char v69; // [rsp+28Ch] [rbp-84h]
  char v70; // [rsp+290h] [rbp-80h] BYREF
  __int64 v71; // [rsp+2D0h] [rbp-40h]
  __int64 v72; // [rsp+2D8h] [rbp-38h]

  v5 = 32 * a3;
  v31 = a2 + v5;
  if ( a2 == a2 + v5 )
    return 1;
  v6 = a2;
  v32 = a5 + 1;
  while ( 1 )
  {
    v8 = *(_BYTE **)v6;
    if ( **(_BYTE **)v6 || v32 != *((_QWORD *)v8 + 13) )
      return 0;
    v39 = 0;
    v41 = 0;
    v42 = 0;
    v9 = *a1;
    v43 = 0;
    v10 = v9 + 312;
    v44 = 0;
    v45 = 0;
    v46 = 0;
    v47 = 0;
    v48 = 0;
    v40 = 8;
    v39 = sub_22077B0(0x40u);
    v11 = (__int64 *)(v39 + ((4 * v40 - 4) & 0xFFFFFFFFFFFFFFF8LL));
    v12 = sub_22077B0(0x200u);
    v44 = v11;
    v50 = 0x400000000LL;
    v13 = 0x2000000000LL;
    v43 = v12 + 512;
    v47 = v12 + 512;
    v49 = &v51;
    v56 = &v58;
    v60 = &v64;
    *v11 = v12;
    v42 = v12;
    v48 = v11;
    v46 = v12;
    v41 = v12;
    v52 = 0;
    v53 = 0;
    v54 = 0;
    v55 = 0;
    v57 = 0x2000000000LL;
    v59 = 0;
    v61 = 8;
    v62 = 0;
    v63 = 1;
    v65 = 0;
    v66 = &v70;
    v67 = 8;
    v68 = 0;
    v69 = 1;
    v71 = v10;
    v72 = 0;
    if ( v12 )
    {
      *(_QWORD *)v12 = 0;
      *(_QWORD *)(v12 + 8) = 0;
      *(_QWORD *)(v12 + 16) = 0;
      *(_DWORD *)(v12 + 24) = 0;
    }
    v45 = v12 + 32;
    v37 = 0x200000000LL;
    v14 = *((_QWORD *)v8 + 3);
    v36 = v38;
    v15 = sub_AD6530(*(_QWORD *)(*(_QWORD *)(v14 + 16) + 8LL), 0x2000000000LL);
    v17 = (unsigned int)v37;
    v18 = (unsigned int)v37 + 1LL;
    if ( v18 > HIDWORD(v37) )
    {
      v13 = (__int64)v38;
      v30 = v15;
      sub_C8D5F0((__int64)&v36, v38, (unsigned int)v37 + 1LL, 8u, v16, v18);
      v17 = (unsigned int)v37;
      v15 = v30;
    }
    v19 = 0;
    *(_QWORD *)&v36[8 * v17] = v15;
    v20 = 0;
    LODWORD(v37) = v37 + 1;
    if ( a5 )
    {
      while ( 1 )
      {
        v21 = *(_QWORD *)(*(_QWORD *)(*((_QWORD *)v8 + 3) + 16LL) + 8LL * (v19 + 2));
        if ( *(_BYTE *)(v21 + 8) != 12 )
          break;
        v13 = *(_QWORD *)(a4 + 8 * v20);
        v22 = sub_ACD640(v21, v13, 0);
        v24 = (unsigned int)v37;
        v25 = (unsigned int)v37 + 1LL;
        if ( v25 > HIDWORD(v37) )
        {
          v13 = (__int64)v38;
          v29 = v22;
          sub_C8D5F0((__int64)&v36, v38, (unsigned int)v37 + 1LL, 8u, v25, v23);
          v24 = (unsigned int)v37;
          v22 = v29;
        }
        ++v19;
        *(_QWORD *)&v36[8 * v24] = v22;
        v20 = v19;
        LODWORD(v37) = v37 + 1;
        if ( v19 == a5 )
          goto LABEL_14;
      }
LABEL_22:
      if ( v36 != v38 )
        _libc_free((unsigned __int64)v36);
      sub_25DE530((__int64)&v39, v13);
      return 0;
    }
LABEL_14:
    v13 = (__int64)v8;
    if ( !(unsigned __int8)sub_29D2770(&v39, v8, &v35, &v36) || *(_BYTE *)v35 != 17 )
      goto LABEL_22;
    v26 = *(_QWORD **)(v35 + 24);
    if ( *(_DWORD *)(v35 + 32) > 0x40u )
      v26 = (_QWORD *)*v26;
    v27 = v36;
    *(_QWORD *)(v6 + 16) = v26;
    if ( v27 != v38 )
      _libc_free((unsigned __int64)v27);
    v6 += 32;
    sub_25DE530((__int64)&v39, (__int64)v8);
    if ( v31 == v6 )
      return 1;
  }
}
