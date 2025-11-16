// Function: sub_33F0610
// Address: 0x33f0610
//
__int64 __fastcall sub_33F0610(__int64 a1, int a2, __int64 a3, unsigned int a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 v9; // rbx
  unsigned __int16 *v10; // rcx
  __int64 *v11; // rsi
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // r14
  __int64 (__fastcall *v15)(__int64, __int64, unsigned int); // r12
  __int64 v16; // rax
  int v17; // edx
  unsigned __int16 v18; // ax
  __int64 v19; // rax
  __int64 v20; // r14
  __int64 v21; // rsi
  int v22; // edx
  _QWORD *v23; // rdi
  __int64 v24; // rax
  __int64 v25; // rdx
  unsigned __int64 v26; // rcx
  unsigned __int64 v27; // rdi
  unsigned int v28; // r15d
  __int64 v29; // rax
  void (***v30)(); // rdi
  void (*v31)(); // rax
  __int64 v32; // r14
  int v34; // [rsp+0h] [rbp-1230h]
  int v35; // [rsp+8h] [rbp-1228h]
  unsigned __int64 v36; // [rsp+60h] [rbp-11D0h] BYREF
  __int64 v37; // [rsp+68h] [rbp-11C8h]
  __int64 v38; // [rsp+70h] [rbp-11C0h]
  char v39[16]; // [rsp+80h] [rbp-11B0h] BYREF
  __int64 v40; // [rsp+90h] [rbp-11A0h]
  __m128i v41; // [rsp+A0h] [rbp-1190h] BYREF
  unsigned int v42; // [rsp+B0h] [rbp-1180h]
  __int64 v43; // [rsp+B8h] [rbp-1178h]
  __int128 v44; // [rsp+C0h] [rbp-1170h]
  __int64 v45; // [rsp+D0h] [rbp-1160h] BYREF
  __int64 v46; // [rsp+D8h] [rbp-1158h]
  __int64 v47; // [rsp+E0h] [rbp-1150h]
  unsigned __int64 v48; // [rsp+E8h] [rbp-1148h]
  __int64 v49; // [rsp+F0h] [rbp-1140h]
  __int64 v50; // [rsp+F8h] [rbp-1138h]
  __int64 v51; // [rsp+100h] [rbp-1130h]
  unsigned __int64 v52; // [rsp+108h] [rbp-1128h] BYREF
  __int64 v53; // [rsp+110h] [rbp-1120h]
  __int64 v54; // [rsp+118h] [rbp-1118h]
  __int64 v55; // [rsp+120h] [rbp-1110h]
  __int64 v56; // [rsp+128h] [rbp-1108h] BYREF
  int v57; // [rsp+130h] [rbp-1100h]
  __int64 v58; // [rsp+138h] [rbp-10F8h]
  _BYTE *v59; // [rsp+140h] [rbp-10F0h]
  __int64 v60; // [rsp+148h] [rbp-10E8h]
  _BYTE v61[1792]; // [rsp+150h] [rbp-10E0h] BYREF
  _BYTE *v62; // [rsp+850h] [rbp-9E0h]
  __int64 v63; // [rsp+858h] [rbp-9D8h]
  _BYTE v64[512]; // [rsp+860h] [rbp-9D0h] BYREF
  _BYTE *v65; // [rsp+A60h] [rbp-7D0h]
  __int64 v66; // [rsp+A68h] [rbp-7C8h]
  _BYTE v67[1792]; // [rsp+A70h] [rbp-7C0h] BYREF
  _BYTE *v68; // [rsp+1170h] [rbp-C0h]
  __int64 v69; // [rsp+1178h] [rbp-B8h]
  _BYTE v70[64]; // [rsp+1180h] [rbp-B0h] BYREF
  __int64 v71; // [rsp+11C0h] [rbp-70h]
  __int64 v72; // [rsp+11C8h] [rbp-68h]
  int v73; // [rsp+11D0h] [rbp-60h]
  char v74; // [rsp+11F0h] [rbp-40h]

  v9 = a2;
  v42 = a4;
  v10 = (unsigned __int16 *)(*(_QWORD *)(a3 + 48) + 16LL * a4);
  v37 = 0;
  v11 = *(__int64 **)(a1 + 64);
  v38 = 0;
  v43 = 0;
  v44 = 0u;
  v41.m128i_i64[1] = a3;
  v12 = *((_QWORD *)v10 + 1);
  v36 = 0;
  v41.m128i_i64[0] = 0;
  v13 = *v10;
  v46 = v12;
  v34 = a6;
  LOWORD(v45) = v13;
  v43 = sub_3007410((__int64)&v45, v11, v13, (__int64)v10, a5, a6);
  sub_332CDC0(&v36, 0, &v41);
  v14 = *(_QWORD *)(a1 + 16);
  v15 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v14 + 32LL);
  v16 = sub_2E79000(*(__int64 **)(a1 + 40));
  if ( v15 == sub_2D42F30 )
  {
    v17 = sub_AE2980(v16, 0)[1];
    v18 = 2;
    if ( v17 != 1 )
    {
      v18 = 3;
      if ( v17 != 2 )
      {
        v18 = 4;
        if ( v17 != 4 )
        {
          v18 = 5;
          if ( v17 != 8 )
          {
            v18 = 6;
            if ( v17 != 16 )
            {
              v18 = 7;
              if ( v17 != 32 )
              {
                v18 = 8;
                if ( v17 != 64 )
                  v18 = 9 * (v17 == 128);
              }
            }
          }
        }
      }
    }
  }
  else
  {
    v18 = v15(v14, v16, 0);
  }
  v19 = sub_33EED90(a1, *(const char **)(*(_QWORD *)(a1 + 16) + 8 * v9 + 525288), v18, 0);
  v45 = 0;
  v20 = v19;
  v48 = 0xFFFFFFFF00000020LL;
  v59 = v61;
  v60 = 0x2000000000LL;
  v63 = 0x2000000000LL;
  v66 = 0x2000000000LL;
  v69 = 0x400000000LL;
  v62 = v64;
  v21 = *(_QWORD *)a7;
  v74 = 0;
  v35 = v22;
  v46 = 0;
  v47 = 0;
  v49 = 0;
  v50 = 0;
  v51 = 0;
  v52 = 0;
  v53 = 0;
  v54 = 0;
  v55 = a1;
  v57 = 0;
  v58 = 0;
  v65 = v67;
  v68 = v70;
  v71 = 0;
  v72 = 0;
  v73 = 0;
  v56 = v21;
  if ( v21 )
    sub_B96E90((__int64)&v56, v21, 1);
  v45 = a5;
  v23 = *(_QWORD **)(a1 + 64);
  v57 = *(_DWORD *)(a7 + 8);
  LODWORD(v46) = v34;
  v24 = sub_BCB120(v23);
  v25 = *(_QWORD *)(a1 + 16);
  v26 = v36;
  v27 = v52;
  v36 = 0;
  v28 = *(_DWORD *)(v25 + 4 * v9 + 531128);
  v47 = v24;
  v50 = v20;
  LODWORD(v51) = v35;
  LODWORD(v49) = v28;
  v52 = v26;
  LODWORD(v24) = -1431655765 * ((__int64)(v37 - v26) >> 4);
  v53 = v37;
  v37 = 0;
  HIDWORD(v48) = v24;
  v29 = v38;
  v38 = 0;
  v54 = v29;
  if ( v27 )
    j_j___libc_free_0(v27);
  v30 = *(void (****)())(v55 + 16);
  v31 = **v30;
  if ( v31 != nullsub_1688 )
    ((void (__fastcall *)(void (***)(), _QWORD, _QWORD, unsigned __int64 *))v31)(v30, *(_QWORD *)(v55 + 40), v28, &v52);
  sub_3377410((__int64)v39, *(_WORD **)(a1 + 16), (__int64)&v45);
  v32 = v40;
  if ( v68 != v70 )
    _libc_free((unsigned __int64)v68);
  if ( v65 != v67 )
    _libc_free((unsigned __int64)v65);
  if ( v62 != v64 )
    _libc_free((unsigned __int64)v62);
  if ( v59 != v61 )
    _libc_free((unsigned __int64)v59);
  if ( v56 )
    sub_B91220((__int64)&v56, v56);
  if ( v52 )
    j_j___libc_free_0(v52);
  if ( v36 )
    j_j___libc_free_0(v36);
  return v32;
}
