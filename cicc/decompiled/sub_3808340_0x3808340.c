// Function: sub_3808340
// Address: 0x3808340
//
__int64 __fastcall sub_3808340(__int64 *a1, unsigned __int64 a2, int a3, __int64 a4)
{
  __int64 v4; // r13
  __int64 v5; // r15
  __int64 v6; // r10
  __int64 v9; // r9
  __int16 *v10; // rax
  unsigned __int16 v11; // si
  __int64 v12; // r8
  __int64 (__fastcall *v13)(__int64, __int64, unsigned int, __int64); // rax
  __int64 (__fastcall *v14)(__int64, __int64, unsigned int); // r14
  __int64 v15; // rsi
  __int64 v16; // r8
  unsigned __int64 v17; // r9
  int v18; // eax
  __int64 v19; // rdx
  __int64 v20; // rdi
  unsigned int v21; // r15d
  _QWORD *v22; // rax
  __int64 v23; // r9
  __int64 v24; // rdx
  __int64 v25; // rdx
  _QWORD *v26; // rdx
  __int64 v27; // rdx
  __int64 v28; // r11
  __int64 v29; // rdx
  _QWORD *v30; // rdx
  __int64 v31; // rdx
  _QWORD *v32; // r11
  unsigned __int64 v33; // r10
  _QWORD *v34; // rcx
  __int64 v35; // rax
  _WORD *v36; // rsi
  __int64 *v37; // r15
  __int64 v38; // r14
  __int64 *v39; // rdi
  __m128i *v40; // rax
  __int64 v41; // rdx
  __int64 (__fastcall *v43)(__int64, __int64, unsigned int); // rdx
  __int64 v44; // [rsp+8h] [rbp-228h]
  _QWORD *v45; // [rsp+10h] [rbp-220h]
  __int64 v46; // [rsp+10h] [rbp-220h]
  __int64 v47; // [rsp+18h] [rbp-218h]
  _QWORD *v48; // [rsp+18h] [rbp-218h]
  __int64 v49; // [rsp+20h] [rbp-210h]
  unsigned __int8 v50; // [rsp+3Ah] [rbp-1F6h]
  unsigned __int16 v52; // [rsp+40h] [rbp-1F0h]
  __int64 v53; // [rsp+48h] [rbp-1E8h]
  __int64 v54; // [rsp+50h] [rbp-1E0h]
  __int64 v55; // [rsp+50h] [rbp-1E0h]
  unsigned int v56; // [rsp+68h] [rbp-1C8h]
  __int64 v58; // [rsp+90h] [rbp-1A0h] BYREF
  int v59; // [rsp+98h] [rbp-198h]
  __int128 v60; // [rsp+A0h] [rbp-190h] BYREF
  __int64 v61; // [rsp+B0h] [rbp-180h]
  __int64 v62; // [rsp+C0h] [rbp-170h] BYREF
  _DWORD v63[2]; // [rsp+C8h] [rbp-168h]
  __int64 v64; // [rsp+D0h] [rbp-160h]
  int v65; // [rsp+D8h] [rbp-158h]
  unsigned __int64 v66[2]; // [rsp+E0h] [rbp-150h] BYREF
  __m128i v67; // [rsp+F0h] [rbp-140h] BYREF
  _QWORD v68[6]; // [rsp+100h] [rbp-130h] BYREF
  __int64 v69; // [rsp+130h] [rbp-100h]
  __int64 v70; // [rsp+138h] [rbp-F8h]
  __int64 v71; // [rsp+140h] [rbp-F0h]
  __m128i v72; // [rsp+150h] [rbp-E0h]
  _QWORD *v73; // [rsp+180h] [rbp-B0h] BYREF
  __int64 v74; // [rsp+188h] [rbp-A8h]
  _QWORD v75[6]; // [rsp+190h] [rbp-A0h] BYREF
  _QWORD *v76; // [rsp+1C0h] [rbp-70h] BYREF
  __int64 v77; // [rsp+1C8h] [rbp-68h]
  _QWORD v78[12]; // [rsp+1D0h] [rbp-60h] BYREF

  v6 = *a1;
  if ( *(_QWORD *)(*a1 + 8LL * a3 + 525288) )
  {
    v9 = *(_QWORD *)(a1[1] + 64);
    v10 = *(__int16 **)(a2 + 48);
    v11 = *v10;
    v12 = *((_QWORD *)v10 + 1);
    v52 = *v10;
    v13 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v6 + 592LL);
    v54 = v12;
    if ( v13 == sub_2D56A50 )
    {
      sub_2FE6CC0((__int64)&v76, v6, v9, v11, v12);
      v14 = (__int64 (__fastcall *)(__int64, __int64, unsigned int))v78[0];
      v56 = (unsigned __int16)v77;
    }
    else
    {
      v5 = v52;
      v56 = v13(v6, v9, v52, v12);
      v14 = v43;
    }
    v15 = *(_QWORD *)(a2 + 80);
    v58 = v15;
    if ( v15 )
      sub_B96E90((__int64)&v58, v15, 1);
    v59 = *(_DWORD *)(a2 + 72);
    LOWORD(v5) = v52;
    v75[0] = sub_3805E70((__int64)a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
    v74 = 0x300000001LL;
    v77 = 0x300000001LL;
    v18 = *(_DWORD *)(a2 + 68);
    v73 = v75;
    v75[1] = v19;
    v76 = v78;
    v78[0] = v5;
    v78[1] = v54;
    v62 = 0;
    v63[0] = 0;
    v64 = 0;
    v65 = 0;
    if ( v18 )
    {
      v20 = a1[1];
      v21 = 0;
      do
      {
        if ( (_DWORD)a4 != v21 || !BYTE4(a4) )
        {
          v22 = sub_33EDFE0(v20, v56, (__int64)v14, 1, v16, v17);
          v16 = v24;
          v25 = (unsigned int)v74;
          if ( (unsigned __int64)(unsigned int)v74 + 1 > HIDWORD(v74) )
          {
            v46 = v16;
            v48 = v22;
            sub_C8D5F0((__int64)&v73, v75, (unsigned int)v74 + 1LL, 0x10u, v16, v23);
            v25 = (unsigned int)v74;
            v16 = v46;
            v22 = v48;
          }
          v26 = &v73[2 * v25];
          *v26 = v22;
          v26[1] = v16;
          v27 = v22[6] + 16LL * (unsigned int)v16;
          LODWORD(v74) = v74 + 1;
          v28 = *(_QWORD *)(v27 + 8);
          LOWORD(v4) = *(_WORD *)v27;
          v29 = (unsigned int)v77;
          v17 = (unsigned int)v77 + 1LL;
          if ( v17 > HIDWORD(v77) )
          {
            v44 = v16;
            v45 = v22;
            v47 = v28;
            sub_C8D5F0((__int64)&v76, v78, (unsigned int)v77 + 1LL, 0x10u, v16, v17);
            v29 = (unsigned int)v77;
            v16 = v44;
            v22 = v45;
            v28 = v47;
          }
          v30 = &v76[2 * v29];
          *v30 = v4;
          v30[1] = v28;
          v31 = 4LL * v21;
          v20 = a1[1];
          *(_QWORD *)&v63[v31 - 2] = v22;
          LODWORD(v77) = v77 + 1;
          v63[v31] = v16;
        }
        ++v21;
      }
      while ( v21 < *(_DWORD *)(a2 + 68) );
      v32 = v73;
      v33 = (unsigned int)v74;
      v34 = v76;
      v35 = (unsigned int)v77;
    }
    else
    {
      v34 = v78;
      v20 = a1[1];
      v35 = 1;
      v33 = 1;
      v32 = v75;
    }
    v68[5] = v35;
    v36 = (_WORD *)*a1;
    LOBYTE(v71) = 20;
    v68[4] = v34;
    v37 = &v62;
    LOWORD(v69) = v52;
    v70 = v54;
    sub_3494590(
      (__int64)v66,
      v36,
      v20,
      a3,
      v56,
      v14,
      (__int64)v32,
      v33,
      (__int64)v34,
      v35,
      v69,
      v54,
      20,
      (__int64)&v58,
      0,
      0);
    v49 = (__int64)v14;
    v38 = 0;
    v72 = _mm_load_si128(&v67);
    while ( 1 )
    {
      if ( BYTE4(a4) && (unsigned int)a4 == v38 )
      {
        sub_375F330((__int64)a1, a2, v38 & 1, v66[0], v66[1]);
      }
      else
      {
        v55 = *v37;
        v53 = v37[1];
        sub_2EAC300((__int64)&v60, *(_QWORD *)(a1[1] + 40), *(_DWORD *)(*v37 + 96), 0);
        v39 = (__int64 *)a1[1];
        memset(v68, 0, 32);
        v40 = sub_33F1F00(
                v39,
                v56,
                v49,
                (__int64)&v58,
                v72.m128i_i64[0],
                v72.m128i_i64[1],
                v55,
                v53,
                v60,
                v61,
                v50,
                0,
                (__int64)v68,
                0);
        sub_375F330((__int64)a1, a2, v38 & 1, (unsigned __int64)v40, v41);
      }
      v37 += 2;
      if ( v38 == 1 )
        break;
      v38 = 1;
    }
    if ( v76 != v78 )
      _libc_free((unsigned __int64)v76);
    if ( v73 != v75 )
      _libc_free((unsigned __int64)v73);
    if ( v58 )
      sub_B91220((__int64)&v58, v58);
  }
  return 0;
}
