// Function: sub_2654C20
// Address: 0x2654c20
//
unsigned __int8 *__fastcall sub_2654C20(__int64 a1, __int64 *a2, _QWORD *a3, __int64 *a4, __int32 a5, __m128i a6)
{
  unsigned int v8; // eax
  __int64 v9; // rcx
  const char *v10; // rax
  __int64 v11; // rdx
  __int64 v12; // r8
  __int64 v13; // r9
  _QWORD *v14; // rax
  __int64 v15; // r13
  _QWORD *v16; // r14
  __int64 v17; // rax
  bool v18; // zf
  unsigned __int64 *v19; // rax
  __int64 v20; // rdx
  _QWORD *v21; // rax
  __int64 v22; // r8
  __int64 v23; // rax
  unsigned __int64 v24; // rdx
  __int64 v25; // rsi
  __int64 v26; // rax
  __int64 *v27; // r13
  __int64 v28; // rax
  unsigned __int64 *v29; // rbx
  unsigned __int64 *v30; // r12
  unsigned __int64 v31; // rdi
  unsigned int v32; // eax
  int v34; // esi
  int v35; // edx
  __int64 v36; // rsi
  unsigned __int64 *v37; // r8
  __int64 *v38; // rsi
  _QWORD *v39; // rbx
  _QWORD *v40; // r12
  __int64 v41; // rax
  __int64 v42; // rdx
  unsigned int v43; // eax
  _QWORD *v44; // rbx
  _QWORD *v45; // r12
  __int64 v46; // rsi
  unsigned __int64 *v47; // [rsp+0h] [rbp-320h]
  unsigned __int8 *v49; // [rsp+28h] [rbp-2F8h]
  __int64 v50; // [rsp+38h] [rbp-2E8h]
  _QWORD *v52; // [rsp+50h] [rbp-2D0h]
  __int64 v53; // [rsp+50h] [rbp-2D0h]
  unsigned __int64 *v54; // [rsp+50h] [rbp-2D0h]
  unsigned __int64 *v55; // [rsp+50h] [rbp-2D0h]
  __int64 v56; // [rsp+58h] [rbp-2C8h]
  unsigned __int8 *v57; // [rsp+70h] [rbp-2B0h] BYREF
  __int64 v58; // [rsp+78h] [rbp-2A8h]
  unsigned __int64 v59[4]; // [rsp+80h] [rbp-2A0h] BYREF
  __int64 v60; // [rsp+A0h] [rbp-280h] BYREF
  _QWORD *v61; // [rsp+A8h] [rbp-278h]
  __int64 v62; // [rsp+B0h] [rbp-270h]
  unsigned int v63; // [rsp+B8h] [rbp-268h]
  _QWORD *v64; // [rsp+C8h] [rbp-258h]
  unsigned int v65; // [rsp+D8h] [rbp-248h]
  char v66; // [rsp+E0h] [rbp-240h]
  unsigned __int8 *v67; // [rsp+F0h] [rbp-230h] BYREF
  _QWORD v68[2]; // [rsp+F8h] [rbp-228h] BYREF
  __int64 v69; // [rsp+108h] [rbp-218h]
  unsigned __int64 v70[6]; // [rsp+110h] [rbp-210h] BYREF
  const char *v71; // [rsp+140h] [rbp-1E0h] BYREF
  __int64 v72; // [rsp+148h] [rbp-1D8h] BYREF
  __int64 v73; // [rsp+150h] [rbp-1D0h]
  __int64 v74; // [rsp+158h] [rbp-1C8h]
  __int64 *v75; // [rsp+160h] [rbp-1C0h]
  unsigned __int64 *v76; // [rsp+190h] [rbp-190h]
  unsigned int v77; // [rsp+198h] [rbp-188h]
  char v78; // [rsp+1A0h] [rbp-180h] BYREF

  v60 = 0;
  v8 = sub_AF1560(0x56u);
  v63 = v8;
  if ( v8 )
  {
    v61 = (_QWORD *)sub_C7D670((unsigned __int64)v8 << 6, 8);
    sub_23FE7B0((__int64)&v60);
  }
  else
  {
    v61 = 0;
    v62 = 0;
  }
  v66 = 0;
  v49 = (unsigned __int8 *)sub_F4BFF0(*a2, (__int64)&v60, 0, v9);
  v10 = sub_BD5D20(*a2);
  LOWORD(v75) = 261;
  v72 = v11;
  v71 = v10;
  sub_2644DA0((__int64 *)v59, a5, v11, (__int64)v59, v12, v13, a6);
  LOWORD(v75) = 260;
  v71 = (const char *)v59;
  sub_BD6B50(v49, &v71);
  v50 = a4[1];
  if ( *a4 != v50 )
  {
    v56 = (__int64)(a3 + 1);
    v14 = a3;
    v15 = *a4;
    v16 = v14;
    while ( 1 )
    {
      v17 = *(_QWORD *)v15;
      v72 = 2;
      v73 = 0;
      v74 = v17;
      if ( v17 != -4096 && v17 != 0 && v17 != -8192 )
        sub_BD73F0((__int64)&v72);
      v75 = &v60;
      v71 = (const char *)&unk_49DD7B0;
      v18 = (unsigned __int8)sub_F9E960((__int64)&v60, (__int64)&v71, &v57) == 0;
      v19 = (unsigned __int64 *)v57;
      if ( v18 )
        break;
      v20 = v74;
      v21 = v57 + 40;
LABEL_10:
      v71 = (const char *)&unk_49DB368;
      if ( v20 != -4096 && v20 != 0 && v20 != -8192 )
      {
        v52 = v21;
        sub_BD60C0(&v72);
        v21 = v52;
      }
      v22 = v21[2];
      v23 = v16[2];
      if ( v23 )
      {
        v24 = *(_QWORD *)v15;
        v25 = v56;
        do
        {
          while ( *(_QWORD *)(v23 + 32) >= v24
               && (*(_QWORD *)(v23 + 32) != v24 || *(_DWORD *)(v23 + 40) >= *(_DWORD *)(v15 + 8)) )
          {
            v25 = v23;
            v23 = *(_QWORD *)(v23 + 16);
            if ( !v23 )
              goto LABEL_20;
          }
          v23 = *(_QWORD *)(v23 + 24);
        }
        while ( v23 );
LABEL_20:
        if ( v25 != v56
          && *(_QWORD *)(v25 + 32) <= v24
          && (*(_QWORD *)(v25 + 32) != v24 || *(_DWORD *)(v15 + 8) >= *(_DWORD *)(v25 + 40)) )
        {
          goto LABEL_25;
        }
      }
      else
      {
        v25 = v56;
      }
      v53 = v22;
      v71 = (const char *)v15;
      v26 = sub_263EBB0(v16, v25, (const __m128i **)&v71);
      v22 = v53;
      v25 = v26;
LABEL_25:
      *(_QWORD *)(v25 + 48) = v22;
      v15 += 16;
      *(_DWORD *)(v25 + 56) = a5;
      if ( v50 == v15 )
        goto LABEL_26;
    }
    v34 = v63;
    v67 = v57;
    ++v60;
    v35 = v62 + 1;
    if ( 4 * ((int)v62 + 1) >= 3 * v63 )
    {
      v34 = 2 * v63;
    }
    else if ( v63 - HIDWORD(v62) - v35 > v63 >> 3 )
    {
      goto LABEL_40;
    }
    sub_CF32C0((__int64)&v60, v34);
    sub_F9E960((__int64)&v60, (__int64)&v71, &v67);
    v35 = v62 + 1;
    v19 = (unsigned __int64 *)v67;
LABEL_40:
    LODWORD(v62) = v35;
    v20 = v19[3];
    if ( v20 == -4096 )
    {
      v36 = v74;
      v37 = v19 + 1;
      if ( v74 != -4096 )
      {
LABEL_45:
        v19[3] = v36;
        if ( v36 == 0 || v36 == -4096 || v36 == -8192 )
        {
          v20 = v74;
        }
        else
        {
          v55 = v19;
          sub_BD6050(v37, v72 & 0xFFFFFFFFFFFFFFF8LL);
          v20 = v74;
          v19 = v55;
        }
      }
    }
    else
    {
      v36 = v74;
      --HIDWORD(v62);
      if ( v74 != v20 )
      {
        v37 = v19 + 1;
        if ( v20 && v20 != -8192 )
        {
          v47 = v19;
          v54 = v19 + 1;
          sub_BD60C0(v19 + 1);
          v36 = v74;
          v19 = v47;
          v37 = v54;
        }
        goto LABEL_45;
      }
    }
    v38 = v75;
    v21 = v19 + 5;
    *v21 = 6;
    v21[1] = 0;
    *(v21 - 1) = v38;
    v21[2] = 0;
    goto LABEL_10;
  }
LABEL_26:
  v27 = (__int64 *)(*(__int64 (__fastcall **)(_QWORD, __int64))(a1 + 360))(*(_QWORD *)(a1 + 368), *a2);
  sub_B17560((__int64)&v71, (__int64)"memprof-context-disambiguation", (__int64)"MemprofClone", 12, *a2);
  sub_B18290((__int64)&v71, "created clone ", 0xEu);
  sub_B16080((__int64)&v67, "NewFunction", 11, v49);
  v28 = sub_23FD640((__int64)&v71, (__int64)&v67);
  sub_1049740(v27, v28);
  sub_2240A30(v70);
  sub_2240A30((unsigned __int64 *)&v67);
  v29 = v76;
  v71 = (const char *)&unk_49D9D40;
  v30 = &v76[10 * v77];
  if ( v76 != v30 )
  {
    do
    {
      v30 -= 10;
      v31 = v30[4];
      if ( (unsigned __int64 *)v31 != v30 + 6 )
        j_j___libc_free_0(v31);
      if ( (unsigned __int64 *)*v30 != v30 + 2 )
        j_j___libc_free_0(*v30);
    }
    while ( v29 != v30 );
    v30 = v76;
  }
  if ( v30 != (unsigned __int64 *)&v78 )
    _libc_free((unsigned __int64)v30);
  v57 = v49;
  LODWORD(v58) = a5;
  sub_2240A30(v59);
  if ( v66 )
  {
    v43 = v65;
    v66 = 0;
    if ( v65 )
    {
      v44 = v64;
      v45 = &v64[2 * v65];
      do
      {
        if ( *v44 != -4096 && *v44 != -8192 )
        {
          v46 = v44[1];
          if ( v46 )
            sub_B91220((__int64)(v44 + 1), v46);
        }
        v44 += 2;
      }
      while ( v45 != v44 );
      v43 = v65;
    }
    sub_C7D6A0((__int64)v64, 16LL * v43, 8);
  }
  v32 = v63;
  if ( v63 )
  {
    v68[0] = 2;
    v39 = v61;
    v68[1] = 0;
    v69 = -4096;
    v40 = &v61[8 * (unsigned __int64)v63];
    v67 = (unsigned __int8 *)&unk_49DD7B0;
    v41 = -4096;
    v70[0] = 0;
    v72 = 2;
    v73 = 0;
    v74 = -8192;
    v71 = (const char *)&unk_49DD7B0;
    v75 = 0;
    while ( 1 )
    {
      v42 = v39[3];
      if ( v42 != v41 )
      {
        v41 = v74;
        if ( v42 != v74 )
        {
          sub_D68D70(v39 + 5);
          v41 = v39[3];
        }
      }
      *v39 = &unk_49DB368;
      if ( v41 != -4096 && v41 != 0 && v41 != -8192 )
        sub_BD60C0(v39 + 1);
      v39 += 8;
      if ( v40 == v39 )
        break;
      v41 = v69;
    }
    v71 = (const char *)&unk_49DB368;
    sub_D68D70(&v72);
    v67 = (unsigned __int8 *)&unk_49DB368;
    sub_D68D70(v68);
    v32 = v63;
  }
  sub_C7D6A0((__int64)v61, (unsigned __int64)v32 << 6, 8);
  return v57;
}
