// Function: sub_205F810
// Address: 0x205f810
//
void __fastcall sub_205F810(__int64 *a1, __int64 a2, double a3, double a4, __m128i a5)
{
  __int64 *v7; // r13
  __int64 v8; // r12
  __int64 v9; // rax
  __int64 (*v10)(); // rdx
  __int64 (*v11)(); // rax
  __int64 v12; // r14
  int v13; // edx
  __int64 v14; // rax
  __int64 v15; // rsi
  __int64 v16; // rax
  __int64 v17; // rax
  int v18; // edi
  __int64 *v19; // r14
  unsigned __int8 *v20; // r13
  __int64 v21; // rax
  unsigned int v22; // edx
  unsigned __int8 v23; // al
  __int64 v24; // rax
  int v25; // edx
  __int64 v26; // r8
  __int64 v27; // r9
  __int64 v28; // rdx
  __int64 *v29; // rax
  __int64 v30; // rdx
  int v31; // edx
  __int64 *v32; // r14
  unsigned __int8 *v33; // r13
  __int64 v34; // rax
  unsigned int v35; // edx
  unsigned __int8 v36; // al
  __int64 v37; // rax
  int v38; // edx
  __int64 v39; // r8
  __int64 v40; // rdx
  __int64 v41; // r9
  __int64 *v42; // rax
  __int64 v43; // rdx
  __int64 v44; // rax
  __int64 *v45; // rdi
  int v46; // edx
  const void ***v47; // rax
  int v48; // edx
  __int64 v49; // r9
  __int64 *v50; // r13
  int v51; // edx
  int v52; // r14d
  __int64 *v53; // rax
  __int64 v54; // rsi
  __int64 v55; // r13
  __int64 v56; // rax
  unsigned int v57; // edx
  unsigned __int8 v58; // al
  int v59; // edx
  __int128 v60; // [rsp-20h] [rbp-150h]
  __int128 v61; // [rsp-10h] [rbp-140h]
  __int128 v62; // [rsp-10h] [rbp-140h]
  int v63; // [rsp+Ch] [rbp-124h]
  int v64; // [rsp+10h] [rbp-120h]
  __int64 v65; // [rsp+10h] [rbp-120h]
  unsigned int v66; // [rsp+18h] [rbp-118h]
  const void ***v67; // [rsp+18h] [rbp-118h]
  const void ***v68; // [rsp+20h] [rbp-110h]
  int v69; // [rsp+20h] [rbp-110h]
  int v70; // [rsp+30h] [rbp-100h]
  __int64 v71; // [rsp+30h] [rbp-100h]
  __int64 v72; // [rsp+38h] [rbp-F8h]
  __int64 v73; // [rsp+80h] [rbp-B0h] BYREF
  int v74; // [rsp+88h] [rbp-A8h]
  __int128 v75; // [rsp+90h] [rbp-A0h] BYREF
  __int128 v76; // [rsp+A0h] [rbp-90h]
  __int64 v77; // [rsp+B0h] [rbp-80h] BYREF
  int v78; // [rsp+B8h] [rbp-78h]
  _QWORD *v79; // [rsp+C0h] [rbp-70h]
  __int64 v80; // [rsp+C8h] [rbp-68h]
  unsigned __int8 *v81; // [rsp+D0h] [rbp-60h] BYREF
  __int64 v82; // [rsp+D8h] [rbp-58h]
  _BYTE v83[80]; // [rsp+E0h] [rbp-50h] BYREF

  sub_1E0D590(a2, *(_QWORD *)(a1[89] + 784));
  v7 = *(__int64 **)(a1[69] + 16);
  v8 = sub_15E38F0(*(_QWORD *)a1[89]);
  v9 = *v7;
  v10 = *(__int64 (**)())(*v7 + 488);
  if ( v10 == sub_1D45FC0 )
  {
LABEL_2:
    v11 = *(__int64 (**)())(v9 + 496);
    if ( v11 == sub_1D45FD0 || !((unsigned int (__fastcall *)(__int64 *, __int64))v11)(v7, v8) )
      return;
    goto LABEL_5;
  }
  if ( !((unsigned int (__fastcall *)(__int64 *, __int64))v10)(v7, v8) )
  {
    v9 = *v7;
    goto LABEL_2;
  }
LABEL_5:
  v12 = *(_QWORD *)a2;
  if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) != 10 )
  {
    v13 = *((_DWORD *)a1 + 134);
    v73 = 0;
    v81 = v83;
    v82 = 0x200000000LL;
    v14 = *a1;
    v74 = v13;
    if ( v14 )
    {
      if ( &v73 != (__int64 *)(v14 + 48) )
      {
        v15 = *(_QWORD *)(v14 + 48);
        v73 = v15;
        if ( v15 )
        {
          sub_1623A60((__int64)&v73, v15, 2);
          v12 = *(_QWORD *)a2;
        }
      }
    }
    v16 = sub_1E0A0C0(*(_QWORD *)(a1[69] + 32));
    sub_20C7CE0(v7, v16, v12, &v81, 0, 0);
    v17 = a1[89];
    v75 = 0;
    v18 = *(_DWORD *)(v17 + 932);
    v76 = 0;
    if ( v18 )
    {
      v19 = (__int64 *)a1[69];
      v20 = v81;
      v21 = sub_1E0A0C0(v19[4]);
      v22 = 8 * sub_15A9520(v21, 0);
      if ( v22 == 32 )
      {
        v23 = 5;
      }
      else if ( v22 > 0x20 )
      {
        v23 = 6;
        if ( v22 != 64 )
        {
          v23 = 0;
          if ( v22 == 128 )
            v23 = 7;
        }
      }
      else
      {
        v23 = 3;
        if ( v22 != 8 )
          v23 = 4 * (v22 == 16);
      }
      v66 = v23;
      v72 = a1[69] + 88;
      v64 = *(_DWORD *)(a1[89] + 932);
      v24 = sub_1D252B0((__int64)v19, v23, 0, 1, 0);
      v70 = v25;
      v68 = (const void ***)v24;
      v77 = v72;
      v78 = 0;
      v79 = sub_1D2A660(v19, v64, v66, 0, v26, v27);
      v80 = v28;
      *((_QWORD *)&v61 + 1) = 2;
      *(_QWORD *)&v61 = &v77;
      v29 = sub_1D36D80(v19, 47, (__int64)&v73, v68, v70, 0.0, a4, a5, (__int64)v68, v61);
      *(_QWORD *)&v75 = sub_1D323C0(
                          v19,
                          (__int64)v29,
                          v30,
                          (__int64)&v73,
                          *(unsigned int *)v20,
                          *((const void ***)v20 + 1),
                          0.0,
                          a4,
                          *(double *)a5.m128i_i64);
      DWORD2(v75) = v31;
    }
    else
    {
      v55 = a1[69];
      v56 = sub_1E0A0C0(*(_QWORD *)(v55 + 32));
      v57 = 8 * sub_15A9520(v56, 0);
      if ( v57 == 32 )
      {
        v58 = 5;
      }
      else if ( v57 > 0x20 )
      {
        v58 = 6;
        if ( v57 != 64 )
        {
          v58 = 0;
          if ( v57 == 128 )
            v58 = 7;
        }
      }
      else
      {
        v58 = 3;
        if ( v57 != 8 )
          v58 = 4 * (v57 == 16);
      }
      *(_QWORD *)&v75 = sub_1D38BB0(v55, 0, (__int64)&v73, v58, 0, 0, (__m128i)0LL, a4, a5, 0);
      DWORD2(v75) = v59;
    }
    v32 = (__int64 *)a1[69];
    v33 = v81 + 16;
    v34 = sub_1E0A0C0(v32[4]);
    v35 = 8 * sub_15A9520(v34, 0);
    if ( v35 == 32 )
    {
      v36 = 5;
    }
    else if ( v35 > 0x20 )
    {
      v36 = 6;
      if ( v35 != 64 )
      {
        v36 = 0;
        if ( v35 == 128 )
          v36 = 7;
      }
    }
    else
    {
      v36 = 3;
      if ( v35 != 8 )
        v36 = 4 * (v35 == 16);
    }
    v65 = v36;
    v71 = a1[69] + 88;
    v63 = *(_DWORD *)(a1[89] + 936);
    v37 = sub_1D252B0((__int64)v32, v36, 0, 1, 0);
    v69 = v38;
    v67 = (const void ***)v37;
    v77 = v71;
    v78 = 0;
    v79 = sub_1D2A660(v32, v63, v65, 0, v39, v65);
    v80 = v40;
    *((_QWORD *)&v62 + 1) = 2;
    *(_QWORD *)&v62 = &v77;
    v42 = sub_1D36D80(v32, 47, (__int64)&v73, v67, v69, 0.0, a4, a5, v41, v62);
    v44 = sub_1D323C0(
            v32,
            (__int64)v42,
            v43,
            (__int64)&v73,
            *(unsigned int *)v33,
            *((const void ***)v33 + 1),
            0.0,
            a4,
            *(double *)a5.m128i_i64);
    v45 = (__int64 *)a1[69];
    *(_QWORD *)&v76 = v44;
    DWORD2(v76) = v46;
    v47 = (const void ***)sub_1D25C30((__int64)v45, v81, (unsigned int)v82);
    *((_QWORD *)&v60 + 1) = 2;
    *(_QWORD *)&v60 = &v75;
    v50 = sub_1D36D80(v45, 51, (__int64)&v73, v47, v48, 0.0, a4, a5, v49, v60);
    v52 = v51;
    v77 = a2;
    v53 = sub_205F5C0((__int64)(a1 + 1), &v77);
    v54 = v73;
    v53[1] = (__int64)v50;
    *((_DWORD *)v53 + 4) = v52;
    if ( v54 )
      sub_161E7C0((__int64)&v73, v54);
    if ( v81 != v83 )
      _libc_free((unsigned __int64)v81);
  }
}
