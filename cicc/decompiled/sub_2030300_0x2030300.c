// Function: sub_2030300
// Address: 0x2030300
//
__int64 *__fastcall sub_2030300(
        __int64 *a1,
        __int64 a2,
        unsigned __int64 a3,
        __int64 a4,
        const void **a5,
        char a6,
        __m128i a7,
        double a8,
        __m128i a9)
{
  int v9; // r13d
  __int64 v12; // rsi
  int v13; // ebx
  __int64 v14; // rax
  char v15; // cl
  const void **v16; // r8
  char v17; // dl
  __int64 *v18; // r12
  __int64 v20; // rcx
  unsigned int v21; // r14d
  _BYTE *v22; // rax
  __int64 v23; // rbx
  _BYTE *v24; // rdx
  unsigned int v25; // eax
  __int64 v26; // r9
  const void **v27; // rdx
  __int64 *v28; // r15
  __int64 v29; // rbx
  unsigned __int64 v30; // r13
  char v31; // al
  __int64 v32; // rcx
  __int128 v33; // rax
  __int64 *v34; // rax
  int v35; // edx
  int v36; // edi
  __int64 *v37; // rdx
  __int64 v38; // rax
  _BYTE *v39; // rax
  __int64 *v40; // r14
  __int64 v41; // rax
  unsigned int v42; // edx
  unsigned int v43; // eax
  _QWORD *v44; // rdi
  _QWORD *v45; // rax
  int v46; // edx
  __int64 v47; // rsi
  int v48; // r8d
  __int64 v49; // rdx
  unsigned __int64 v50; // rcx
  __int64 *v51; // rax
  int v52; // edx
  __int64 v53; // rcx
  __int64 v54; // r8
  __int64 v55; // r9
  __int64 v56; // rax
  int v57; // edx
  __int64 v58; // rsi
  int v59; // edi
  unsigned __int64 v60; // rax
  unsigned int v61; // edx
  __int64 v62; // rax
  _BYTE *v63; // rax
  __int64 *v64; // r13
  __int64 v65; // r12
  __int64 (__fastcall *v66)(__int64, __int64); // rbx
  __int64 v67; // rax
  unsigned int v68; // edx
  unsigned __int8 v69; // al
  __int128 v70; // rax
  __int128 v71; // [rsp-10h] [rbp-230h]
  __int128 v72; // [rsp-10h] [rbp-230h]
  unsigned int v73; // [rsp+14h] [rbp-20Ch]
  unsigned int v75; // [rsp+24h] [rbp-1FCh]
  unsigned int v76; // [rsp+38h] [rbp-1E8h]
  const void **v77; // [rsp+40h] [rbp-1E0h]
  const void **v78; // [rsp+40h] [rbp-1E0h]
  __int64 v79; // [rsp+48h] [rbp-1D8h]
  char v81; // [rsp+58h] [rbp-1C8h]
  __int64 v82; // [rsp+58h] [rbp-1C8h]
  __int64 (__fastcall *v84)(__int64, __int64); // [rsp+60h] [rbp-1C0h]
  _QWORD *v85; // [rsp+60h] [rbp-1C0h]
  int v86; // [rsp+60h] [rbp-1C0h]
  int v88; // [rsp+68h] [rbp-1B8h]
  __int64 v89; // [rsp+A0h] [rbp-180h] BYREF
  const void **v90; // [rsp+A8h] [rbp-178h]
  unsigned int v91; // [rsp+B0h] [rbp-170h] BYREF
  const void **v92; // [rsp+B8h] [rbp-168h]
  __int64 v93; // [rsp+C0h] [rbp-160h] BYREF
  int v94; // [rsp+C8h] [rbp-158h]
  __m128i v95; // [rsp+D0h] [rbp-150h] BYREF
  _BYTE *v96; // [rsp+E0h] [rbp-140h] BYREF
  __int64 v97; // [rsp+E8h] [rbp-138h]
  _BYTE v98[304]; // [rsp+F0h] [rbp-130h] BYREF

  v89 = a4;
  v12 = *(_QWORD *)(a2 + 72);
  v79 = (unsigned int)a3;
  v13 = a3;
  v14 = *(_QWORD *)(a2 + 40) + 16LL * (unsigned int)a3;
  v90 = a5;
  v15 = *(_BYTE *)v14;
  v16 = *(const void ***)(v14 + 8);
  LOBYTE(v91) = *(_BYTE *)v14;
  v92 = v16;
  v93 = v12;
  if ( v12 )
  {
    v77 = v16;
    v81 = v15;
    sub_1623A60((__int64)&v93, v12, 2);
    v16 = v77;
    v15 = v81;
  }
  v17 = v89;
  v94 = *(_DWORD *)(a2 + 64);
  if ( (_BYTE)v89 == v15 )
  {
    if ( v90 == v16 || v15 )
    {
      v18 = (__int64 *)a2;
      goto LABEL_7;
    }
LABEL_37:
    v43 = sub_1F58D30((__int64)&v91);
    v17 = v89;
    v21 = v43;
    if ( !(_BYTE)v89 )
      goto LABEL_38;
    goto LABEL_12;
  }
  if ( !v15 )
    goto LABEL_37;
  v20 = (unsigned __int8)(v15 - 14);
  v21 = word_4305480[v20];
  if ( !(_BYTE)v89 )
  {
LABEL_38:
    v75 = sub_1F58D30((__int64)&v89);
    goto LABEL_13;
  }
LABEL_12:
  v75 = word_4305480[(unsigned __int8)(v17 - 14)];
LABEL_13:
  if ( v75 > v21 && !(v75 % v21) )
  {
    v86 = v75 / v21;
    v96 = v98;
    v95.m128i_i64[0] = 0;
    v95.m128i_i32[2] = 0;
    v97 = 0x1000000000LL;
    sub_202F910((__int64)&v96, v75 / v21, &v95, v20, (int)v16, a6);
    if ( a6 )
      v56 = sub_1D38BB0(a1[1], 0, (__int64)&v93, v91, v92, 0, a7, a8, a9, 0);
    else
      v56 = (__int64)sub_1D2B530((_QWORD *)a1[1], v91, (__int64)v92, v53, v54, v55);
    v58 = v56;
    v59 = v57;
    v60 = (unsigned __int64)v96;
    v61 = 1;
    *(_QWORD *)v96 = a2;
    *(_DWORD *)(v60 + 8) = v13;
    if ( v86 != 1 )
    {
      do
      {
        v62 = v61++;
        v63 = &v96[16 * v62];
        *(_QWORD *)v63 = v58;
        *((_DWORD *)v63 + 2) = v59;
      }
      while ( v86 != v61 );
    }
    *((_QWORD *)&v72 + 1) = (unsigned int)v97;
    *(_QWORD *)&v72 = v96;
    v51 = sub_1D359D0(
            (__int64 *)a1[1],
            107,
            (__int64)&v93,
            (unsigned int)v89,
            v90,
            0,
            *(double *)a7.m128i_i64,
            a8,
            a9,
            v72);
    goto LABEL_49;
  }
  if ( v75 >= v21 || !(v21 % v75) )
  {
    v22 = v98;
    v96 = v98;
    v97 = 0x1000000000LL;
    if ( v75 > 0x10 )
    {
      sub_16CD150((__int64)&v96, v98, v75, 16, (int)v16, a6);
      v22 = v96;
    }
    v23 = 16LL * v75;
    v24 = &v22[v23];
    for ( LODWORD(v97) = v75; v24 != v22; v22 += 16 )
    {
      if ( v22 )
      {
        *(_QWORD *)v22 = 0;
        *((_DWORD *)v22 + 2) = 0;
      }
    }
    LOBYTE(v25) = sub_1F7E0F0((__int64)&v89);
    v76 = v25;
    v78 = v27;
    if ( v75 <= v21 )
      v21 = v75;
    v73 = v21;
    if ( v21 )
    {
      LODWORD(v28) = v9;
      v29 = 0;
      v30 = a3;
      do
      {
        v40 = (__int64 *)a1[1];
        v82 = *a1;
        v84 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)*a1 + 48LL);
        v41 = sub_1E0A0C0(v40[4]);
        if ( v84 == sub_1D13A20 )
        {
          v42 = 8 * sub_15A9520(v41, 0);
          if ( v42 == 32 )
          {
            v31 = 5;
          }
          else if ( v42 <= 0x20 )
          {
            v31 = 3;
            if ( v42 != 8 )
              v31 = 4 * (v42 == 16);
          }
          else
          {
            v31 = 6;
            if ( v42 != 64 )
            {
              v31 = 0;
              if ( v42 == 128 )
                v31 = 7;
            }
          }
        }
        else
        {
          v31 = v84(v82, v41);
        }
        LOBYTE(v28) = v31;
        v32 = (unsigned int)v28;
        v28 = &v93;
        *(_QWORD *)&v33 = sub_1D38BB0((__int64)v40, v29, (__int64)&v93, v32, 0, 0, a7, a8, a9, 0);
        v30 = v30 & 0xFFFFFFFF00000000LL | v79;
        v34 = sub_1D332F0(v40, 106, (__int64)&v93, v76, v78, 0, *(double *)a7.m128i_i64, a8, a9, a2, v30, v33);
        v36 = v35;
        v37 = v34;
        v38 = v29++;
        v39 = &v96[16 * v38];
        *(_QWORD *)v39 = v37;
        *((_DWORD *)v39 + 2) = v36;
      }
      while ( v73 != v29 );
    }
    v44 = (_QWORD *)a1[1];
    if ( a6 )
    {
      v47 = sub_1D38BB0((__int64)v44, 0, (__int64)&v93, v76, v78, 0, a7, a8, a9, 0);
      v48 = v52;
    }
    else
    {
      v95.m128i_i64[0] = 0;
      v95.m128i_i32[2] = 0;
      v45 = sub_1D2B300(v44, 0x30u, (__int64)&v95, v76, (__int64)v78, v26);
      if ( v95.m128i_i64[0] )
      {
        v85 = v45;
        v88 = v46;
        sub_161E7C0((__int64)&v95, v95.m128i_i64[0]);
        v45 = v85;
        v46 = v88;
      }
      v47 = (__int64)v45;
      v48 = v46;
    }
    if ( v75 > v73 )
    {
      v49 = 16LL * v73;
      do
      {
        v50 = (unsigned __int64)v96;
        *(_QWORD *)&v96[v49] = v47;
        *(_DWORD *)(v50 + v49 + 8) = v48;
        v49 += 16;
      }
      while ( 16 * (v73 + (unsigned __int64)(v75 - 1 - v73) + 1) != v49 );
    }
    *((_QWORD *)&v71 + 1) = (unsigned int)v97;
    *(_QWORD *)&v71 = v96;
    v51 = sub_1D359D0((__int64 *)a1[1], 104, (__int64)&v93, v89, v90, 0, *(double *)a7.m128i_i64, a8, a9, v71);
LABEL_49:
    v18 = v51;
    if ( v96 != v98 )
      _libc_free((unsigned __int64)v96);
    goto LABEL_7;
  }
  v64 = (__int64 *)a1[1];
  v65 = *a1;
  v66 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)*a1 + 48LL);
  v67 = sub_1E0A0C0(v64[4]);
  if ( v66 == sub_1D13A20 )
  {
    v68 = 8 * sub_15A9520(v67, 0);
    if ( v68 == 32 )
    {
      v69 = 5;
    }
    else if ( v68 > 0x20 )
    {
      v69 = 6;
      if ( v68 != 64 )
      {
        v69 = 0;
        if ( v68 == 128 )
          v69 = 7;
      }
    }
    else
    {
      v69 = 3;
      if ( v68 != 8 )
        v69 = 4 * (v68 == 16);
    }
  }
  else
  {
    v69 = v66(v65, v67);
  }
  *(_QWORD *)&v70 = sub_1D38BB0((__int64)v64, 0, (__int64)&v93, v69, 0, 0, a7, a8, a9, 0);
  v18 = sub_1D332F0(v64, 109, (__int64)&v93, (unsigned int)v89, v90, 0, *(double *)a7.m128i_i64, a8, a9, a2, a3, v70);
LABEL_7:
  if ( v93 )
    sub_161E7C0((__int64)&v93, v93);
  return v18;
}
