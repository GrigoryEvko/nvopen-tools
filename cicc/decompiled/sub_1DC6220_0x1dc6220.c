// Function: sub_1DC6220
// Address: 0x1dc6220
//
void __fastcall sub_1DC6220(const __m128i *a1, __int64 a2, char a3)
{
  const __m128i *v3; // r15
  __int64 v4; // r13
  _QWORD *v5; // rax
  __int64 (*v6)(void); // rdx
  int v7; // ebx
  __int64 v8; // r12
  __m128i *v9; // r14
  char v10; // al
  __int64 v11; // rcx
  int v12; // eax
  unsigned int v13; // r11d
  __int64 v14; // rdx
  int v15; // ecx
  unsigned int v16; // r8d
  int v17; // r9d
  __int64 v18; // rbx
  __int64 v19; // rcx
  __int64 v20; // rdx
  __int64 v21; // r8
  __int64 v22; // rsi
  _QWORD *v23; // r13
  _QWORD *v24; // r14
  __int64 v25; // rdx
  int v26; // ecx
  unsigned int v27; // r8d
  _QWORD *v28; // rax
  _QWORD *v29; // r8
  __int64 *v30; // r10
  __int64 *v31; // r12
  __int64 *v32; // r15
  _QWORD *v33; // r13
  int v34; // r14d
  __int64 v35; // rbx
  __int64 v36; // rax
  int v37; // r8d
  int v38; // r9d
  __int64 v39; // rdx
  __int64 v40; // rdx
  __int64 v41; // rbx
  __int64 v42; // r9
  __int64 v43; // rax
  __int64 v44; // r15
  __int64 v45; // r12
  __int64 v46; // r13
  __int64 (__fastcall *v47)(const __m128i **, const __m128i *, int); // rdx
  __int64 v48; // rcx
  __m128i *v49; // rax
  __int64 v50; // rax
  __int64 v51; // [rsp+8h] [rbp-338h]
  __m128i *v52; // [rsp+10h] [rbp-330h]
  __int64 v53; // [rsp+18h] [rbp-328h]
  const __m128i *v54; // [rsp+20h] [rbp-320h]
  __int64 v55; // [rsp+28h] [rbp-318h]
  __int64 v56; // [rsp+30h] [rbp-310h]
  const void *v57; // [rsp+38h] [rbp-308h]
  unsigned int v58; // [rsp+38h] [rbp-308h]
  const void *v59; // [rsp+40h] [rbp-300h]
  __int64 *v60; // [rsp+48h] [rbp-2F8h]
  unsigned int v61; // [rsp+48h] [rbp-2F8h]
  __int64 v62; // [rsp+48h] [rbp-2F8h]
  int v64; // [rsp+58h] [rbp-2E8h]
  unsigned int v65; // [rsp+5Ch] [rbp-2E4h]
  __int64 v66; // [rsp+60h] [rbp-2E0h]
  unsigned int v67; // [rsp+60h] [rbp-2E0h]
  __int64 *v68; // [rsp+60h] [rbp-2E0h]
  const __m128i *v69; // [rsp+60h] [rbp-2E0h]
  __int64 v70; // [rsp+68h] [rbp-2D8h]
  __m128i v71; // [rsp+70h] [rbp-2D0h] BYREF
  __int64 (__fastcall *v72)(const __m128i **, const __m128i *, int); // [rsp+80h] [rbp-2C0h]
  __int64 (__fastcall *v73)(__int64 *, __int64 *); // [rsp+88h] [rbp-2B8h]
  __int64 v74; // [rsp+90h] [rbp-2B0h]
  unsigned __int64 v75; // [rsp+98h] [rbp-2A8h]
  __int64 v76; // [rsp+A0h] [rbp-2A0h]
  int v77; // [rsp+A8h] [rbp-298h]
  __int64 v78; // [rsp+B0h] [rbp-290h]
  _QWORD *v79; // [rsp+B8h] [rbp-288h]
  __int64 v80; // [rsp+C0h] [rbp-280h]
  unsigned int v81; // [rsp+C8h] [rbp-278h]
  _QWORD *v82; // [rsp+D0h] [rbp-270h]
  __int64 v83; // [rsp+D8h] [rbp-268h]
  _QWORD v84[3]; // [rsp+E0h] [rbp-260h] BYREF
  _BYTE *v85; // [rsp+F8h] [rbp-248h]
  __int64 v86; // [rsp+100h] [rbp-240h]
  _BYTE v87[568]; // [rsp+108h] [rbp-238h] BYREF

  v3 = a1;
  v4 = a2;
  v5 = (_QWORD *)a1->m128i_i64[1];
  v70 = 0;
  v6 = *(__int64 (**)(void))(**(_QWORD **)(*v5 + 16LL) + 112LL);
  if ( v6 != sub_1D00B10 )
  {
    v70 = v6();
    v5 = (_QWORD *)a1->m128i_i64[1];
  }
  v7 = *(_DWORD *)(a2 + 112);
  v65 = v7;
  if ( v7 < 0 )
    v8 = *(_QWORD *)(v5[3] + 16LL * (v7 & 0x7FFFFFFF) + 8);
  else
    v8 = *(_QWORD *)(v5[34] + 8LL * (unsigned int)v7);
  if ( !v8 )
    goto LABEL_21;
  while ( (*(_BYTE *)(v8 + 4) & 8) != 0 )
  {
    v8 = *(_QWORD *)(v8 + 32);
    if ( !v8 )
      goto LABEL_21;
  }
  v9 = &v71;
LABEL_8:
  if ( (*(_BYTE *)(v8 + 3) & 0x10) == 0 )
  {
    v10 = *(_BYTE *)(v8 + 4);
    if ( (v10 & 1) != 0 || (v10 & 2) != 0 )
      goto LABEL_20;
  }
  v11 = *(_QWORD *)(v4 + 104);
  a2 = *(_DWORD *)v8 >> 8;
  v12 = (*(_DWORD *)v8 >> 8) & 0xFFF;
  LOWORD(a2) = a2 & 0xFFF;
  if ( v11 )
  {
    if ( !v12 )
    {
      v13 = sub_1E69F40(v3->m128i_i64[1], v65);
      if ( *(_QWORD *)(v4 + 104) )
        goto LABEL_14;
LABEL_43:
      if ( *(_DWORD *)(v4 + 8) )
      {
        v67 = v13;
        v64 = sub_1E69F40(v3->m128i_i64[1], v65);
        v60 = (__int64 *)v3[2].m128i_i64[0];
        v28 = (_QWORD *)sub_145CDC0(0x78u, v60);
        v13 = v67;
        v29 = v28;
        if ( v28 )
        {
          v28[12] = 0;
          v59 = v28 + 2;
          *v28 = v28 + 2;
          v28[1] = 0x200000000LL;
          v57 = v28 + 10;
          v28[8] = v28 + 10;
          v28[9] = 0x200000000LL;
          if ( (_QWORD *)v4 != v28 )
          {
            v30 = *(__int64 **)(v4 + 64);
            v56 = (__int64)(v28 + 8);
            v68 = &v30[*(unsigned int *)(v4 + 72)];
            if ( v30 != v68 )
            {
              v55 = v8;
              v31 = v60;
              v61 = v13;
              v54 = v3;
              v32 = *(__int64 **)(v4 + 64);
              v53 = v4;
              v33 = v28;
              v52 = v9;
              v34 = 0;
              do
              {
                v35 = *v32;
                v36 = sub_145CDC0(0x10u, v31);
                if ( v36 )
                {
                  v39 = *(_QWORD *)(v35 + 8);
                  *(_DWORD *)v36 = v34;
                  *(_QWORD *)(v36 + 8) = v39;
                }
                v40 = *((unsigned int *)v33 + 18);
                if ( (unsigned int)v40 >= *((_DWORD *)v33 + 19) )
                {
                  v51 = v36;
                  sub_16CD150(v56, v57, 0, 8, v37, v38);
                  v40 = *((unsigned int *)v33 + 18);
                  v36 = v51;
                }
                ++v32;
                *(_QWORD *)(v33[8] + 8 * v40) = v36;
                v34 = *((_DWORD *)v33 + 18) + 1;
                *((_DWORD *)v33 + 18) = v34;
              }
              while ( v68 != v32 );
              v29 = v33;
              v8 = v55;
              v13 = v61;
              v3 = v54;
              v4 = v53;
              v9 = v52;
            }
            v41 = *(_QWORD *)v4;
            v42 = *(_QWORD *)v4 + 24LL * *(unsigned int *)(v4 + 8);
            if ( *(_QWORD *)v4 != v42 )
            {
              v43 = *((unsigned int *)v29 + 2);
              v69 = v3;
              v44 = v8;
              v45 = (__int64)v29;
              v62 = v4;
              v46 = *(_QWORD *)v4 + 24LL * *(unsigned int *)(v4 + 8);
              do
              {
                v47 = *(__int64 (__fastcall **)(const __m128i **, const __m128i *, int))(*(_QWORD *)(v45 + 64)
                                                                                       + 8LL
                                                                                       * **(unsigned int **)(v41 + 16));
                v48 = *(_QWORD *)(v41 + 8);
                v71.m128i_i64[0] = *(_QWORD *)v41;
                v71.m128i_i64[1] = v48;
                v72 = v47;
                if ( (unsigned int)v43 >= *(_DWORD *)(v45 + 12) )
                {
                  v58 = v13;
                  sub_16CD150(v45, v59, 0, 24, (int)v29, v42);
                  v43 = *(unsigned int *)(v45 + 8);
                  v13 = v58;
                }
                v41 += 24;
                v49 = (__m128i *)(*(_QWORD *)v45 + 24 * v43);
                *v49 = _mm_loadu_si128(v9);
                v49[1].m128i_i64[0] = v9[1].m128i_i64[0];
                v43 = (unsigned int)(*(_DWORD *)(v45 + 8) + 1);
                *(_DWORD *)(v45 + 8) = v43;
              }
              while ( v46 != v41 );
              v29 = (_QWORD *)v45;
              v4 = v62;
              v8 = v44;
              v3 = v69;
            }
          }
          *((_DWORD *)v29 + 28) = v64;
        }
        v50 = *(_QWORD *)(v4 + 104);
        *(_QWORD *)(v4 + 104) = v29;
        v29[13] = v50;
      }
      goto LABEL_14;
    }
  }
  else if ( !v12 || !a3 )
  {
    if ( (*(_BYTE *)(v8 + 3) & 0x10) == 0 )
      goto LABEL_20;
    goto LABEL_41;
  }
  v13 = *(_DWORD *)(*(_QWORD *)(v70 + 248) + 4LL * (unsigned __int16)a2);
  if ( !v11 )
    goto LABEL_43;
LABEL_14:
  a2 = v3[2].m128i_i64[0];
  v73 = sub_1DC3420;
  v72 = sub_1DC32A0;
  v71.m128i_i64[0] = v8;
  v71.m128i_i64[1] = (__int64)v3;
  sub_1DB5D80(v4, a2, v13, (__int64)v9);
  if ( v72 )
  {
    a2 = (__int64)v9;
    v72((const __m128i **)v9, v9, 3);
  }
  if ( (*(_BYTE *)(v8 + 3) & 0x10) != 0 && !*(_QWORD *)(v4 + 104) )
  {
LABEL_41:
    a2 = v3[2].m128i_i64[0];
    sub_1DC3350(v3[1].m128i_i64[0], (__int64 *)a2, (__int64 *)v4, v8);
  }
LABEL_20:
  while ( 1 )
  {
    v8 = *(_QWORD *)(v8 + 32);
    if ( !v8 )
      break;
    if ( (*(_BYTE *)(v8 + 4) & 8) == 0 )
      goto LABEL_8;
  }
LABEL_21:
  sub_1DB4C70(v4);
  v18 = *(_QWORD *)(v4 + 104);
  if ( v18 )
  {
    v66 = v4;
    do
    {
      v71 = 0u;
      v19 = v3[1].m128i_i64[1];
      v20 = v3[1].m128i_i64[0];
      v21 = v3[2].m128i_i64[0];
      v22 = v3->m128i_i64[0];
      v82 = v84;
      v86 = 0x1000000000LL;
      v72 = 0;
      v73 = 0;
      v74 = 0;
      v75 = 0;
      v76 = 0;
      v77 = 0;
      v78 = 0;
      v79 = 0;
      v80 = 0;
      v81 = 0;
      v83 = 0;
      v84[0] = 0;
      v84[1] = 0;
      v85 = v87;
      sub_1DC3BD0(&v71, v22, v20, v19, v21, v17);
      sub_1DC5DD0((__int64)&v71, v18, v65, *(_DWORD *)(v18 + 112), v66);
      if ( v85 != v87 )
        _libc_free((unsigned __int64)v85);
      if ( v82 != v84 )
        _libc_free((unsigned __int64)v82);
      if ( v81 )
      {
        v23 = v79;
        v24 = &v79[7 * v81];
        do
        {
          if ( *v23 != -16 && *v23 != -8 )
          {
            _libc_free(v23[4]);
            _libc_free(v23[1]);
          }
          v23 += 7;
        }
        while ( v24 != v23 );
      }
      j___libc_free_0(v79);
      _libc_free(v75);
      v18 = *(_QWORD *)(v18 + 104);
    }
    while ( v18 );
    *(_DWORD *)(v66 + 72) = 0;
    *(_DWORD *)(v66 + 8) = 0;
    sub_1DC6170((__int64)v3, v66, v25, v26, v27, v17);
  }
  else
  {
    sub_1DC3680(v3, a2, v14, v15, v16, v17);
    sub_1DC5DD0((__int64)v3, v4, v65, -1, 0);
  }
}
