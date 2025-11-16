// Function: sub_35A13F0
// Address: 0x35a13f0
//
__int64 __fastcall sub_35A13F0(__int64 a1)
{
  __int64 v1; // r15
  __int64 *v2; // rax
  __int64 **v3; // rdx
  __int64 v4; // r13
  __int64 v5; // r12
  __int64 *v6; // rbx
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rbx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rdx
  int v13; // ebx
  unsigned __int32 v14; // eax
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // rdx
  unsigned __int32 v18; // r13d
  __int64 v19; // r14
  __int64 v20; // r12
  __int64 v21; // rcx
  __int64 v22; // rax
  __int64 *v23; // r12
  __int64 v24; // r14
  __int64 v25; // r15
  __int64 v26; // rdi
  _QWORD *v27; // rax
  __int64 v28; // rcx
  _QWORD *v29; // rax
  __int64 v30; // r14
  __int64 v31; // rdx
  __int64 v32; // r13
  __int64 v33; // rax
  unsigned int v34; // esi
  __int64 v35; // rbx
  __int64 v36; // rcx
  int v37; // r10d
  __int64 *v38; // rdi
  unsigned int i; // edx
  __int64 *v40; // rax
  __int64 v41; // r9
  __int64 v42; // rcx
  __int64 v43; // r8
  __int64 v44; // r9
  __int64 *v45; // rdi
  __int64 v46; // rsi
  __int64 v47; // rcx
  __int64 (*v48)(); // rax
  __int64 v49; // rdi
  __int64 *v50; // rcx
  __int64 v51; // rdx
  __int64 v52; // rsi
  void (__fastcall *v53)(__int64, __int64, __int64, __int64 *, __int64 *, _QWORD, __m128i *, _QWORD); // rax
  __int64 v54; // rdi
  __int64 v56; // rcx
  int v57; // edx
  __int64 v58; // rdx
  unsigned int v59; // edx
  int v60; // edx
  __int64 v61; // [rsp+18h] [rbp-1B8h]
  __int64 v62; // [rsp+20h] [rbp-1B0h]
  unsigned __int64 v63; // [rsp+28h] [rbp-1A8h]
  __int64 *v64; // [rsp+30h] [rbp-1A0h]
  __int64 v65; // [rsp+38h] [rbp-198h]
  __int64 *v66; // [rsp+48h] [rbp-188h]
  __int64 v67; // [rsp+58h] [rbp-178h]
  __int64 v68; // [rsp+68h] [rbp-168h]
  __int64 *v69; // [rsp+78h] [rbp-158h]
  __int64 v70; // [rsp+88h] [rbp-148h] BYREF
  __int64 v71; // [rsp+90h] [rbp-140h] BYREF
  __int64 *v72; // [rsp+98h] [rbp-138h] BYREF
  __int64 *v73[4]; // [rsp+A0h] [rbp-130h] BYREF
  __m128i v74; // [rsp+C0h] [rbp-110h] BYREF
  __int64 v75; // [rsp+D0h] [rbp-100h]
  __int64 v76; // [rsp+D8h] [rbp-F8h]
  __int64 v77; // [rsp+E0h] [rbp-F0h]
  __int64 *v78; // [rsp+F0h] [rbp-E0h] BYREF
  __int64 v79; // [rsp+F8h] [rbp-D8h]
  _BYTE v80[208]; // [rsp+100h] [rbp-D0h] BYREF

  v1 = a1;
  v2 = *(__int64 **)(a1 + 48);
  v3 = (__int64 **)v2[14];
  v4 = v2[4];
  v66 = *v3;
  if ( v2 == *v3 )
    v66 = v3[1];
  LOBYTE(v79) = 0;
  v5 = sub_2E7AAE0(v4, v2[2], (__int64)v78, 0);
  v6 = *(__int64 **)(*(_QWORD *)(a1 + 48) + 8LL);
  sub_2E33BD0(v4 + 320, v5);
  v7 = *(_QWORD *)v5;
  v8 = *v6;
  *(_QWORD *)(v5 + 8) = v6;
  v8 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v5 = v8 | v7 & 7;
  *(_QWORD *)(v8 + 8) = v5;
  *v6 = v5 | *v6 & 7;
  v9 = *(_QWORD *)(a1 + 48);
  v61 = sub_2E311E0(v9);
  v68 = *(_QWORD *)(v9 + 56);
  v70 = v68;
  if ( v68 != v61 )
  {
    v67 = v5;
    v63 = (unsigned __int64)(((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4)) << 32;
    v65 = a1 + 288;
    v62 = a1 + 256;
    v64 = (__int64 *)(v5 + 48);
    do
    {
      v12 = *(_QWORD *)(v68 + 32);
      v13 = *(_DWORD *)(v12 + 128);
      v14 = sub_2EC06C0(
              *(_QWORD *)(v1 + 24),
              *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v1 + 24) + 56LL) + 16LL * (*(_DWORD *)(v12 + 8) & 0x7FFFFFFF))
            & 0xFFFFFFFFFFFFFFF8LL,
              byte_3F871B3,
              0,
              v10,
              v11);
      v17 = *(_QWORD *)(v1 + 24);
      v18 = v14;
      v78 = (__int64 *)v80;
      v79 = 0x400000000LL;
      if ( v13 < 0 )
        v19 = *(_QWORD *)(*(_QWORD *)(v17 + 56) + 16LL * (v13 & 0x7FFFFFFF) + 8);
      else
        v19 = *(_QWORD *)(*(_QWORD *)(v17 + 304) + 8LL * (unsigned int)v13);
      if ( v19 )
      {
        if ( (*(_BYTE *)(v19 + 3) & 0x10) != 0 )
        {
          while ( 1 )
          {
            v19 = *(_QWORD *)(v19 + 32);
            if ( !v19 )
              break;
            if ( (*(_BYTE *)(v19 + 3) & 0x10) == 0 )
              goto LABEL_9;
          }
        }
        else
        {
LABEL_9:
          v20 = *(_QWORD *)(v19 + 16);
          v21 = 0;
LABEL_10:
          v22 = v20;
          if ( *(_QWORD *)(v1 + 48) != *(_QWORD *)(v20 + 24) )
          {
            if ( v21 + 1 > (unsigned __int64)HIDWORD(v79) )
            {
              sub_C8D5F0((__int64)&v78, v80, v21 + 1, 8u, v15, v16);
              v21 = (unsigned int)v79;
            }
            v78[v21] = v20;
            v21 = (unsigned int)(v79 + 1);
            LODWORD(v79) = v79 + 1;
            v22 = *(_QWORD *)(v19 + 16);
          }
          while ( 1 )
          {
            v19 = *(_QWORD *)(v19 + 32);
            if ( !v19 )
              break;
            if ( (*(_BYTE *)(v19 + 3) & 0x10) == 0 )
            {
              v20 = *(_QWORD *)(v19 + 16);
              if ( v22 != v20 )
                goto LABEL_10;
            }
          }
          v69 = &v78[v21];
          if ( v69 != v78 )
          {
            v23 = v78;
            v24 = v1;
            do
            {
              v25 = *v23++;
              v26 = *(_QWORD *)(**(_QWORD **)(v24 + 24) + 16LL);
              v27 = (_QWORD *)(*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v26 + 200LL))(v26);
              sub_2E8A790(v25, v13, v18, 0, v27);
            }
            while ( v69 != v23 );
            v1 = v24;
          }
        }
      }
      v28 = *(_QWORD *)(*(_QWORD *)(v1 + 32) + 8LL);
      v72 = 0;
      memset(v73, 0, 24);
      v29 = sub_2F26260(v67, v64, (__int64 *)v73, v28, v18);
      v74.m128i_i32[2] = v13;
      v30 = (__int64)v29;
      v32 = v31;
      v74.m128i_i64[0] = 0;
      v75 = 0;
      v76 = 0;
      v77 = 0;
      sub_2E8EAD0(v31, (__int64)v29, &v74);
      v33 = *(_QWORD *)(v1 + 48);
      v74.m128i_i8[0] = 4;
      v75 = 0;
      v74.m128i_i32[0] &= 0xFFF000FF;
      v76 = v33;
      sub_2E8EAD0(v32, v30, &v74);
      v71 = v32;
      if ( v73[0] )
        sub_B91220((__int64)v73, (__int64)v73[0]);
      if ( v72 )
        sub_B91220((__int64)&v72, (__int64)v72);
      v34 = *(_DWORD *)(v1 + 312);
      v35 = v71;
      v74.m128i_i64[0] = v67;
      v74.m128i_i64[1] = v68;
      if ( v34 )
      {
        v36 = *(_QWORD *)(v1 + 296);
        v37 = 1;
        v38 = 0;
        for ( i = (v34 - 1)
                & (((0xBF58476D1CE4E5B9LL * (v63 | ((unsigned int)v68 >> 9) ^ ((unsigned int)v68 >> 4))) >> 31)
                 ^ (484763065 * (v63 | ((unsigned int)v68 >> 9) ^ ((unsigned int)v68 >> 4)))); ; i = (v34 - 1) & v59 )
        {
          v40 = (__int64 *)(v36 + 24LL * i);
          v41 = *v40;
          if ( v67 == *v40 && v68 == v40[1] )
            break;
          if ( v41 == -4096 )
          {
            if ( v40[1] == -4096 )
            {
              v60 = *(_DWORD *)(v1 + 304);
              if ( v38 )
                v40 = v38;
              ++*(_QWORD *)(v1 + 288);
              v57 = v60 + 1;
              v73[0] = v40;
              if ( 4 * v57 >= 3 * v34 )
                goto LABEL_54;
              v56 = v67;
              if ( v34 - *(_DWORD *)(v1 + 308) - v57 > v34 >> 3 )
                goto LABEL_56;
              goto LABEL_55;
            }
          }
          else if ( v41 == -8192 && v40[1] == -8192 && !v38 )
          {
            v38 = (__int64 *)(v36 + 24LL * i);
          }
          v59 = v37 + i;
          ++v37;
        }
      }
      else
      {
        ++*(_QWORD *)(v1 + 288);
        v73[0] = 0;
LABEL_54:
        v34 *= 2;
LABEL_55:
        sub_35A1120(v65, v34);
        sub_359BDE0(v65, v74.m128i_i64, v73);
        v56 = v74.m128i_i64[0];
        v57 = *(_DWORD *)(v1 + 304) + 1;
        v40 = v73[0];
LABEL_56:
        *(_DWORD *)(v1 + 304) = v57;
        if ( *v40 != -4096 || v40[1] != -4096 )
          --*(_DWORD *)(v1 + 308);
        *v40 = v56;
        v58 = v74.m128i_i64[1];
        v40[2] = 0;
        v40[1] = v58;
      }
      v40[2] = v35;
      *sub_359C4A0(v62, &v71) = v68;
      if ( v78 != (__int64 *)v80 )
        _libc_free((unsigned __int64)v78);
      sub_2FD79B0(&v70);
      v68 = v70;
    }
    while ( v70 != v61 );
    v5 = v67;
  }
  sub_2E33690(*(_QWORD *)(v1 + 48), (__int64)v66, v5);
  sub_2E32770((__int64)v66, *(_QWORD *)(v1 + 48), v5);
  sub_2E33F80(v5, (__int64)v66, -1, v42, v43, v44);
  v45 = *(__int64 **)(v1 + 32);
  v72 = 0;
  v73[0] = 0;
  v46 = *(_QWORD *)(v1 + 48);
  v78 = (__int64 *)v80;
  v79 = 0x400000000LL;
  v47 = *v45;
  v48 = *(__int64 (**)())(*v45 + 344);
  if ( v48 != sub_2DB1AE0 )
  {
    ((void (__fastcall *)(__int64 *, __int64, __int64 **, __int64 **, __int64 **, _QWORD))v48)(
      v45,
      v46,
      &v72,
      v73,
      &v78,
      0);
    v45 = *(__int64 **)(v1 + 32);
    v46 = *(_QWORD *)(v1 + 48);
    v47 = *v45;
  }
  (*(void (__fastcall **)(__int64 *, __int64, _QWORD))(v47 + 360))(v45, v46, 0);
  v49 = *(_QWORD *)(v1 + 32);
  v50 = v73[0];
  v51 = (__int64)v72;
  v52 = *(_QWORD *)(v1 + 48);
  v53 = *(void (__fastcall **)(__int64, __int64, __int64, __int64 *, __int64 *, _QWORD, __m128i *, _QWORD))(*(_QWORD *)v49 + 368LL);
  if ( v73[0] == v66 )
    v50 = (__int64 *)v5;
  if ( v72 == v66 )
    v51 = v5;
  v74.m128i_i64[0] = 0;
  v53(v49, v52, v51, v50, v78, (unsigned int)v79, &v74, 0);
  sub_9C6650(&v74);
  v54 = *(_QWORD *)(v1 + 32);
  v74.m128i_i64[0] = 0;
  (*(void (__fastcall **)(__int64, __int64, __int64 *, _QWORD, _QWORD, _QWORD, __m128i *, _QWORD))(*(_QWORD *)v54 + 368LL))(
    v54,
    v5,
    v66,
    0,
    0,
    0,
    &v74,
    0);
  sub_9C6650(&v74);
  if ( v78 != (__int64 *)v80 )
    _libc_free((unsigned __int64)v78);
  return v5;
}
