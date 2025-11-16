// Function: sub_2BE6C70
// Address: 0x2be6c70
//
__int64 __fastcall sub_2BE6C70(__int64 a1)
{
  int v2; // eax
  int v3; // r12d
  unsigned int v4; // r14d
  unsigned __int64 v6; // rdi
  __m128i v7; // xmm5
  unsigned __int64 v8; // rdi
  __int64 v9; // rax
  unsigned __int64 *v10; // rbx
  __m128i *v11; // rsi
  __m128i v12; // xmm0
  bool v13; // zf
  __m128i v14; // xmm1
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rax
  unsigned __int64 v18; // rsi
  __int64 v19; // rsi
  __int64 v20; // rax
  __int64 v21; // rbx
  unsigned __int64 v22; // r14
  char v23; // di
  int v24; // eax
  __int64 v25; // r14
  unsigned __int64 v26; // rax
  __int64 v27; // rbx
  __int8 v28; // r13
  unsigned __int64 *v29; // rax
  __int64 v30; // rsi
  __int64 v31; // rdx
  unsigned __int64 *v32; // rax
  __m128i v33; // xmm0
  __m128i v34; // xmm3
  __m128i v35; // xmm2
  __m128i *v36; // rsi
  __m128i v37; // xmm0
  __m128i v38; // xmm1
  __int64 v39; // rdx
  __int64 v40; // rcx
  __int64 v41; // rdx
  unsigned __int64 v42; // rsi
  __int64 v43; // rsi
  unsigned __int64 v44; // rcx
  unsigned __int64 v45; // rbx
  unsigned __int64 v46; // rax
  __int64 v47; // rcx
  __m128i *v48; // rax
  const __m128i *v49; // rax
  __m128i v50; // xmm7
  __int64 v51; // rax
  __int64 *v52; // rdx
  __int64 v53; // rax
  __int64 v54; // rdx
  unsigned __int64 v55; // rdi
  __int64 v56; // rax
  __int64 v57; // rbx
  __int64 v58; // rdx
  __int64 v59; // rdx
  __int64 v60; // rdx
  __int8 v61; // r13
  __int64 v62; // rbx
  __m128i v63; // rax
  __int8 v64; // r13
  unsigned __int64 v65; // rdx
  __int64 v66; // rcx
  __int8 v67; // r13
  __int64 v68; // r12
  unsigned __int64 v69; // rbx
  unsigned __int64 v70; // rax
  __int64 v71; // rdx
  char v72; // al
  unsigned __int8 v73; // [rsp+Fh] [rbp-191h]
  __int8 v74; // [rsp+10h] [rbp-190h]
  unsigned __int64 v75; // [rsp+10h] [rbp-190h]
  unsigned __int64 *v76; // [rsp+10h] [rbp-190h]
  char v77; // [rsp+18h] [rbp-188h]
  __int64 v78; // [rsp+20h] [rbp-180h]
  __int64 v79; // [rsp+20h] [rbp-180h]
  __int64 v80; // [rsp+28h] [rbp-178h]
  __m128i v81; // [rsp+30h] [rbp-170h] BYREF
  __int64 v82; // [rsp+40h] [rbp-160h]
  __m128i v83; // [rsp+50h] [rbp-150h] BYREF
  __int64 i; // [rsp+60h] [rbp-140h]
  unsigned __int64 *v85[4]; // [rsp+70h] [rbp-130h] BYREF
  __m128i v86; // [rsp+90h] [rbp-110h] BYREF
  __m128i v87; // [rsp+A0h] [rbp-100h] BYREF
  __m128i v88; // [rsp+B0h] [rbp-F0h] BYREF
  __m128i v89; // [rsp+C0h] [rbp-E0h] BYREF
  __m128i v90; // [rsp+D0h] [rbp-D0h] BYREF
  __m128i v91; // [rsp+E0h] [rbp-C0h] BYREF
  __m128i v92; // [rsp+F0h] [rbp-B0h] BYREF
  __m128i v93; // [rsp+100h] [rbp-A0h] BYREF
  __m128i v94; // [rsp+110h] [rbp-90h] BYREF
  __m128i v95; // [rsp+120h] [rbp-80h] BYREF
  unsigned __int64 v96; // [rsp+130h] [rbp-70h]
  __int64 v97; // [rsp+138h] [rbp-68h]
  __int64 v98; // [rsp+140h] [rbp-60h]
  __int64 v99; // [rsp+148h] [rbp-58h]
  unsigned __int64 *v100; // [rsp+150h] [rbp-50h]
  __int64 v101; // [rsp+158h] [rbp-48h]
  __int64 v102; // [rsp+160h] [rbp-40h]
  __int64 *v103; // [rsp+168h] [rbp-38h]

  v2 = *(_DWORD *)(a1 + 152);
  v3 = *(_DWORD *)a1 & 0x10;
  if ( v2 == 20 )
  {
    v4 = sub_2BE0030(a1);
    if ( (_BYTE)v4 )
    {
      if ( *(_QWORD *)(a1 + 352) == *(_QWORD *)(a1 + 320) )
        goto LABEL_111;
      v61 = 0;
      if ( v3 && *(_DWORD *)(a1 + 152) == 18 )
        v61 = sub_2BE0030(a1);
      sub_2BDEE20(&v92, (_QWORD *)a1);
      v62 = v93.m128i_i64[0];
      v63.m128i_i64[1] = sub_2BE0960(*(unsigned __int64 **)(a1 + 256), -1, v92.m128i_i64[1], v61);
      v63.m128i_i64[0] = *(_QWORD *)(a1 + 256);
      v95 = v63;
      v96 = v63.m128i_u64[1];
      *(_QWORD *)(*(_QWORD *)(v92.m128i_i64[0] + 56) + 48 * v62 + 8) = v63.m128i_i64[1];
      v48 = *(__m128i **)(a1 + 352);
      if ( v48 == (__m128i *)(*(_QWORD *)(a1 + 368) - 24LL) )
      {
        sub_2BE3350((unsigned __int64 *)(a1 + 304), &v95);
        return v4;
      }
      if ( v48 )
      {
LABEL_85:
        *v48 = _mm_loadu_si128(&v95);
        v48[1].m128i_i64[0] = v96;
        v48 = *(__m128i **)(a1 + 352);
      }
LABEL_65:
      *(_QWORD *)(a1 + 352) = (char *)v48 + 24;
      return v4;
    }
    v2 = *(_DWORD *)(a1 + 152);
  }
  if ( v2 != 21 )
    goto LABEL_3;
  v4 = sub_2BE0030(a1);
  if ( (_BYTE)v4 )
  {
    if ( *(_QWORD *)(a1 + 352) == *(_QWORD *)(a1 + 320) )
      goto LABEL_111;
    v64 = 0;
    if ( v3 && *(_DWORD *)(a1 + 152) == 18 )
      v64 = sub_2BE0030(a1);
    sub_2BDEE20(&v95, (_QWORD *)a1);
    v65 = sub_2BE0960(*(unsigned __int64 **)(a1 + 256), -1, v95.m128i_i64[1], v64);
    *(_QWORD *)(*(_QWORD *)(v95.m128i_i64[0] + 56) + 48 * v96 + 8) = v65;
    v66 = *(_QWORD *)(a1 + 368);
    v48 = *(__m128i **)(a1 + 352);
    v96 = v65;
    if ( v48 == (__m128i *)(v66 - 24) )
    {
      sub_2BE3350((unsigned __int64 *)(a1 + 304), &v95);
      return v4;
    }
    if ( v48 )
      goto LABEL_85;
    goto LABEL_65;
  }
  v2 = *(_DWORD *)(a1 + 152);
LABEL_3:
  if ( v2 == 18 )
  {
    v4 = sub_2BE0030(a1);
    if ( (_BYTE)v4 )
    {
      if ( *(_QWORD *)(a1 + 352) == *(_QWORD *)(a1 + 320) )
        goto LABEL_111;
      v67 = 0;
      if ( v3 && *(_DWORD *)(a1 + 152) == 18 )
        v67 = sub_2BE0030(a1);
      sub_2BDEE20(&v92, (_QWORD *)a1);
      v68 = v93.m128i_i64[0];
      v69 = sub_2BE04C0(*(unsigned __int64 **)(a1 + 256));
      v70 = sub_2BE0960(*(unsigned __int64 **)(a1 + 256), -1, v92.m128i_i64[1], v67);
      v71 = *(_QWORD *)(a1 + 256);
      v96 = v70;
      v95.m128i_i64[1] = v70;
      v95.m128i_i64[0] = v71;
      *(_QWORD *)(*(_QWORD *)(v92.m128i_i64[0] + 56) + 48 * v68 + 8) = v69;
      *(_QWORD *)(*(_QWORD *)(v71 + 56) + 48 * v96 + 8) = v69;
      v96 = v69;
      sub_2BE3450((unsigned __int64 *)(a1 + 304), &v95);
      return v4;
    }
    v2 = *(_DWORD *)(a1 + 152);
  }
  if ( v2 == 12 && (unsigned __int8)sub_2BE0030(a1) )
  {
    if ( *(_QWORD *)(a1 + 352) != *(_QWORD *)(a1 + 320)
      && *(_DWORD *)(a1 + 152) == 26
      && (unsigned __int8)sub_2BE0030(a1) )
    {
      v6 = *(_QWORD *)(a1 + 352);
      if ( v6 == *(_QWORD *)(a1 + 360) )
      {
        v49 = *(const __m128i **)(*(_QWORD *)(a1 + 376) - 8LL);
        v50 = _mm_loadu_si128(v49 + 30);
        v51 = v49[31].m128i_i64[0];
        v81 = v50;
        v82 = v51;
        j_j___libc_free_0(v6);
        v52 = (__int64 *)(*(_QWORD *)(a1 + 376) - 8LL);
        *(_QWORD *)(a1 + 376) = v52;
        v53 = *v52;
        v54 = *v52 + 504;
        *(_QWORD *)(a1 + 360) = v53;
        *(_QWORD *)(a1 + 368) = v54;
        *(_QWORD *)(a1 + 352) = v53 + 480;
      }
      else
      {
        v7 = _mm_loadu_si128((const __m128i *)(v6 - 24));
        v8 = v6 - 24;
        v81 = v7;
        v9 = *(_QWORD *)(v8 + 16);
        *(_QWORD *)(a1 + 352) = v8;
        v82 = v9;
      }
      v10 = *(unsigned __int64 **)(a1 + 256);
      v86.m128i_i32[0] = 10;
      v86.m128i_i64[1] = -1;
      v11 = (__m128i *)v10[8];
      if ( v11 == (__m128i *)v10[9] )
      {
        sub_2BE00E0(v10 + 7, v11, &v86);
        v18 = v10[8];
      }
      else
      {
        if ( v11 )
        {
          *v11 = _mm_loadu_si128(&v86);
          v12 = _mm_loadu_si128(&v87);
          v13 = v86.m128i_i32[0] == 11;
          v11[1] = v12;
          v11[2] = _mm_loadu_si128(&v88);
          if ( v13 )
          {
            v14 = _mm_loadu_si128(&v87);
            v87 = v12;
            v15 = v11[2].m128i_i64[1];
            v11[2].m128i_i64[0] = 0;
            v11[1] = v14;
            v16 = v88.m128i_i64[0];
            v88.m128i_i64[0] = 0;
            v11[2].m128i_i64[0] = v16;
            v17 = v88.m128i_i64[1];
            v88.m128i_i64[1] = v15;
            v11[2].m128i_i64[1] = v17;
          }
          v11 = (__m128i *)v10[8];
        }
        v18 = (unsigned __int64)&v11[3];
        v10[8] = v18;
      }
      v19 = v18 - v10[7];
      if ( (unsigned __int64)v19 <= 0x493E00 )
      {
        if ( v86.m128i_i32[0] == 11 && v88.m128i_i64[0] )
          ((void (__fastcall *)(__m128i *, __m128i *, __int64))v88.m128i_i64[0])(&v87, &v87, 3);
        v20 = *(_QWORD *)(a1 + 256);
        v13 = *(_QWORD *)(a1 + 280) == 0;
        v83.m128i_i64[1] = 0xAAAAAAAAAAAAAAABLL * (v19 >> 4) - 1;
        i = v83.m128i_i64[1];
        v83.m128i_i64[0] = v20;
        if ( v13 )
        {
          v78 = 0;
        }
        else
        {
          v21 = 0;
          v22 = 0;
          do
          {
            v23 = *(_BYTE *)(*(_QWORD *)(a1 + 272) + v22++);
            v21 = 10 * v21 + (int)sub_2BDCD80(v23, 10);
          }
          while ( *(_QWORD *)(a1 + 280) > v22 );
          v78 = (int)v21;
        }
        v24 = *(_DWORD *)(a1 + 152);
        v77 = 0;
        v80 = 0;
        if ( v24 == 25 )
        {
          v72 = sub_2BE0030(a1);
          v13 = v72 == 0;
          v77 = v72;
          v24 = *(_DWORD *)(a1 + 152);
          if ( !v13 && v24 == 26 )
          {
            if ( (unsigned __int8)sub_2BE0030(a1) )
            {
              v77 = 0;
              v80 = (int)sub_2BE08E0(a1, 10) - v78;
            }
            v24 = *(_DWORD *)(a1 + 152);
          }
        }
        if ( v24 == 13 )
        {
          v4 = sub_2BE0030(a1);
          if ( (_BYTE)v4 )
          {
            v74 = 0;
            if ( v3 && *(_DWORD *)(a1 + 152) == 18 )
              v74 = sub_2BE0030(a1);
            if ( v78 > 0 )
            {
              v73 = v4;
              v25 = 0;
              do
              {
                ++v25;
                sub_2BE6410((unsigned __int64 **)&v95, &v81);
                *(_QWORD *)(*(_QWORD *)(v83.m128i_i64[0] + 56) + 48 * i + 8) = v95.m128i_i64[1];
                i = v96;
              }
              while ( v78 != v25 );
              v4 = v73;
            }
            if ( v77 )
            {
              sub_2BE6410((unsigned __int64 **)&v95, &v81);
              v45 = v96;
              v46 = sub_2BE0960(*(unsigned __int64 **)(a1 + 256), -1, v95.m128i_i64[1], v74);
              v47 = v83.m128i_i64[0];
              *(_QWORD *)(*(_QWORD *)(v95.m128i_i64[0] + 56) + 48 * v45 + 8) = v46;
              *(_QWORD *)(*(_QWORD *)(v47 + 56) + 48 * i + 8) = v46;
              i = v46;
LABEL_62:
              v48 = *(__m128i **)(a1 + 352);
              if ( v48 == (__m128i *)(*(_QWORD *)(a1 + 368) - 24LL) )
              {
                sub_2BE3350((unsigned __int64 *)(a1 + 304), &v83);
                return v4;
              }
              if ( v48 )
              {
                *v48 = _mm_loadu_si128(&v83);
                v48[1].m128i_i64[0] = i;
                v48 = *(__m128i **)(a1 + 352);
              }
              goto LABEL_65;
            }
            if ( v80 >= 0 )
            {
              v26 = sub_2BE04C0(*(unsigned __int64 **)(a1 + 256));
              v95 = 0u;
              v79 = v26;
              v96 = 0;
              v97 = 0;
              v98 = 0;
              v99 = 0;
              v100 = 0;
              v101 = 0;
              v102 = 0;
              v103 = 0;
              sub_2BE2EE0(v95.m128i_i64, 0);
              if ( v80 )
              {
                v27 = 0;
                v28 = v74;
                do
                {
                  sub_2BE6410(v85, &v81);
                  v89.m128i_i32[0] = 2;
                  v90.m128i_i8[8] = v28;
                  v32 = *(unsigned __int64 **)(a1 + 256);
                  v89.m128i_i64[1] = (__int64)v85[1];
                  v33 = _mm_loadu_si128(&v89);
                  v34 = _mm_loadu_si128(&v91);
                  v90.m128i_i64[0] = v79;
                  v35 = _mm_loadu_si128(&v90);
                  v92 = v33;
                  v93 = v35;
                  v94 = v34;
                  v36 = (__m128i *)v32[8];
                  if ( v36 == (__m128i *)v32[9] )
                  {
                    v76 = v32;
                    sub_2BE00E0(v32 + 7, v36, &v92);
                    v32 = v76;
                    v42 = v76[8];
                  }
                  else
                  {
                    if ( v36 )
                    {
                      *v36 = v33;
                      v37 = _mm_loadu_si128(&v93);
                      v36[1] = v37;
                      v36[2] = _mm_loadu_si128(&v94);
                      if ( v92.m128i_i32[0] == 11 )
                      {
                        v36[2].m128i_i64[0] = 0;
                        v38 = _mm_loadu_si128(&v93);
                        v93 = v37;
                        v36[1] = v38;
                        v39 = v94.m128i_i64[0];
                        v94.m128i_i64[0] = 0;
                        v40 = v36[2].m128i_i64[1];
                        v36[2].m128i_i64[0] = v39;
                        v41 = v94.m128i_i64[1];
                        v94.m128i_i64[1] = v40;
                        v36[2].m128i_i64[1] = v41;
                      }
                      v36 = (__m128i *)v32[8];
                    }
                    v42 = (unsigned __int64)&v36[3];
                    v32[8] = v42;
                  }
                  v43 = v42 - v32[7];
                  if ( (unsigned __int64)v43 > 0x493E00 )
                    goto LABEL_111;
                  v44 = 0xAAAAAAAAAAAAAAABLL * (v43 >> 4) - 1;
                  if ( v92.m128i_i32[0] == 11 && v94.m128i_i64[0] )
                  {
                    ((void (__fastcall *)(__m128i *, __m128i *, __int64))v94.m128i_i64[0])(&v93, &v93, 3);
                    v44 = 0xAAAAAAAAAAAAAAABLL * (v43 >> 4) - 1;
                  }
                  if ( v89.m128i_i32[0] == 11 && v91.m128i_i64[0] )
                  {
                    v75 = v44;
                    ((void (__fastcall *)(__m128i *, __m128i *, __int64))v91.m128i_i64[0])(&v90, &v90, 3);
                    v44 = v75;
                  }
                  v29 = v100;
                  v92.m128i_i64[0] = v44;
                  if ( v100 == (unsigned __int64 *)(v102 - 8) )
                  {
                    sub_2BE2FD0((unsigned __int64 *)&v95, &v92);
                    v44 = v92.m128i_i64[0];
                  }
                  else
                  {
                    if ( v100 )
                    {
                      *v100 = v44;
                      v29 = v100;
                    }
                    v100 = v29 + 1;
                  }
                  v30 = v83.m128i_i64[0];
                  ++v27;
                  v31 = (__int64)v85[2];
                  *(_QWORD *)(*(_QWORD *)(v83.m128i_i64[0] + 56) + 48 * i + 8) = v44;
                  i = v31;
                }
                while ( v80 > v27 );
                v4 = (unsigned __int8)v4;
              }
              else
              {
                v30 = v83.m128i_i64[0];
                v31 = i;
              }
              *(_QWORD *)(*(_QWORD *)(v30 + 56) + 48 * v31 + 8) = v79;
              v55 = (unsigned __int64)v100;
              for ( i = v79; v100 != (unsigned __int64 *)v96; v55 = (unsigned __int64)v100 )
              {
                v59 = *(_QWORD *)(*(_QWORD *)(a1 + 256) + 56LL);
                if ( v101 == v55 )
                {
                  v57 = v59 + 48LL * *(_QWORD *)(*(v103 - 1) + 504);
                  j_j___libc_free_0(v55);
                  v60 = *--v103 + 512;
                  v101 = *v103;
                  v102 = v60;
                  v100 = (unsigned __int64 *)(v101 + 504);
                }
                else
                {
                  v56 = *(_QWORD *)(v55 - 8);
                  v100 = (unsigned __int64 *)(v55 - 8);
                  v57 = v59 + 48 * v56;
                }
                v58 = *(_QWORD *)(v57 + 16);
                *(_QWORD *)(v57 + 16) = *(_QWORD *)(v57 + 8);
                *(_QWORD *)(v57 + 8) = v58;
              }
              sub_2BE1050((unsigned __int64 *)&v95);
              goto LABEL_62;
            }
          }
        }
      }
    }
LABEL_111:
    abort();
  }
  return 0;
}
