// Function: sub_37954A0
// Address: 0x37954a0
//
unsigned __int8 *__fastcall sub_37954A0(
        __int64 *a1,
        unsigned __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6)
{
  unsigned __int16 *v7; // rdx
  int v8; // eax
  __int64 v9; // rdx
  unsigned __int16 v10; // ax
  __int64 v11; // rdx
  const __m128i *v12; // rcx
  __int64 v13; // rsi
  unsigned int v14; // ebx
  __m128i v15; // xmm0
  __int64 v16; // r12
  __int64 *v17; // rax
  __int64 *v18; // r12
  unsigned __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rbx
  unsigned int v22; // r12d
  __int64 v23; // r15
  __int64 *v24; // rax
  const __m128i *v25; // roff
  __int64 v26; // rcx
  __int64 v27; // rax
  int v28; // edx
  __int64 v29; // rax
  unsigned __int16 v30; // si
  __int64 v31; // r8
  bool v32; // al
  unsigned __int8 *v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rcx
  __int64 v36; // r8
  _QWORD *v37; // r9
  unsigned __int8 *v38; // r10
  __int64 v39; // r11
  __int64 v40; // r8
  __int64 v41; // rax
  __int64 v42; // rdi
  __int64 *v43; // rax
  int v44; // r9d
  _QWORD *v45; // r13
  unsigned int *v46; // rax
  __int64 v47; // rdx
  unsigned __int8 *v48; // r13
  __int64 v50; // rdx
  unsigned __int64 v51; // rdx
  __int128 v52; // [rsp-10h] [rbp-190h]
  __int64 v54; // [rsp+20h] [rbp-160h]
  int v55; // [rsp+34h] [rbp-14Ch]
  __int64 v56; // [rsp+38h] [rbp-148h]
  unsigned __int8 *v57; // [rsp+40h] [rbp-140h]
  __int64 v58; // [rsp+48h] [rbp-138h]
  __int64 v59; // [rsp+50h] [rbp-130h]
  __int64 v60; // [rsp+50h] [rbp-130h]
  __int16 v61; // [rsp+52h] [rbp-12Eh]
  __int64 v62; // [rsp+58h] [rbp-128h]
  int v63; // [rsp+58h] [rbp-128h]
  __m128i v64; // [rsp+60h] [rbp-120h] BYREF
  __m128i v65; // [rsp+70h] [rbp-110h]
  __m128i v66; // [rsp+80h] [rbp-100h]
  __m128i v67; // [rsp+90h] [rbp-F0h]
  __int64 v68; // [rsp+A0h] [rbp-E0h] BYREF
  int v69; // [rsp+A8h] [rbp-D8h]
  unsigned __int16 v70; // [rsp+B0h] [rbp-D0h] BYREF
  __int64 v71; // [rsp+B8h] [rbp-C8h]
  char v72[32]; // [rsp+C0h] [rbp-C0h] BYREF
  unsigned __int16 v73; // [rsp+E0h] [rbp-A0h] BYREF
  __int64 v74; // [rsp+E8h] [rbp-98h]
  __int16 v75; // [rsp+F0h] [rbp-90h]
  __int64 v76; // [rsp+F8h] [rbp-88h]
  __int64 *v77; // [rsp+100h] [rbp-80h] BYREF
  __int64 v78; // [rsp+108h] [rbp-78h]
  __int64 v79; // [rsp+110h] [rbp-70h] BYREF
  __int32 v80; // [rsp+118h] [rbp-68h]

  v7 = *(unsigned __int16 **)(a2 + 48);
  v8 = *v7;
  v9 = *((_QWORD *)v7 + 1);
  LOWORD(v77) = v8;
  v78 = v9;
  if ( (_WORD)v8 )
  {
    v10 = word_4456580[v8 - 1];
    v11 = 0;
  }
  else
  {
    v10 = sub_3009970((__int64)&v77, a2, v9, a4, a5);
  }
  v12 = *(const __m128i **)(a2 + 40);
  v13 = *(_QWORD *)(a2 + 80);
  v14 = *(_DWORD *)(a2 + 64);
  v15 = _mm_loadu_si128(v12);
  v73 = v10;
  v74 = v11;
  v75 = 1;
  v76 = 0;
  v68 = v13;
  if ( v13 )
  {
    v64 = v15;
    sub_B96E90((__int64)&v68, v13, 1);
    v15 = _mm_load_si128(&v64);
  }
  v16 = v14;
  v69 = *(_DWORD *)(a2 + 72);
  v77 = &v79;
  v78 = 0x400000000LL;
  if ( v14 )
  {
    if ( v14 > 4uLL )
    {
      v64 = v15;
      sub_C8D5F0((__int64)&v77, &v79, v14, 0x10u, a5, a6);
      v51 = (unsigned __int64)v77;
      v15 = _mm_load_si128(&v64);
      v18 = &v77[2 * v14];
      v17 = &v77[2 * (unsigned int)v78];
      if ( v18 == v17 )
      {
        v65 = v15;
        LODWORD(v78) = v14;
        *v77 = v15.m128i_i64[0];
        *(_DWORD *)(v51 + 8) = v65.m128i_i32[2];
LABEL_12:
        v20 = v14 - 2;
        v21 = 2;
        HIWORD(v22) = v61;
        v23 = 40;
        v62 = 40 * v20 + 80;
        while ( 1 )
        {
          v25 = (const __m128i *)(v23 + *(_QWORD *)(a2 + 40));
          v26 = v25->m128i_i64[0];
          v27 = v25->m128i_u32[2];
          v64 = _mm_loadu_si128(v25);
          v28 = v27;
          v29 = *(_QWORD *)(v26 + 48) + 16 * v27;
          v30 = *(_WORD *)v29;
          v31 = *(_QWORD *)(v29 + 8);
          v70 = v30;
          v71 = v31;
          if ( v30 )
          {
            if ( (unsigned __int16)(v30 - 17) <= 0xD3u )
              goto LABEL_17;
          }
          else
          {
            v55 = v28;
            v56 = v31;
            v59 = v26;
            v32 = sub_30070B0((__int64)&v70);
            v26 = v59;
            v31 = v56;
            v28 = v55;
            if ( v32 )
            {
LABEL_17:
              LOWORD(v22) = v30;
              sub_2FE6CC0((__int64)v72, *a1, *(_QWORD *)(a1[1] + 64), v22, v31);
              if ( v72[0] == 5 )
              {
                v26 = sub_37946F0((__int64)a1, v64.m128i_u64[0], v64.m128i_i64[1]);
              }
              else
              {
                v60 = a1[1];
                v33 = sub_3400EE0(v60, 0, (__int64)&v68, 0, v15);
                v37 = (_QWORD *)v60;
                v38 = v33;
                v39 = v34;
                if ( v70 )
                {
                  v40 = 0;
                  LOWORD(v41) = word_4456580[v70 - 1];
                }
                else
                {
                  v57 = v33;
                  v58 = v34;
                  v41 = sub_3009970((__int64)&v70, 0, v34, v35, v36);
                  v38 = v57;
                  v39 = v58;
                  v54 = v41;
                  v37 = (_QWORD *)v60;
                  v40 = v50;
                }
                v42 = v54;
                *((_QWORD *)&v52 + 1) = v39;
                *(_QWORD *)&v52 = v38;
                LOWORD(v42) = v41;
                v54 = v42;
                v26 = (__int64)sub_3406EB0(
                                 v37,
                                 0x9Eu,
                                 (__int64)&v68,
                                 (unsigned int)v42,
                                 v40,
                                 (__int64)v37,
                                 *(_OWORD *)&v64,
                                 v52);
              }
            }
          }
          v23 += 40;
          v24 = &v77[v21];
          v21 += 2;
          *v24 = v26;
          *((_DWORD *)v24 + 2) = v28;
          if ( v62 == v23 )
            goto LABEL_22;
        }
      }
    }
    else
    {
      v17 = &v79;
      v18 = &v79 + 2 * v14;
    }
    do
    {
      if ( v17 )
      {
        *v17 = 0;
        *((_DWORD *)v17 + 2) = 0;
      }
      v17 += 2;
    }
    while ( v17 != v18 );
    v19 = (unsigned __int64)v77;
    v67 = v15;
    LODWORD(v78) = v14;
    *v77 = v15.m128i_i64[0];
    *(_DWORD *)(v19 + 8) = v67.m128i_i32[2];
    if ( v14 != 1 )
      goto LABEL_12;
LABEL_22:
    v43 = v77;
    v16 = (unsigned int)v78;
  }
  else
  {
    v66 = v15;
    v79 = v15.m128i_i64[0];
    v80 = v15.m128i_i32[2];
    v43 = &v79;
  }
  v44 = *(_DWORD *)(a2 + 28);
  v64.m128i_i64[1] = v16;
  v45 = (_QWORD *)a1[1];
  v63 = v44;
  v64.m128i_i64[0] = (__int64)v43;
  v46 = (unsigned int *)sub_33E5830(v45, &v73, 2);
  v48 = sub_3410740(
          v45,
          *(unsigned int *)(a2 + 24),
          (__int64)&v68,
          v46,
          v47,
          v63,
          v15,
          (__m128i *)v64.m128i_i64[0],
          v64.m128i_i64[1]);
  sub_3760E70((__int64)a1, a2, 1, (unsigned __int64)v48, 1);
  if ( v77 != &v79 )
    _libc_free((unsigned __int64)v77);
  if ( v68 )
    sub_B91220((__int64)&v68, v68);
  return v48;
}
