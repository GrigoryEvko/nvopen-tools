// Function: sub_1166410
// Address: 0x1166410
//
unsigned __int8 *__fastcall sub_1166410(__m128i *a1, __int64 a2)
{
  __m128i v4; // xmm0
  __m128i v5; // xmm1
  __m128i v6; // xmm3
  __int64 v7; // rax
  unsigned __int8 *v8; // rdi
  unsigned __int8 *v9; // rsi
  __int64 v10; // rax
  void *v12; // r9
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r8
  unsigned __int8 *v17; // rdi
  __m128i v18; // xmm5
  unsigned __int64 v19; // xmm6_8
  __int64 v20; // rax
  unsigned __int8 *v21; // r14
  __int64 v22; // rbx
  __m128i v23; // xmm7
  __int64 v24; // rax
  __int64 v25; // r12
  __int64 v26; // rbx
  __int64 v27; // r15
  unsigned int **v28; // r12
  _BYTE *v29; // rax
  __int64 v30; // rax
  char v31; // al
  unsigned int **v32; // rdi
  __int64 v33; // rax
  __int64 v34; // r15
  __int64 v35; // rbx
  __int64 v36; // rbx
  __int64 v37; // r12
  __int64 v38; // rdx
  unsigned int v39; // esi
  char v40; // al
  __m128i *v41; // rsi
  __int64 v42; // rcx
  __m128i *v43; // rdi
  __int64 v44; // rax
  char v45; // al
  unsigned int **v46; // rdi
  __int64 v47; // rdx
  _BYTE *v48; // rcx
  __int64 v49; // rax
  __int64 v50; // r12
  __int64 *v51; // r15
  __m128i v52; // rax
  __int64 v53; // r12
  __int64 v54; // r15
  __int64 v55; // rdx
  unsigned int v56; // esi
  __int64 v57; // rax
  bool v58; // al
  unsigned int **v59; // r12
  __int64 v60; // rax
  __int64 *v61; // r15
  __m128i v62; // rax
  __m128i v63; // rax
  __int64 v64; // [rsp-8h] [rbp-C8h]
  __int64 v65; // [rsp+0h] [rbp-C0h]
  __int64 *v66; // [rsp+0h] [rbp-C0h]
  unsigned __int8 *v67; // [rsp+8h] [rbp-B8h]
  _BYTE *v68[4]; // [rsp+10h] [rbp-B0h] BYREF
  __int16 v69; // [rsp+30h] [rbp-90h]
  __m128i v70; // [rsp+40h] [rbp-80h] BYREF
  __m128i v71; // [rsp+50h] [rbp-70h]
  __m128i v72; // [rsp+60h] [rbp-60h]
  __m128i v73; // [rsp+70h] [rbp-50h]
  __int64 v74; // [rsp+80h] [rbp-40h]

  v4 = _mm_loadu_si128(a1 + 6);
  v5 = _mm_loadu_si128(a1 + 7);
  v6 = _mm_loadu_si128(a1 + 9);
  v7 = a1[10].m128i_i64[0];
  v72 = _mm_loadu_si128(a1 + 8);
  v8 = *(unsigned __int8 **)(a2 - 64);
  v72.m128i_i64[1] = a2;
  v9 = *(unsigned __int8 **)(a2 - 32);
  v74 = v7;
  v70 = v4;
  v71 = v5;
  v73 = v6;
  v10 = sub_101AFD0(v8, v9, &v70);
  if ( v10 )
    return sub_F162A0((__int64)a1, a2, v10);
  v12 = sub_F0F270((__int64)a1, (unsigned __int8 *)a2);
  if ( !v12 )
  {
    v12 = sub_1166190(a1, a2);
    if ( !v12 )
    {
      v12 = sub_11597C0((__int64)a1, a2, v13, v14, v15, 0);
      if ( !v12 )
      {
        v12 = sub_11560B0((unsigned __int8 *)a2, (__int64)a1);
        if ( !v12 )
        {
          v17 = *(unsigned __int8 **)(a2 - 32);
          v18 = _mm_loadu_si128(a1 + 7);
          v19 = _mm_loadu_si128(a1 + 8).m128i_u64[0];
          v20 = a1[10].m128i_i64[0];
          v67 = v17;
          v21 = *(unsigned __int8 **)(a2 - 64);
          v22 = *(_QWORD *)(a2 + 8);
          v70 = _mm_loadu_si128(a1 + 6);
          v23 = _mm_loadu_si128(a1 + 9);
          v72.m128i_i64[0] = v19;
          v74 = v20;
          v72.m128i_i64[1] = a2;
          v71 = v18;
          v73 = v23;
          if ( (unsigned __int8)sub_9A1DB0(v17, 1, 0, (__int64)&v70, v16) )
          {
            v24 = sub_AD62B0(v22);
            v25 = a1[2].m128i_i64[0];
            v26 = v24;
            v69 = 257;
            v27 = (*(__int64 (__fastcall **)(_QWORD, __int64, unsigned __int8 *, __int64, _QWORD, _QWORD))(**(_QWORD **)(v25 + 80) + 32LL))(
                    *(_QWORD *)(v25 + 80),
                    13,
                    v17,
                    v24,
                    0,
                    0);
            if ( !v27 )
            {
              v72.m128i_i16[0] = 257;
              v27 = sub_B504D0(13, (__int64)v17, v26, (__int64)&v70, 0, 0);
              (*(void (__fastcall **)(_QWORD, __int64, _BYTE **, _QWORD, _QWORD))(**(_QWORD **)(v25 + 88) + 16LL))(
                *(_QWORD *)(v25 + 88),
                v27,
                v68,
                *(_QWORD *)(v25 + 56),
                *(_QWORD *)(v25 + 64));
              v36 = *(_QWORD *)v25;
              v37 = *(_QWORD *)v25 + 16LL * *(unsigned int *)(v25 + 8);
              while ( v37 != v36 )
              {
                v38 = *(_QWORD *)(v36 + 8);
                v39 = *(_DWORD *)v36;
                v36 += 16;
                sub_B99FD0(v27, v39, v38);
              }
            }
            v72.m128i_i16[0] = 257;
            return (unsigned __int8 *)sub_B504D0(28, (__int64)v21, v27, (__int64)&v70, 0, 0);
          }
          v70.m128i_i64[0] = 0;
          if ( (unsigned __int8)sub_993A50(&v70, (__int64)v21) )
          {
            v28 = (unsigned int **)a1[2].m128i_i64[0];
            v72.m128i_i16[0] = 257;
            v29 = (_BYTE *)sub_AD64C0(v22, 1, 0);
            v30 = sub_92B530(v28, 0x21u, (__int64)v17, v29, (__int64)&v70);
            v72.m128i_i16[0] = 257;
            return (unsigned __int8 *)sub_B520B0(v30, v22, (__int64)&v70, 0, 0);
          }
          else
          {
            v70.m128i_i64[0] = 0;
            v31 = sub_1006860((__int64 **)&v70, (__int64)v17);
            v12 = 0;
            if ( v31 )
            {
              if ( !sub_98EF80(v21, 0, 0, 0, 0) )
              {
                v51 = (__int64 *)a1[2].m128i_i64[0];
                v52.m128i_i64[0] = (__int64)sub_BD5D20((__int64)v21);
                v72.m128i_i16[0] = 773;
                v70 = v52;
                v71.m128i_i64[0] = (__int64)".fr";
                v21 = (unsigned __int8 *)sub_1156690(v51, (__int64)v21, (__int64)&v70);
              }
              v32 = (unsigned int **)a1[2].m128i_i64[0];
              v72.m128i_i16[0] = 257;
              v33 = sub_92B530(v32, 0x24u, (__int64)v21, v67, (__int64)&v70);
              v34 = a1[2].m128i_i64[0];
              v65 = v33;
              v69 = 257;
              v35 = (*(__int64 (__fastcall **)(_QWORD, __int64, unsigned __int8 *, unsigned __int8 *, _QWORD, _QWORD))(**(_QWORD **)(v34 + 80) + 32LL))(
                      *(_QWORD *)(v34 + 80),
                      15,
                      v21,
                      v67,
                      0,
                      0);
              if ( !v35 )
              {
                v72.m128i_i16[0] = 257;
                v35 = sub_B504D0(15, (__int64)v21, (__int64)v67, (__int64)&v70, 0, 0);
                (*(void (__fastcall **)(_QWORD, __int64, _BYTE **, _QWORD, _QWORD))(**(_QWORD **)(v34 + 88) + 16LL))(
                  *(_QWORD *)(v34 + 88),
                  v35,
                  v68,
                  *(_QWORD *)(v34 + 56),
                  *(_QWORD *)(v34 + 64));
                v53 = *(_QWORD *)v34;
                v54 = *(_QWORD *)v34 + 16LL * *(unsigned int *)(v34 + 8);
                while ( v54 != v53 )
                {
                  v55 = *(_QWORD *)(v53 + 8);
                  v56 = *(_DWORD *)v53;
                  v53 += 16;
                  sub_B99FD0(v35, v56, v55);
                }
              }
              v72.m128i_i16[0] = 257;
              return sub_109FEA0(v65, (__int64)v21, v35, (const char **)&v70, 0, 0, 0);
            }
            if ( *v17 == 69
              && (v57 = *((_QWORD *)v17 - 4)) != 0
              && (v68[0] = *((_BYTE **)v17 - 4), v58 = sub_1001970(*(_QWORD *)(v57 + 8), 1), v12 = 0, v58) )
            {
              if ( !sub_98EF80(v21, 0, 0, 0, 0) )
              {
                v61 = (__int64 *)a1[2].m128i_i64[0];
                v62.m128i_i64[0] = (__int64)sub_BD5D20((__int64)v21);
                v72.m128i_i16[0] = 773;
                v70 = v62;
                v71.m128i_i64[0] = (__int64)".frozen";
                v21 = (unsigned __int8 *)sub_1156690(v61, (__int64)v21, (__int64)&v70);
              }
              v59 = (unsigned int **)a1[2].m128i_i64[0];
              v72.m128i_i16[0] = 257;
              v60 = sub_AD62B0(v22);
              v47 = (__int64)v21;
              v48 = (_BYTE *)v60;
              v46 = v59;
            }
            else
            {
              v70 = (__m128i)(unsigned __int64)v68;
              if ( *v21 != 42 )
                return (unsigned __int8 *)v12;
              if ( !*((_QWORD *)v21 - 8) )
                return (unsigned __int8 *)v12;
              v68[0] = *((_BYTE **)v21 - 8);
              v40 = sub_993A50((_QWORD **)&v70.m128i_i64[1], *((_QWORD *)v21 - 4));
              v12 = 0;
              if ( !v40 )
                return (unsigned __int8 *)v12;
              v41 = a1 + 6;
              v42 = 18;
              v43 = &v70;
              while ( v42 )
              {
                v43->m128i_i32[0] = v41->m128i_i32[0];
                v41 = (__m128i *)((char *)v41 + 4);
                v43 = (__m128i *)((char *)v43 + 4);
                --v42;
              }
              v72.m128i_i64[1] = a2;
              v44 = sub_1016CC0(0x24u, v68[0], v67, v70.m128i_i64);
              v12 = 0;
              if ( !v44 )
                return (unsigned __int8 *)v12;
              v70.m128i_i64[0] = 0;
              v45 = sub_993A50(&v70, v44);
              v12 = 0;
              if ( !v45 )
                return (unsigned __int8 *)v12;
              if ( !sub_98EF80(v21, 0, 0, 0, 0) )
              {
                v66 = (__int64 *)a1[2].m128i_i64[0];
                v63.m128i_i64[0] = (__int64)sub_BD5D20((__int64)v21);
                v72.m128i_i16[0] = 773;
                v70 = v63;
                v71.m128i_i64[0] = (__int64)".frozen";
                v21 = (unsigned __int8 *)sub_1156690(v66, (__int64)v21, (__int64)&v70);
              }
              v46 = (unsigned int **)a1[2].m128i_i64[0];
              v47 = (__int64)v21;
              v72.m128i_i16[0] = 257;
              v48 = v67;
            }
            v49 = sub_92B530(v46, 0x20u, v47, v48, (__int64)&v70);
            v72.m128i_i16[0] = 257;
            v50 = v49;
            v64 = sub_AD6530(v22, 32);
            return sub_109FEA0(v50, v64, (__int64)v21, (const char **)&v70, 0, 0, 0);
          }
        }
      }
    }
  }
  return (unsigned __int8 *)v12;
}
