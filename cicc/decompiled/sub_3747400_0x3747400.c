// Function: sub_3747400
// Address: 0x3747400
//
__int64 __fastcall sub_3747400(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  int v6; // edx
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rcx
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rdi
  __int64 (*v13)(); // rax
  unsigned __int16 *v14; // rax
  __int64 v15; // r8
  unsigned __int64 *v16; // r9
  __int32 v17; // edx
  unsigned __int16 *v18; // r10
  __int64 v19; // rax
  __int64 v20; // r15
  unsigned __int64 v21; // r11
  __m128i *v22; // r13
  unsigned __int64 v23; // rdx
  __m128i *v24; // rax
  int v25; // r14d
  _QWORD *v26; // r13
  __int64 v27; // rdx
  __int64 v28; // r15
  __int64 v29; // rsi
  __int64 v30; // rax
  __int64 *v31; // r14
  __int64 v32; // r15
  _QWORD *v33; // r13
  __int64 v34; // r15
  __int64 v35; // rdx
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // rdx
  unsigned __int64 v39; // r14
  const __m128i *v40; // rdx
  __int64 v41; // rsi
  __int64 v42; // rax
  __int64 v43; // r15
  __int64 v44; // r14
  __int64 *v45; // r9
  _QWORD *v46; // r13
  _QWORD *v47; // rax
  __int64 *v48; // r9
  __int64 v49; // r15
  __int64 v50; // rdx
  __int64 v51; // rax
  __int64 v52; // rdx
  __int64 v53; // rdx
  __int8 *v55; // r13
  unsigned __int16 *v56; // [rsp+0h] [rbp-590h]
  unsigned __int8 v57; // [rsp+Fh] [rbp-581h]
  int v58; // [rsp+18h] [rbp-578h]
  __int64 v59; // [rsp+18h] [rbp-578h]
  _DWORD *v60; // [rsp+18h] [rbp-578h]
  __int64 *v61; // [rsp+18h] [rbp-578h]
  __int64 *v62; // [rsp+18h] [rbp-578h]
  __int64 *v63; // [rsp+18h] [rbp-578h]
  unsigned __int64 *v64; // [rsp+18h] [rbp-578h]
  unsigned __int64 *v65; // [rsp+18h] [rbp-578h]
  __m128i v66; // [rsp+20h] [rbp-570h] BYREF
  __int64 v67; // [rsp+30h] [rbp-560h]
  __int64 v68; // [rsp+38h] [rbp-558h]
  __int64 v69; // [rsp+40h] [rbp-550h]
  unsigned __int64 v70; // [rsp+50h] [rbp-540h] BYREF
  __int64 v71; // [rsp+58h] [rbp-538h]
  _DWORD v72[4]; // [rsp+60h] [rbp-530h] BYREF
  __int64 v73; // [rsp+70h] [rbp-520h]
  __int64 v74; // [rsp+78h] [rbp-518h]
  unsigned int v75; // [rsp+88h] [rbp-508h]
  __int64 v76; // [rsp+98h] [rbp-4F8h]
  __int64 v77; // [rsp+A0h] [rbp-4F0h]

  v6 = *(_DWORD *)(a2 + 4);
  v70 = (unsigned __int64)v72;
  v7 = v6 & 0x7FFFFFF;
  v71 = 0x2000000000LL;
  v8 = *(_QWORD *)(a2 - 32 * v7);
  if ( *(_DWORD *)(v8 + 32) <= 0x40u )
    v9 = *(_QWORD *)(v8 + 24);
  else
    v9 = **(_QWORD **)(v8 + 24);
  v73 = 0;
  v74 = v9;
  LODWORD(v71) = 1;
  v72[0] = v72[0] & 0xFFF00000 | 1;
  v10 = *(_QWORD *)(a2 + 32 * (1 - v7));
  if ( *(_DWORD *)(v10 + 32) <= 0x40u )
    v11 = *(_QWORD *)(v10 + 24);
  else
    v11 = **(_QWORD **)(v10 + 24);
  v77 = v11;
  v76 = 0;
  v75 = v75 & 0xFFF00000 | 1;
  LODWORD(v71) = 2;
  v57 = sub_3746D50(a1, (__int64)&v70, (unsigned __int8 *)a2, 2u, a5, (__int64)&v70);
  if ( v57 )
  {
    v12 = a1[16];
    v13 = *(__int64 (**)())(*(_QWORD *)v12 + 2384LL);
    if ( v13 == sub_302E260 )
      BUG();
    v14 = (unsigned __int16 *)((__int64 (__fastcall *)(__int64, _QWORD))v13)(v12, (*(_WORD *)(a2 + 2) >> 2) & 0x3FF);
    v16 = &v70;
    v17 = *v14;
    v18 = v14;
    if ( (_WORD)v17 )
    {
      v19 = (unsigned int)v71;
      LODWORD(v20) = 0;
      do
      {
        v21 = v19 + 1;
        v22 = &v66;
        v66.m128i_i32[2] = v17;
        v23 = v70;
        v67 = 0;
        v68 = 0;
        v69 = 0;
        v66.m128i_i64[0] = 0x430000000LL;
        if ( v19 + 1 > (unsigned __int64)HIDWORD(v71) )
        {
          if ( v70 > (unsigned __int64)&v66 )
          {
            v56 = v18;
LABEL_41:
            v65 = v16;
            sub_C8D5F0((__int64)v16, v72, v21, 0x28u, v15, (__int64)v16);
            v23 = v70;
            v19 = (unsigned int)v71;
            v16 = v65;
            v18 = v56;
            goto LABEL_10;
          }
          v56 = v18;
          if ( (unsigned __int64)&v66 >= v70 + 40 * v19 )
            goto LABEL_41;
          v55 = &v66.m128i_i8[-v70];
          v64 = v16;
          sub_C8D5F0((__int64)v16, v72, v21, 0x28u, v15, (__int64)v16);
          v23 = v70;
          v19 = (unsigned int)v71;
          v18 = v56;
          v16 = v64;
          v22 = (__m128i *)&v55[v70];
        }
LABEL_10:
        v24 = (__m128i *)(v23 + 40 * v19);
        *v24 = _mm_loadu_si128(v22);
        v24[1] = _mm_loadu_si128(v22 + 1);
        v24[2].m128i_i64[0] = v22[2].m128i_i64[0];
        v20 = (unsigned int)(v20 + 1);
        v19 = (unsigned int)(v71 + 1);
        LODWORD(v71) = v71 + 1;
        v17 = v18[v20];
      }
      while ( (_WORD)v17 );
    }
    v25 = 0;
    v26 = sub_3740F30(
            *(_QWORD *)(a1[5] + 744),
            *(__int64 **)(a1[5] + 752),
            (__int64)(a1 + 10),
            *(_QWORD *)(a1[15] + 8) - 40LL * *(unsigned int *)(a1[15] + 64));
    v28 = v27;
    v58 = *(unsigned __int16 *)(*(_QWORD *)(v27 + 16) + 2LL);
    if ( *(_WORD *)(*(_QWORD *)(v27 + 16) + 2LL) )
    {
      do
      {
        ++v25;
        v66.m128i_i64[0] = 1;
        v67 = 0;
        v68 = 0;
        sub_2E8EAD0(v28, (__int64)v26, &v66);
      }
      while ( v25 != v58 );
    }
    v29 = a1[10];
    v30 = a1[5];
    v31 = *(__int64 **)(v30 + 752);
    v32 = *(_QWORD *)(a1[15] + 8) - 1040LL;
    v33 = *(_QWORD **)(*(_QWORD *)(v30 + 744) + 32LL);
    v59 = *(_QWORD *)(v30 + 744);
    v66.m128i_i64[0] = v29;
    if ( v29 )
      sub_B96E90((__int64)&v66, v29, 1);
    v34 = (__int64)sub_2E7B380(v33, v32, (unsigned __int8 **)&v66, 0);
    if ( v66.m128i_i64[0] )
      sub_B91220((__int64)&v66, v66.m128i_i64[0]);
    sub_2E31040((__int64 *)(v59 + 40), v34);
    v35 = *v31;
    v36 = *(_QWORD *)v34;
    *(_QWORD *)(v34 + 8) = v31;
    v35 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)v34 = v35 | v36 & 7;
    *(_QWORD *)(v35 + 8) = v34;
    *v31 = v34 | *v31 & 7;
    v37 = a1[11];
    if ( v37 )
      sub_2E882B0(v34, (__int64)v33, v37);
    v38 = a1[12];
    if ( v38 )
      sub_2E88680(v34, (__int64)v33, v38);
    v39 = v70;
    v60 = (_DWORD *)(v70 + 40LL * (unsigned int)v71);
    if ( v60 != (_DWORD *)v70 )
    {
      do
      {
        v40 = (const __m128i *)v39;
        v39 += 40LL;
        sub_2E8EAD0(v34, (__int64)v33, v40);
      }
      while ( v60 != (_DWORD *)v39 );
    }
    v41 = a1[10];
    v42 = a1[5];
    v43 = *(_QWORD *)(a1[15] + 8) - 40LL * *(unsigned int *)(a1[15] + 68);
    v44 = *(_QWORD *)(v42 + 744);
    v45 = *(__int64 **)(v42 + 752);
    v46 = *(_QWORD **)(v44 + 32);
    v66.m128i_i64[0] = v41;
    if ( v41 )
    {
      v61 = v45;
      sub_B96E90((__int64)&v66, v41, 1);
      v45 = v61;
    }
    v62 = v45;
    v47 = sub_2E7B380(v46, v43, (unsigned __int8 **)&v66, 0);
    v48 = v62;
    v49 = (__int64)v47;
    if ( v66.m128i_i64[0] )
    {
      sub_B91220((__int64)&v66, v66.m128i_i64[0]);
      v48 = v62;
    }
    v63 = v48;
    sub_2E31040((__int64 *)(v44 + 40), v49);
    v50 = *v63;
    v51 = *(_QWORD *)v49 & 7LL;
    *(_QWORD *)(v49 + 8) = v63;
    v50 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)v49 = v50 | v51;
    *(_QWORD *)(v50 + 8) = v49;
    *v63 = v49 | *v63 & 7;
    v52 = a1[11];
    if ( v52 )
      sub_2E882B0(v49, (__int64)v46, v52);
    v53 = a1[12];
    if ( v53 )
      sub_2E88680(v49, (__int64)v46, v53);
    v66.m128i_i64[0] = 1;
    v67 = 0;
    v68 = 0;
    sub_2E8EAD0(v49, (__int64)v46, &v66);
    v66.m128i_i64[0] = 1;
    v67 = 0;
    v68 = 0;
    sub_2E8EAD0(v49, (__int64)v46, &v66);
    *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(a1[5] + 8) + 48LL) + 39LL) = 1;
  }
  if ( (_DWORD *)v70 != v72 )
    _libc_free(v70);
  return v57;
}
