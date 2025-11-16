// Function: sub_ED9210
// Address: 0xed9210
//
__int64 __fastcall sub_ED9210(__int64 *a1, __int64 *a2, __int64 *a3, _QWORD *a4, __int64 *a5, __int64 *a6)
{
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rdx
  bool v11; // cf
  unsigned __int64 v12; // rax
  __int64 v13; // rbx
  __int64 v14; // rax
  __int64 v15; // r11
  __int64 v16; // r9
  __int64 v17; // r10
  __int64 v18; // rcx
  __int64 v19; // r12
  __int64 v20; // rdx
  __int64 v21; // rdi
  __int64 v22; // r8
  __int64 v23; // rsi
  _QWORD *v24; // rbx
  __int64 v25; // r12
  __int64 *v26; // r15
  __int64 v27; // rax
  __int64 v28; // rax
  __m128i v29; // xmm1
  _QWORD *v30; // rax
  _QWORD *v31; // r13
  _QWORD *v32; // rdi
  _QWORD *v33; // rbx
  _QWORD *v34; // r14
  __int64 v35; // rdi
  __int64 *v36; // rdx
  __int64 v37; // rcx
  __int64 v38; // rsi
  __m128i v39; // xmm0
  __int64 v40; // rsi
  __int64 v42; // rsi
  __int64 v43; // r13
  __int64 v44; // rdx
  __int64 v45; // rax
  __int64 v46; // rdx
  _QWORD *v47; // [rsp+0h] [rbp-70h]
  __int64 v48; // [rsp+8h] [rbp-68h]
  __int64 v49; // [rsp+10h] [rbp-60h]
  __int64 v51; // [rsp+20h] [rbp-50h]
  __int64 v52; // [rsp+28h] [rbp-48h]
  __int64 v54; // [rsp+30h] [rbp-40h]
  _QWORD *v55; // [rsp+38h] [rbp-38h]
  __int64 v56; // [rsp+38h] [rbp-38h]
  __int64 v57; // [rsp+38h] [rbp-38h]
  __int64 v58; // [rsp+38h] [rbp-38h]

  v48 = a1[1];
  v52 = *a1;
  v7 = 0xCCCCCCCCCCCCCCCDLL * ((v48 - *a1) >> 4);
  if ( v7 == 0x199999999999999LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v8 = 1;
  if ( v7 )
    v8 = 0xCCCCCCCCCCCCCCCDLL * ((v48 - *a1) >> 4);
  v11 = __CFADD__(v8, v7);
  v12 = v8 - 0x3333333333333333LL * ((v48 - *a1) >> 4);
  v13 = (__int64)a2 - v52;
  if ( v11 )
  {
    v44 = 0x7FFFFFFFFFFFFFD0LL;
  }
  else
  {
    if ( !v12 )
    {
      v49 = 0;
      v14 = 80;
      v51 = 0;
      goto LABEL_7;
    }
    if ( v12 > 0x199999999999999LL )
      v12 = 0x199999999999999LL;
    v44 = 80 * v12;
  }
  v47 = a4;
  v58 = v44;
  v45 = sub_22077B0(v44);
  a4 = v47;
  v51 = v45;
  v46 = v45 + v58;
  v14 = v45 + 80;
  v49 = v46;
LABEL_7:
  v15 = *a4;
  v16 = *a3;
  v17 = a3[1];
  v18 = a5[1];
  a5[1] = 0;
  v19 = *a5;
  v20 = a5[2];
  *a5 = 0;
  a5[2] = 0;
  v21 = *a6;
  v22 = a6[1];
  v23 = a6[2];
  a6[1] = 0;
  a6[2] = 0;
  *a6 = 0;
  v24 = (_QWORD *)(v51 + v13);
  if ( v24 )
  {
    *v24 = v19;
    v24[1] = v18;
    v24[2] = v20;
    v24[3] = v21;
    v24[4] = v22;
    v24[5] = v23;
    v24[6] = 0;
    v24[7] = v16;
    v24[8] = v17;
    v24[9] = v15;
  }
  else
  {
    v42 = v23 - v21;
    v43 = v20 - v19;
    if ( v21 )
    {
      v56 = v14;
      j_j___libc_free_0(v21, v42);
      v14 = v56;
    }
    if ( v19 )
    {
      v57 = v14;
      j_j___libc_free_0(v19, v43);
      v14 = v57;
    }
  }
  if ( a2 != (__int64 *)v52 )
  {
    v25 = v51;
    v26 = (__int64 *)v52;
    while ( 1 )
    {
      if ( v25 )
      {
        *(_QWORD *)v25 = *v26;
        *(_QWORD *)(v25 + 8) = v26[1];
        *(_QWORD *)(v25 + 16) = v26[2];
        v27 = v26[3];
        v26[2] = 0;
        v26[1] = 0;
        *v26 = 0;
        *(_QWORD *)(v25 + 24) = v27;
        *(_QWORD *)(v25 + 32) = v26[4];
        *(_QWORD *)(v25 + 40) = v26[5];
        v28 = v26[6];
        v26[5] = 0;
        v26[4] = 0;
        v26[3] = 0;
        *(_QWORD *)(v25 + 48) = v28;
        v29 = _mm_loadu_si128((const __m128i *)(v26 + 7));
        v26[6] = 0;
        *(__m128i *)(v25 + 56) = v29;
        *(_QWORD *)(v25 + 72) = v26[9];
      }
      v30 = (_QWORD *)v26[6];
      v55 = v30;
      if ( v30 )
      {
        v31 = v30 + 9;
        do
        {
          v32 = (_QWORD *)*(v31 - 3);
          v33 = (_QWORD *)*(v31 - 2);
          v31 -= 3;
          v34 = v32;
          if ( v33 != v32 )
          {
            do
            {
              if ( *v34 )
                j_j___libc_free_0(*v34, v34[2] - *v34);
              v34 += 3;
            }
            while ( v33 != v34 );
            v32 = (_QWORD *)*v31;
          }
          if ( v32 )
            j_j___libc_free_0(v32, v31[2] - (_QWORD)v32);
        }
        while ( v55 != v31 );
        j_j___libc_free_0(v55, 72);
      }
      v35 = v26[3];
      if ( v35 )
        j_j___libc_free_0(v35, v26[5] - v35);
      if ( *v26 )
        j_j___libc_free_0(*v26, v26[2] - *v26);
      v26 += 10;
      if ( v26 == a2 )
        break;
      v25 += 80;
    }
    v14 = v25 + 160;
  }
  if ( a2 != (__int64 *)v48 )
  {
    v36 = a2;
    v37 = v14;
    do
    {
      v38 = *v36;
      v39 = _mm_loadu_si128((const __m128i *)(v36 + 7));
      v36 += 10;
      v37 += 80;
      *(_QWORD *)(v37 - 80) = v38;
      v40 = *(v36 - 9);
      *(__m128i *)(v37 - 24) = v39;
      *(_QWORD *)(v37 - 72) = v40;
      *(_QWORD *)(v37 - 64) = *(v36 - 8);
      *(_QWORD *)(v37 - 56) = *(v36 - 7);
      *(_QWORD *)(v37 - 48) = *(v36 - 6);
      *(_QWORD *)(v37 - 40) = *(v36 - 5);
      *(_QWORD *)(v37 - 32) = *(v36 - 4);
      *(_QWORD *)(v37 - 8) = *(v36 - 1);
    }
    while ( v36 != (__int64 *)v48 );
    v14 += 16
         * (5 * ((0xCCCCCCCCCCCCCCDLL * ((unsigned __int64)((char *)v36 - (char *)a2 - 80) >> 4)) & 0xFFFFFFFFFFFFFFFLL)
          + 5);
  }
  if ( v52 )
  {
    v54 = v14;
    j_j___libc_free_0(v52, a1[2] - v52);
    v14 = v54;
  }
  a1[1] = v14;
  *a1 = v51;
  a1[2] = v49;
  return v49;
}
