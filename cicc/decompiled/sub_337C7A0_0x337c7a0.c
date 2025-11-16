// Function: sub_337C7A0
// Address: 0x337c7a0
//
__int64 __fastcall sub_337C7A0(__int64 a1, char a2, char a3, int a4, int a5, _QWORD *a6, unsigned __int64 *a7)
{
  _QWORD *v9; // rbx
  unsigned __int64 *v10; // r12
  int v11; // eax
  unsigned int v12; // esi
  __int64 v13; // rax
  __m128i *v14; // rsi
  __int32 v15; // edx
  __m128i v16; // rax
  unsigned int v17; // r15d
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // r13
  unsigned __int16 v21; // di
  const __m128i *v22; // rdx
  __int64 (__fastcall *v23)(__int64, __int64, __int64, unsigned __int64); // rax
  __m128i v24; // xmm0
  int v25; // eax
  _QWORD *v26; // r12
  int v27; // ebx
  __m128i *v28; // rsi
  __int64 v29; // rax
  __m128i v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rax
  __int64 (__fastcall *v33)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v34; // rax
  __int64 v35; // rcx
  __int64 (__fastcall *v36)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v37; // rax
  __int64 v38; // rcx
  __int64 (__fastcall *v39)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v40; // rax
  unsigned __int64 v41; // rcx
  unsigned __int16 v42; // dx
  __int64 v43; // rax
  __int8 v44; // cl
  __int64 v45; // rax
  unsigned int v46; // eax
  int v47; // eax
  __int64 v48; // r15
  __int64 v49; // r13
  __m128i *v50; // rsi
  __int64 v51; // rdx
  __int64 v52; // rdx
  unsigned __int64 v53; // rdx
  __int64 v55; // [rsp+8h] [rbp-128h]
  __int64 v56; // [rsp+10h] [rbp-120h]
  __int64 v57; // [rsp+18h] [rbp-118h]
  __int64 v58; // [rsp+20h] [rbp-110h]
  __int64 v59; // [rsp+28h] [rbp-108h]
  unsigned __int64 v60; // [rsp+28h] [rbp-108h]
  unsigned __int64 v61; // [rsp+28h] [rbp-108h]
  int v62; // [rsp+38h] [rbp-F8h]
  __int64 v63; // [rsp+48h] [rbp-E8h]
  __int64 v64; // [rsp+50h] [rbp-E0h]
  __int64 v65; // [rsp+60h] [rbp-D0h]
  unsigned __int16 v66; // [rsp+76h] [rbp-BAh] BYREF
  unsigned int v67; // [rsp+78h] [rbp-B8h] BYREF
  unsigned int v68; // [rsp+7Ch] [rbp-B4h]
  __m128i v69; // [rsp+80h] [rbp-B0h] BYREF
  __m128i v70; // [rsp+90h] [rbp-A0h] BYREF
  __m128i v71; // [rsp+A0h] [rbp-90h] BYREF
  __int64 v72; // [rsp+B0h] [rbp-80h]
  __int64 v73; // [rsp+B8h] [rbp-78h]
  __int64 v74; // [rsp+C0h] [rbp-70h] BYREF
  __int64 v75; // [rsp+C8h] [rbp-68h]
  __int64 v76; // [rsp+D0h] [rbp-60h] BYREF
  __int64 v77; // [rsp+D8h] [rbp-58h]
  __m128i v78; // [rsp+E0h] [rbp-50h] BYREF
  unsigned __int64 v79; // [rsp+F0h] [rbp-40h]
  unsigned __int64 *v80; // [rsp+140h] [rbp+10h]

  v9 = (_QWORD *)a1;
  v10 = a7;
  v64 = a6[2];
  v11 = *(_DWORD *)(a1 + 120);
  v12 = (8 * v11) | a2 & 7;
  if ( a3 )
  {
    v12 = (a4 << 16) | v12 & 0x8000FFFF | 0x80000000;
  }
  else if ( v11 )
  {
    v47 = **(_DWORD **)(a1 + 112);
    if ( v47 < 0 )
      v12 = ((*(unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a6[5] + 32LL) + 56LL)
                                                          + 16LL * (v47 & 0x7FFFFFFF))
                                              & 0xFFFFFFFFFFFFFFF8LL)
                                  + 24LL)
            + 1) << 16)
          | v12 & 0xC000FFFF;
  }
  v13 = sub_3400BD0((_DWORD)a6, v12, a5, 7, 0, 1, 0);
  v14 = (__m128i *)a7[1];
  v69.m128i_i64[0] = v13;
  v69.m128i_i32[2] = v15;
  if ( v14 == (__m128i *)a7[2] )
  {
    sub_33764F0(a7, v14, &v69);
  }
  else
  {
    if ( v14 )
    {
      *v14 = _mm_loadu_si128(&v69);
      v14 = (__m128i *)a7[1];
    }
    a7[1] = (unsigned __int64)&v14[1];
  }
  v16.m128i_i64[0] = *(unsigned int *)(a1 + 8);
  if ( a2 == 4 )
  {
    v48 = 0;
    v49 = 2LL * v16.m128i_u32[0];
    if ( v16.m128i_i32[0] )
    {
      do
      {
        v16.m128i_i64[0] = sub_33F0B60(
                             a6,
                             *(unsigned int *)(*(_QWORD *)(a1 + 112) + 2 * v48),
                             *(unsigned __int16 *)(*(_QWORD *)(a1 + 80) + v48),
                             0);
        v50 = (__m128i *)a7[1];
        v78 = v16;
        if ( v50 == (__m128i *)a7[2] )
        {
          v16.m128i_i64[0] = sub_337C620(a7, v50, &v78);
        }
        else
        {
          if ( v50 )
          {
            *v50 = v16;
            v50 = (__m128i *)a7[1];
          }
          a7[1] = (unsigned __int64)&v50[1];
        }
        v48 += 2;
      }
      while ( v49 != v48 );
    }
  }
  else if ( v16.m128i_i32[0] )
  {
    v17 = 0;
    v65 = 0;
    v63 = 2 * v16.m128i_i64[0];
    do
    {
      v18 = v9[10];
      v19 = *v9;
      BYTE2(v68) = 1;
      v20 = a6[8];
      v21 = *(_WORD *)(v18 + v65);
      v22 = (const __m128i *)(v19 + 8 * v65);
      v23 = *(__int64 (__fastcall **)(__int64, __int64, __int64, unsigned __int64))(*(_QWORD *)v64 + 736LL);
      LOWORD(v68) = v21;
      if ( v23 == sub_2FEA1A0 )
      {
        v24 = _mm_loadu_si128(v22);
        v70 = v24;
        if ( v24.m128i_i16[0] )
        {
          v25 = *(unsigned __int16 *)(v64 + 2LL * v24.m128i_u16[0] + 2304);
        }
        else
        {
          if ( !sub_30070B0((__int64)&v70) )
          {
            if ( !sub_3007070((__int64)&v70) )
              goto LABEL_65;
            v72 = sub_3007260((__int64)&v70);
            v73 = v31;
            v78.m128i_i64[0] = v72;
            v78.m128i_i8[8] = v31;
            v62 = sub_CA1930(&v78);
            v32 = v70.m128i_u16[0];
            v71 = v70;
            if ( v70.m128i_i16[0] )
              goto LABEL_43;
            v58 = v70.m128i_i64[1];
            v59 = v70.m128i_i64[0];
            if ( sub_30070B0((__int64)&v71) )
            {
              LOWORD(v74) = 0;
              v78.m128i_i16[0] = 0;
              v78.m128i_i64[1] = 0;
              sub_2FE8D10(
                v64,
                v20,
                v71.m128i_u32[0],
                v71.m128i_u64[1],
                v78.m128i_i64,
                (unsigned int *)&v76,
                (unsigned __int16 *)&v74);
              v42 = v74;
            }
            else
            {
              if ( !sub_3007070((__int64)&v71) )
                goto LABEL_65;
              v33 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v64 + 592LL);
              if ( v33 == sub_2D56A50 )
              {
                sub_2FE6CC0((__int64)&v78, v64, v20, v59, v58);
                v34 = v57;
                LOWORD(v34) = v78.m128i_i16[4];
                v35 = v79;
                v57 = v34;
              }
              else
              {
                v57 = v33(v64, v20, v71.m128i_u32[0], v71.m128i_i64[1]);
                v35 = v51;
              }
              v75 = v35;
              v32 = (unsigned __int16)v57;
              v74 = v57;
              if ( (_WORD)v57 )
                goto LABEL_43;
              v60 = v35;
              if ( sub_30070B0((__int64)&v74) )
              {
                v78.m128i_i16[0] = 0;
                LOWORD(v67) = 0;
                v78.m128i_i64[1] = 0;
                sub_2FE8D10(
                  v64,
                  v20,
                  (unsigned int)v74,
                  v60,
                  v78.m128i_i64,
                  (unsigned int *)&v76,
                  (unsigned __int16 *)&v67);
                v42 = v67;
              }
              else
              {
                if ( !sub_3007070((__int64)&v74) )
                  goto LABEL_65;
                v36 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v64 + 592LL);
                if ( v36 == sub_2D56A50 )
                {
                  sub_2FE6CC0((__int64)&v78, v64, v20, v74, v75);
                  v37 = v56;
                  LOWORD(v37) = v78.m128i_i16[4];
                  v38 = v79;
                  v56 = v37;
                }
                else
                {
                  v56 = v36(v64, v20, v74, v60);
                  v38 = v52;
                }
                v77 = v38;
                v32 = (unsigned __int16)v56;
                v76 = v56;
                if ( !(_WORD)v56 )
                {
                  v61 = v38;
                  if ( sub_30070B0((__int64)&v76) )
                  {
                    v78.m128i_i16[0] = 0;
                    v66 = 0;
                    v78.m128i_i64[1] = 0;
                    sub_2FE8D10(v64, v20, (unsigned int)v76, v61, v78.m128i_i64, &v67, &v66);
                    v42 = v66;
                  }
                  else
                  {
                    if ( !sub_3007070((__int64)&v76) )
LABEL_65:
                      BUG();
                    v39 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v64 + 592LL);
                    if ( v39 == sub_2D56A50 )
                    {
                      sub_2FE6CC0((__int64)&v78, v64, v20, v76, v77);
                      v40 = v55;
                      LOWORD(v40) = v78.m128i_i16[4];
                      v41 = v79;
                      v55 = v40;
                    }
                    else
                    {
                      v55 = v39(v64, v20, v76, v61);
                      v41 = v53;
                    }
                    v42 = sub_2FE98B0(v64, v20, (unsigned int)v55, v41);
                  }
                  goto LABEL_44;
                }
LABEL_43:
                v42 = *(_WORD *)(v64 + 2 * v32 + 2852);
              }
            }
LABEL_44:
            if ( v42 <= 1u || (unsigned __int16)(v42 - 504) <= 7u )
              BUG();
            v43 = 16LL * (v42 - 1);
            v44 = byte_444C4A0[v43 + 8];
            v45 = *(_QWORD *)&byte_444C4A0[v43];
            v78.m128i_i8[8] = v44;
            v78.m128i_i64[0] = v45;
            v46 = sub_CA1930(&v78);
            v25 = (v62 + v46 - 1) / v46;
            goto LABEL_13;
          }
          LOWORD(v74) = 0;
          v78.m128i_i16[0] = 0;
          v78.m128i_i64[1] = 0;
          v25 = sub_2FE8D10(
                  v64,
                  v20,
                  v70.m128i_u32[0],
                  v70.m128i_u64[1],
                  v78.m128i_i64,
                  (unsigned int *)&v76,
                  (unsigned __int16 *)&v74);
        }
      }
      else
      {
        v25 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD, __int64, _QWORD))v23)(
                v64,
                v20,
                v22->m128i_u32[0],
                v22->m128i_i64[1],
                v68);
      }
LABEL_13:
      if ( v25 )
      {
        v80 = v10;
        v26 = v9;
        v27 = v25 + v17;
        do
        {
          while ( 1 )
          {
            v29 = v17++;
            v30.m128i_i64[0] = sub_33F0B60(a6, *(unsigned int *)(v26[14] + 4 * v29), v21, 0);
            v28 = (__m128i *)v80[1];
            v78 = v30;
            if ( v28 != (__m128i *)v80[2] )
              break;
            sub_337C620(v80, v28, &v78);
            if ( v17 == v27 )
              goto LABEL_20;
          }
          if ( v28 )
          {
            *v28 = v30;
            v28 = (__m128i *)v80[1];
          }
          v80[1] = (unsigned __int64)&v28[1];
        }
        while ( v17 != v27 );
LABEL_20:
        v9 = v26;
        v10 = v80;
      }
      v65 += 2;
      v16.m128i_i64[0] = v65;
    }
    while ( v63 != v65 );
  }
  return v16.m128i_i64[0];
}
