// Function: sub_39F8140
// Address: 0x39f8140
//
__int64 __fastcall sub_39F8140(_QWORD *a1, _DWORD a2, __int64 a3, _DWORD a4, _DWORD a5, _DWORD a6, char a7)
{
  __int64 v7; // rax
  __m128i v8; // xmm1
  __m128i v9; // xmm2
  __m128i v10; // xmm3
  __m128i v11; // xmm4
  __m128i v12; // xmm5
  __m128i v13; // xmm6
  __m128i v14; // xmm7
  __m128i v15; // xmm0
  __m128i v16; // xmm1
  __m128i v17; // xmm2
  __m128i v18; // xmm3
  __m128i v19; // xmm4
  __m128i v20; // xmm5
  __m128i v21; // xmm6
  __int64 result; // rax
  int v23; // eax
  _QWORD *v24; // rdx
  __m128i v25; // xmm7
  unsigned __int64 v26; // rdx
  __m128i v27; // xmm0
  __m128i v28; // xmm7
  __m128i v29; // xmm1
  __m128i v30; // xmm2
  __m128i v31; // xmm3
  __m128i v32; // xmm7
  __m128i v33; // xmm4
  __m128i v34; // xmm5
  __m128i v35; // xmm6
  __m128i v36; // xmm7
  __m128i v37; // xmm0
  __m128i v38; // xmm7
  __m128i v39; // xmm7
  __int64 v40; // rsi
  __int64 v41; // rax
  __m128i v42; // [rsp+0h] [rbp-3A0h] BYREF
  __m128i v43; // [rsp+10h] [rbp-390h] BYREF
  __m128i v44; // [rsp+20h] [rbp-380h] BYREF
  __m128i v45; // [rsp+30h] [rbp-370h] BYREF
  __m128i v46; // [rsp+40h] [rbp-360h] BYREF
  __m128i v47; // [rsp+50h] [rbp-350h] BYREF
  __m128i v48; // [rsp+60h] [rbp-340h] BYREF
  __m128i v49; // [rsp+70h] [rbp-330h] BYREF
  __m128i v50; // [rsp+80h] [rbp-320h] BYREF
  __m128i v51; // [rsp+90h] [rbp-310h] BYREF
  __m128i v52; // [rsp+A0h] [rbp-300h] BYREF
  __m128i v53; // [rsp+B0h] [rbp-2F0h] BYREF
  __m128i v54; // [rsp+C0h] [rbp-2E0h] BYREF
  __m128i v55; // [rsp+D0h] [rbp-2D0h] BYREF
  __m128i v56; // [rsp+E0h] [rbp-2C0h] BYREF
  __m128i v57; // [rsp+F0h] [rbp-2B0h] BYREF
  __m128i v58; // [rsp+100h] [rbp-2A0h]
  __m128i v59; // [rsp+110h] [rbp-290h]
  __m128i v60; // [rsp+120h] [rbp-280h]
  __m128i v61; // [rsp+130h] [rbp-270h]
  __m128i v62; // [rsp+140h] [rbp-260h]
  __m128i v63; // [rsp+150h] [rbp-250h]
  __m128i v64; // [rsp+160h] [rbp-240h]
  __m128i v65; // [rsp+170h] [rbp-230h]
  __m128i v66; // [rsp+180h] [rbp-220h]
  __m128i v67; // [rsp+190h] [rbp-210h]
  __m128i v68; // [rsp+1A0h] [rbp-200h]
  __m128i v69; // [rsp+1B0h] [rbp-1F0h]
  __m128i v70; // [rsp+1C0h] [rbp-1E0h]
  __m128i v71; // [rsp+1D0h] [rbp-1D0h]
  __int64 v72[42]; // [rsp+1E0h] [rbp-1C0h] BYREF
  __int64 (__fastcall *v73)(__int64, __int64, _QWORD, _QWORD *, __m128i *); // [rsp+330h] [rbp-70h]
  __int64 v74; // [rsp+348h] [rbp-58h]
  __int64 v75; // [rsp+368h] [rbp-38h]
  __int64 v76; // [rsp+370h] [rbp-30h]
  __int64 retaddr; // [rsp+3A8h] [rbp+8h]

  v76 = a3;
  v75 = v7;
  sub_39F7A80(&v42, (__int64)&a7, retaddr);
  v8 = _mm_loadu_si128(&v43);
  v9 = _mm_loadu_si128(&v44);
  v10 = _mm_loadu_si128(&v45);
  v11 = _mm_loadu_si128(&v46);
  v12 = _mm_loadu_si128(&v47);
  v57 = _mm_loadu_si128(&v42);
  v13 = _mm_loadu_si128(&v48);
  v14 = _mm_loadu_si128(&v49);
  v58 = v8;
  v15 = _mm_loadu_si128(&v50);
  v16 = _mm_loadu_si128(&v51);
  v59 = v9;
  v60 = v10;
  v17 = _mm_loadu_si128(&v52);
  v18 = _mm_loadu_si128(&v53);
  v61 = v11;
  v19 = _mm_loadu_si128(&v54);
  v62 = v12;
  v20 = _mm_loadu_si128(&v55);
  v63 = v13;
  v21 = _mm_loadu_si128(&v56);
  v64 = v14;
  v65 = v15;
  v66 = v16;
  v67 = v17;
  v68 = v18;
  v69 = v19;
  v70 = v20;
  v71 = v21;
  while ( 1 )
  {
    result = sub_39F7420(&v57, (char *)v72);
    if ( (_DWORD)result == 5 )
      return result;
    while ( 1 )
    {
      if ( (_DWORD)result )
        return 3;
      if ( v73 )
      {
        v23 = v73(1, 1, *a1, a1, &v57);
        if ( v23 == 6 )
        {
          v25 = _mm_loadu_si128(&v42);
          v26 = v69.m128i_i64[0];
          a1[2] = 0;
          v27 = _mm_loadu_si128(&v47);
          v57 = v25;
          v28 = _mm_loadu_si128(&v43);
          v29 = _mm_loadu_si128(&v48);
          v30 = _mm_loadu_si128(&v49);
          v31 = _mm_loadu_si128(&v50);
          v62 = v27;
          v58 = v28;
          v32 = _mm_loadu_si128(&v44);
          v33 = _mm_loadu_si128(&v51);
          v34 = _mm_loadu_si128(&v52);
          v35 = _mm_loadu_si128(&v53);
          a1[3] = v66.m128i_i64[0] - (v26 >> 63);
          v59 = v32;
          v36 = _mm_loadu_si128(&v45);
          v37 = _mm_loadu_si128(&v55);
          v63 = v29;
          v60 = v36;
          v38 = _mm_loadu_si128(&v46);
          v64 = v30;
          v61 = v38;
          v39 = _mm_loadu_si128(&v54);
          v65 = v31;
          v66 = v33;
          v67 = v34;
          v68 = v35;
          v69 = v39;
          v70 = v37;
          v71 = _mm_loadu_si128(&v56);
          result = sub_39F7C00(a1, &v57, v72);
          if ( (_DWORD)result == 7 )
          {
            sub_39F5CF0((__int64)&v42, (__int64)&v57);
            v40 = v66.m128i_i64[1];
            nullsub_2004();
            *(__int64 *)((char *)&retaddr + v41) = v40;
            return v75;
          }
          return result;
        }
        if ( v23 != 8 )
          return 3;
      }
      sub_39F6770(&v57, (__int64)v72);
      if ( LODWORD(v72[2 * v74 + 1]) != 6 )
        break;
      v66.m128i_i64[1] = 0;
      result = sub_39F7420(&v57, (char *)v72);
      if ( (_DWORD)result == 5 )
        return result;
    }
    if ( (int)v74 > 17 )
      goto LABEL_19;
    v24 = (_QWORD *)v57.m128i_i64[(int)v74];
    if ( (v69.m128i_i8[7] & 0x40) == 0 || !v70.m128i_i8[(int)v74 + 8] )
    {
      if ( byte_5057700[(int)v74] == 8 )
      {
        v24 = (_QWORD *)*v24;
        goto LABEL_15;
      }
LABEL_19:
      abort();
    }
LABEL_15:
    v66.m128i_i64[1] = (__int64)v24;
  }
}
