// Function: sub_1A41720
// Address: 0x1a41720
//
__int64 __fastcall sub_1A41720(
        __int64 a1,
        _QWORD *a2,
        __int64 *a3,
        __m128 a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  unsigned __int64 v12; // r12
  __int64 result; // rax
  __int64 v14; // r14
  __int64 v15; // rax
  __int64 v16; // r8
  int v17; // r9d
  unsigned __int8 *v18; // rsi
  __int64 v19; // rax
  unsigned __int64 *v20; // rax
  __int64 v21; // r8
  int v22; // r9d
  unsigned __int64 v23; // rax
  int v24; // r8d
  int v25; // r9d
  double v26; // xmm4_8
  double v27; // xmm5_8
  _QWORD *v28; // rax
  _BYTE *v29; // rdx
  _QWORD *i; // rdx
  __int64 v31; // rbx
  _QWORD *v32; // rdx
  char v33; // al
  _BYTE *v34; // r12
  __int64 *v35; // rsi
  __int64 *v36; // r14
  __int64 *v37; // rdi
  __int64 v38; // r15
  __int64 v39; // rdx
  __int64 v40; // rax
  __int64 v41; // rdx
  char v42; // al
  int v43; // r8d
  __int64 v44; // rdi
  __int64 v45; // rax
  __int64 v46; // rsi
  __int64 v47; // rdx
  unsigned __int8 *v48; // rsi
  int v50; // [rsp+28h] [rbp-298h]
  __int64 *v51; // [rsp+28h] [rbp-298h]
  __int64 v52; // [rsp+48h] [rbp-278h]
  __int64 v53; // [rsp+50h] [rbp-270h]
  _QWORD v54[2]; // [rsp+60h] [rbp-260h] BYREF
  __m128i v55; // [rsp+70h] [rbp-250h] BYREF
  __int64 v56; // [rsp+80h] [rbp-240h]
  _QWORD *v57; // [rsp+90h] [rbp-230h] BYREF
  __int16 v58; // [rsp+A0h] [rbp-220h]
  __m128 v59; // [rsp+B0h] [rbp-210h] BYREF
  __int64 v60; // [rsp+C0h] [rbp-200h]
  _QWORD v61[2]; // [rsp+D0h] [rbp-1F0h] BYREF
  __int16 v62; // [rsp+E0h] [rbp-1E0h]
  unsigned __int8 *v63; // [rsp+F0h] [rbp-1D0h] BYREF
  __int64 v64; // [rsp+F8h] [rbp-1C8h]
  __int64 *v65; // [rsp+100h] [rbp-1C0h]
  __int64 v66; // [rsp+108h] [rbp-1B8h]
  __int64 v67; // [rsp+110h] [rbp-1B0h]
  int v68; // [rsp+118h] [rbp-1A8h]
  __int64 v69; // [rsp+120h] [rbp-1A0h]
  __int64 v70; // [rsp+128h] [rbp-198h]
  _BYTE *v71; // [rsp+140h] [rbp-180h] BYREF
  __int64 v72; // [rsp+148h] [rbp-178h]
  _BYTE v73[64]; // [rsp+150h] [rbp-170h] BYREF
  __int64 v74[5]; // [rsp+190h] [rbp-130h] BYREF
  char *v75; // [rsp+1B8h] [rbp-108h]
  char v76; // [rsp+1C8h] [rbp-F8h] BYREF
  unsigned __int8 *v77[5]; // [rsp+210h] [rbp-B0h] BYREF
  char *v78; // [rsp+238h] [rbp-88h]
  char v79; // [rsp+248h] [rbp-78h] BYREF

  v12 = (unsigned __int64)a2;
  if ( !*(_DWORD *)(a1 + 496) || (result = sub_1A3F5B0(a1, (__int64)a2), (_BYTE)result) )
  {
    result = 0;
    if ( *(_BYTE *)(*a2 + 8LL) == 16 )
    {
      v14 = *(_QWORD *)(*a2 + 32LL);
      v15 = sub_16498A0((__int64)a2);
      v18 = (unsigned __int8 *)a2[6];
      v63 = 0;
      v66 = v15;
      v19 = *(_QWORD *)(v12 + 40);
      v67 = 0;
      v64 = v19;
      v65 = (__int64 *)(v12 + 24);
      v68 = 0;
      v69 = 0;
      v70 = 0;
      v77[0] = v18;
      if ( v18 )
      {
        sub_1623A60((__int64)v77, (__int64)v18, 2);
        v63 = v77[0];
        if ( v77[0] )
          sub_1623210((__int64)v77, v77[0], (__int64)&v63);
      }
      if ( (*(_BYTE *)(v12 + 23) & 0x40) != 0 )
        v20 = *(unsigned __int64 **)(v12 - 8);
      else
        v20 = (unsigned __int64 *)(v12 - 24LL * (*(_DWORD *)(v12 + 20) & 0xFFFFFFF));
      sub_1A41500((__int64)v74, (_QWORD *)a1, v12, *v20, v16, v17);
      if ( (*(_BYTE *)(v12 + 23) & 0x40) != 0 )
        v23 = *(_QWORD *)(v12 - 8);
      else
        v23 = v12 - 24LL * (*(_DWORD *)(v12 + 20) & 0xFFFFFFF);
      sub_1A41500((__int64)v77, (_QWORD *)a1, v12, *(_QWORD *)(v23 + 24), v21, v22);
      v72 = 0x800000000LL;
      v28 = v73;
      v29 = v73;
      v71 = v73;
      v53 = (unsigned int)v14;
      if ( (_DWORD)v14 )
      {
        if ( (unsigned int)v14 > 8uLL )
        {
          sub_16CD150((__int64)&v71, v73, (unsigned int)v14, 8, v24, v25);
          v29 = v71;
          v28 = &v71[8 * (unsigned int)v72];
        }
        for ( i = &v29[8 * (unsigned int)v14]; i != v28; ++v28 )
        {
          if ( v28 )
            *v28 = 0;
        }
        LODWORD(v72) = v14;
      }
      v31 = 0;
      if ( (_DWORD)v14 )
      {
        v52 = v12;
        do
        {
          LODWORD(v57) = v31;
          v58 = 265;
          v54[0] = sub_1649960(v52);
          v54[1] = v39;
          v55.m128i_i64[0] = (__int64)v54;
          v55.m128i_i64[1] = (__int64)".i";
          v33 = v58;
          LOWORD(v56) = 773;
          if ( (_BYTE)v58 )
          {
            if ( (_BYTE)v58 == 1 )
            {
              a4 = (__m128)_mm_loadu_si128(&v55);
              v59 = a4;
              v60 = v56;
            }
            else
            {
              v32 = v57;
              if ( HIBYTE(v58) != 1 )
              {
                v32 = &v57;
                v33 = 2;
              }
              v59.m128_u64[1] = (unsigned __int64)v32;
              LOBYTE(v60) = 2;
              v59.m128_u64[0] = (unsigned __int64)&v55;
              BYTE1(v60) = v33;
            }
          }
          else
          {
            LOWORD(v60) = 256;
          }
          v34 = sub_1A3F820((__int64 *)v77, v31);
          v35 = (__int64 *)sub_1A3F820(v74, v31);
          v36 = (__int64 *)&v71[8 * v31];
          v37 = (__int64 *)((unsigned int)*(unsigned __int8 *)(*a3 + 16) - 24);
          if ( *((_BYTE *)v35 + 16) > 0x10u
            || v34[16] > 0x10u
            || (v38 = sub_15A2A30(v37, v35, (__int64)v34, 0, 0, *(double *)a4.m128_u64, a5, a6)) == 0 )
          {
            v62 = 257;
            v40 = sub_15FB440((int)v37, v35, (__int64)v34, (__int64)v61, 0);
            v41 = *(_QWORD *)v40;
            v38 = v40;
            v42 = *(_BYTE *)(*(_QWORD *)v40 + 8LL);
            if ( v42 == 16 )
              v42 = *(_BYTE *)(**(_QWORD **)(v41 + 16) + 8LL);
            if ( (unsigned __int8)(v42 - 1) <= 5u || *(_BYTE *)(v38 + 16) == 76 )
            {
              v43 = v68;
              if ( v67 )
              {
                v50 = v68;
                sub_1625C10(v38, 3, v67);
                v43 = v50;
              }
              sub_15F2440(v38, v43);
            }
            if ( v64 )
            {
              v51 = v65;
              sub_157E9D0(v64 + 40, v38);
              v44 = *v51;
              v45 = *(_QWORD *)(v38 + 24) & 7LL;
              *(_QWORD *)(v38 + 32) = v51;
              v44 &= 0xFFFFFFFFFFFFFFF8LL;
              *(_QWORD *)(v38 + 24) = v44 | v45;
              *(_QWORD *)(v44 + 8) = v38 + 24;
              *v51 = *v51 & 7 | (v38 + 24);
            }
            sub_164B780(v38, (__int64 *)&v59);
            if ( v63 )
            {
              v61[0] = v63;
              sub_1623A60((__int64)v61, (__int64)v63, 2);
              v46 = *(_QWORD *)(v38 + 48);
              v47 = v38 + 48;
              if ( v46 )
              {
                sub_161E7C0(v38 + 48, v46);
                v47 = v38 + 48;
              }
              v48 = (unsigned __int8 *)v61[0];
              *(_QWORD *)(v38 + 48) = v61[0];
              if ( v48 )
                sub_1623210((__int64)v61, v48, v47);
            }
          }
          if ( *(_BYTE *)(v38 + 16) > 0x17u )
            sub_15F2530((unsigned __int8 *)v38, *a3, 1);
          *v36 = v38;
          ++v31;
        }
        while ( v31 != v53 );
        v12 = v52;
      }
      sub_1A41120(a1, v12, &v71, a4, a5, a6, a7, v26, v27, a10, a11);
      if ( v71 != v73 )
        _libc_free((unsigned __int64)v71);
      if ( v78 != &v79 )
        _libc_free((unsigned __int64)v78);
      if ( v75 != &v76 )
        _libc_free((unsigned __int64)v75);
      result = 1;
      if ( v63 )
      {
        sub_161E7C0((__int64)&v63, (__int64)v63);
        return 1;
      }
    }
  }
  return result;
}
