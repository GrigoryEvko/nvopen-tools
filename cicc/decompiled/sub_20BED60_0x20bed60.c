// Function: sub_20BED60
// Address: 0x20bed60
//
__int64 __fastcall sub_20BED60(
        __m128i *a1,
        __int64 *a2,
        char a3,
        double a4,
        double a5,
        __m128i a6,
        __int64 a7,
        __int64 a8,
        __m128i *a9,
        unsigned int *a10,
        __int64 a11)
{
  unsigned int v15; // esi
  _BOOL4 v16; // r10d
  _BOOL4 v17; // ecx
  int v18; // r10d
  int v19; // ecx
  _BOOL4 v20; // r10d
  _BOOL4 v21; // ecx
  _BOOL4 v22; // r10d
  _BOOL4 v23; // ecx
  _BOOL4 v24; // r10d
  _BOOL4 v25; // ecx
  _BOOL4 v26; // r10d
  _BOOL4 v27; // ecx
  _BOOL4 v28; // r10d
  _BOOL4 v29; // ecx
  _BOOL4 v30; // r10d
  _BOOL4 v31; // ecx
  _BOOL4 v32; // r10d
  _BOOL4 v33; // ecx
  _BOOL4 v34; // r10d
  _BOOL4 v35; // ecx
  _BOOL4 v36; // r10d
  _BOOL4 v37; // ecx
  int v38; // eax
  unsigned int v39; // r15d
  __m128 v40; // xmm0
  __m128i v41; // xmm1
  __int64 v42; // rcx
  __int64 v43; // r8
  __int64 v44; // r9
  __int32 v45; // edx
  int v46; // edi
  __int64 result; // rax
  __int64 v48; // rdx
  __int64 v49; // rdx
  __int64 v50; // rax
  unsigned __int64 v51; // rax
  const void **v52; // rdx
  unsigned __int64 v53; // rdx
  __int64 v54; // rdx
  __int64 v55; // rcx
  __int64 v56; // r8
  __int64 v57; // r9
  __int64 v58; // rdx
  __int64 v59; // rax
  unsigned __int64 v60; // rax
  const void **v61; // rdx
  int v62; // edx
  const void ***v63; // rax
  unsigned int v64; // edx
  _BOOL4 v65; // r10d
  _BOOL4 v66; // ecx
  int v67; // eax
  _BOOL4 v68; // r10d
  _BOOL4 v69; // ecx
  _BOOL4 v70; // r10d
  _BOOL4 v71; // ecx
  _BOOL4 v72; // r10d
  _BOOL4 v73; // ecx
  __int64 v74; // [rsp-10h] [rbp-100h]
  __int64 v75; // [rsp-8h] [rbp-F8h]
  char v76; // [rsp+10h] [rbp-E0h]
  __int64 (__fastcall *v77)(__m128i *, __int64, __int64, _QWORD, _QWORD); // [rsp+10h] [rbp-E0h]
  __int64 *v78; // [rsp+10h] [rbp-E0h]
  unsigned __int64 v79; // [rsp+18h] [rbp-D8h]
  int v80; // [rsp+28h] [rbp-C8h]
  __int64 v81; // [rsp+28h] [rbp-C8h]
  __int64 (__fastcall *v82)(__m128i *, __int64, __int64, _QWORD, _QWORD); // [rsp+28h] [rbp-C8h]
  int v83; // [rsp+30h] [rbp-C0h]
  __int64 v84; // [rsp+30h] [rbp-C0h]
  __int64 v85; // [rsp+30h] [rbp-C0h]
  __int64 v86; // [rsp+38h] [rbp-B8h]
  int v87; // [rsp+40h] [rbp-B0h]
  __int64 v88; // [rsp+40h] [rbp-B0h]
  __int64 v89; // [rsp+48h] [rbp-A8h]
  _OWORD v90[2]; // [rsp+80h] [rbp-70h] BYREF
  __int64 v91; // [rsp+A0h] [rbp-50h] BYREF
  int v92; // [rsp+A8h] [rbp-48h]

  v15 = *a10;
  switch ( *a10 )
  {
    case 1u:
    case 0x11u:
      if ( a3 == 9 )
      {
        v76 = 0;
        v18 = 294;
        v19 = 294;
        v87 = 462;
      }
      else
      {
        v76 = 0;
        if ( a3 == 10 )
        {
          v18 = 295;
          v19 = 295;
          v87 = 462;
        }
        else
        {
          v87 = 462;
          v20 = a3 != 12;
          v21 = v20;
          v18 = v20 + 296;
          v19 = v21 + 296;
        }
      }
      break;
    case 2u:
    case 0x12u:
      if ( a3 == 9 )
      {
        v76 = 0;
        v18 = 314;
        v19 = 314;
        v87 = 462;
      }
      else
      {
        v76 = 0;
        if ( a3 == 10 )
        {
          v18 = 315;
          v19 = 315;
          v87 = 462;
        }
        else
        {
          v87 = 462;
          v22 = a3 != 12;
          v23 = v22;
          v18 = v22 + 316;
          v19 = v23 + 316;
        }
      }
      break;
    case 3u:
    case 0x13u:
      if ( a3 == 9 )
      {
        v76 = 0;
        v18 = 302;
        v19 = 302;
        v87 = 462;
      }
      else
      {
        v76 = 0;
        if ( a3 == 10 )
        {
          v18 = 303;
          v19 = 303;
          v87 = 462;
        }
        else
        {
          v87 = 462;
          v24 = a3 != 12;
          v25 = v24;
          v18 = v24 + 304;
          v19 = v25 + 304;
        }
      }
      break;
    case 4u:
    case 0x14u:
      if ( a3 == 9 )
      {
        v76 = 0;
        v18 = 306;
        v19 = 306;
        v87 = 462;
      }
      else
      {
        v76 = 0;
        if ( a3 == 10 )
        {
          v18 = 307;
          v19 = 307;
          v87 = 462;
        }
        else
        {
          v87 = 462;
          v26 = a3 != 12;
          v27 = v26;
          v18 = v26 + 308;
          v19 = v27 + 308;
        }
      }
      break;
    case 5u:
    case 0x15u:
      if ( a3 == 9 )
      {
        v76 = 0;
        v18 = 310;
        v19 = 310;
        v87 = 462;
      }
      else
      {
        v76 = 0;
        if ( a3 == 10 )
        {
          v18 = 311;
          v19 = 311;
          v87 = 462;
        }
        else
        {
          v87 = 462;
          v28 = a3 != 12;
          v29 = v28;
          v18 = v28 + 312;
          v19 = v29 + 312;
        }
      }
      break;
    case 6u:
      if ( a3 == 9 )
      {
        v76 = 0;
        v18 = 306;
        v19 = 306;
        v87 = 314;
      }
      else
      {
        v76 = 0;
        if ( a3 == 10 )
        {
          v18 = 307;
          v19 = 307;
          v87 = 315;
        }
        else
        {
          v36 = a3 != 12;
          v37 = v36;
          v38 = v36 + 316;
          v18 = v36 + 308;
          v19 = v37 + 308;
          v87 = v38;
        }
      }
      break;
    case 7u:
      if ( a3 == 9 )
      {
        v76 = 0;
        v18 = 322;
        v19 = 322;
        v87 = 462;
      }
      else
      {
        v76 = 0;
        if ( a3 == 10 )
        {
          v18 = 323;
          v19 = 323;
          v87 = 462;
        }
        else
        {
          v87 = 462;
          v32 = a3 != 12;
          v33 = v32;
          v18 = v32 + 324;
          v19 = v33 + 324;
        }
      }
      break;
    case 8u:
      if ( a3 == 9 )
      {
        v76 = 0;
        v18 = 318;
        v19 = 318;
        v87 = 462;
      }
      else
      {
        v76 = 0;
        if ( a3 == 10 )
        {
          v18 = 319;
          v19 = 319;
          v87 = 462;
        }
        else
        {
          v87 = 462;
          v34 = a3 != 12;
          v35 = v34;
          v18 = v34 + 320;
          v19 = v35 + 320;
        }
      }
      break;
    case 9u:
      if ( a3 == 9 )
      {
        v76 = 0;
        v18 = 318;
        v19 = 318;
        v87 = 294;
      }
      else
      {
        v76 = 0;
        if ( a3 == 10 )
        {
          v18 = 319;
          v19 = 319;
          v87 = 295;
        }
        else
        {
          v65 = a3 != 12;
          v66 = v65;
          v67 = v65 + 296;
          v18 = v65 + 320;
          v19 = v66 + 320;
          v87 = v67;
        }
      }
      break;
    case 0xEu:
    case 0x16u:
      if ( a3 == 9 )
      {
        v76 = 0;
        v18 = 298;
        v19 = 298;
        v87 = 462;
      }
      else
      {
        v76 = 0;
        if ( a3 == 10 )
        {
          v18 = 299;
          v19 = 299;
          v87 = 462;
        }
        else
        {
          v87 = 462;
          v30 = a3 != 12;
          v31 = v30;
          v18 = v30 + 300;
          v19 = v31 + 300;
        }
      }
      break;
    default:
      if ( v15 == 12 )
      {
        if ( a3 == 9 )
        {
          v76 = 1;
          v18 = 302;
          v19 = 302;
          v87 = 462;
        }
        else
        {
          v76 = 1;
          if ( a3 == 10 )
          {
            v18 = 303;
            v19 = 303;
            v87 = 462;
          }
          else
          {
            v87 = 462;
            v72 = a3 != 12;
            v73 = v72;
            v18 = v72 + 304;
            v19 = v73 + 304;
          }
        }
      }
      else if ( v15 > 0xC )
      {
        if ( a3 == 9 )
        {
          v76 = 1;
          v18 = 314;
          v19 = 314;
          v87 = 462;
        }
        else
        {
          v76 = 1;
          if ( a3 == 10 )
          {
            v18 = 315;
            v19 = 315;
            v87 = 462;
          }
          else
          {
            v87 = 462;
            v70 = a3 != 12;
            v71 = v70;
            v18 = v70 + 316;
            v19 = v71 + 316;
          }
        }
      }
      else if ( v15 == 10 )
      {
        if ( a3 == 9 )
        {
          v76 = 1;
          v18 = 310;
          v19 = 310;
          v87 = 462;
        }
        else
        {
          v76 = 1;
          if ( a3 == 10 )
          {
            v18 = 311;
            v19 = 311;
            v87 = 462;
          }
          else
          {
            v87 = 462;
            v68 = a3 != 12;
            v69 = v68;
            v18 = v68 + 312;
            v19 = v69 + 312;
          }
        }
      }
      else if ( a3 == 9 )
      {
        v76 = 1;
        v18 = 306;
        v19 = 306;
        v87 = 462;
      }
      else
      {
        v76 = 1;
        if ( a3 == 10 )
        {
          v18 = 307;
          v19 = 307;
          v87 = 462;
        }
        else
        {
          v87 = 462;
          v16 = a3 != 12;
          v17 = v16;
          v18 = v16 + 308;
          v19 = v17 + 308;
        }
      }
      break;
  }
  v83 = v19;
  v80 = v18;
  v39 = (*(unsigned __int8 (__fastcall **)(__m128i *))(a1->m128i_i64[0] + 272))(a1);
  v40 = (__m128)_mm_loadu_si128((const __m128i *)a8);
  v41 = _mm_loadu_si128(a9);
  v90[0] = v40;
  v90[1] = v41;
  sub_20BE530((__int64)&v91, a1, (__int64)a2, v83, v39, 0, (__m128i)v40, v41, a6, (__int64)v90, 2u, 0, a11, 0, 1);
  *(_QWORD *)a8 = v91;
  *(_DWORD *)(a8 + 8) = v92;
  a9->m128i_i64[0] = sub_1D38BB0((__int64)a2, 0, a11, v39, 0, 0, (__m128i)v40, *(double *)v41.m128i_i64, a6, 0);
  a9->m128i_i32[2] = v45;
  v46 = a1[4862].m128i_i32[v80 + 2];
  *a10 = v46;
  result = v74;
  v48 = v75;
  if ( v76 )
  {
    result = sub_1D16EF0(v46, 1);
    *a10 = result;
  }
  if ( v87 != 462 )
  {
    v84 = sub_1D28D50(a2, *a10, v48, v42, v43, v44);
    v86 = v49;
    v81 = a2[6];
    v77 = *(__int64 (__fastcall **)(__m128i *, __int64, __int64, _QWORD, _QWORD))(a1->m128i_i64[0] + 264);
    v50 = sub_1E0A0C0(a2[4]);
    v51 = v77(a1, v50, v81, v39, 0);
    v78 = sub_1D3A900(
            a2,
            0x89u,
            a11,
            v51,
            v52,
            0,
            v40,
            *(double *)v41.m128i_i64,
            a6,
            *(_QWORD *)a8,
            *(__int16 **)(a8 + 8),
            (__int128)*a9,
            v84,
            v86);
    v79 = v53;
    sub_20BE530((__int64)&v91, a1, (__int64)a2, v87, v39, 0, (__m128i)v40, v41, a6, (__int64)v90, 2u, 0, a11, 0, 1);
    *(_QWORD *)a8 = v91;
    *(_DWORD *)(a8 + 8) = v92;
    v88 = sub_1D28D50(a2, a1[4862].m128i_u32[v87 + 2], v54, v55, v56, v57);
    v89 = v58;
    v85 = a2[6];
    v82 = *(__int64 (__fastcall **)(__m128i *, __int64, __int64, _QWORD, _QWORD))(a1->m128i_i64[0] + 264);
    v59 = sub_1E0A0C0(a2[4]);
    v60 = v82(a1, v59, v85, v39, 0);
    *(_QWORD *)a8 = sub_1D3A900(
                      a2,
                      0x89u,
                      a11,
                      v60,
                      v61,
                      0,
                      v40,
                      *(double *)v41.m128i_i64,
                      a6,
                      *(_QWORD *)a8,
                      *(__int16 **)(a8 + 8),
                      (__int128)*a9,
                      v88,
                      v89);
    *(_DWORD *)(a8 + 8) = v62;
    v63 = (const void ***)(v78[5] + 16LL * (unsigned int)v79);
    *(_QWORD *)a8 = sub_1D332F0(
                      a2,
                      119,
                      a11,
                      *(unsigned __int8 *)v63,
                      v63[1],
                      0,
                      *(double *)v40.m128_u64,
                      *(double *)v41.m128i_i64,
                      a6,
                      (__int64)v78,
                      v79,
                      *(_OWORD *)a8);
    result = v64;
    *(_DWORD *)(a8 + 8) = v64;
    a9->m128i_i64[0] = 0;
    a9->m128i_i32[2] = 0;
  }
  return result;
}
