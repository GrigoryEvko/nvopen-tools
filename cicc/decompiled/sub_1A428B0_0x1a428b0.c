// Function: sub_1A428B0
// Address: 0x1a428b0
//
__int64 __fastcall sub_1A428B0(
        __int64 a1,
        unsigned __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 result; // rax
  __int64 v12; // r13
  __int64 v13; // rax
  __int64 v14; // r8
  int v15; // r9d
  unsigned __int8 *v16; // rsi
  __int64 v17; // rax
  int v18; // r8d
  int v19; // r9d
  double v20; // xmm4_8
  double v21; // xmm5_8
  _BYTE *v22; // rdx
  _QWORD *v23; // rax
  _QWORD *i; // rdx
  __int64 v25; // rbx
  _QWORD *v26; // rdx
  char v27; // al
  _BYTE *v28; // rax
  _QWORD *v29; // r14
  int v30; // edi
  _QWORD *v31; // r13
  __int64 v32; // rdx
  __int64 v33; // rax
  unsigned __int64 v34; // rsi
  __int64 v35; // rax
  __int64 v36; // rsi
  __int64 v37; // rdx
  unsigned __int8 *v38; // rsi
  __int64 v40; // [rsp+38h] [rbp-208h]
  __int64 v41; // [rsp+40h] [rbp-200h]
  __int64 **v42; // [rsp+48h] [rbp-1F8h]
  unsigned __int64 *v43; // [rsp+48h] [rbp-1F8h]
  unsigned __int8 *v44; // [rsp+58h] [rbp-1E8h] BYREF
  _QWORD v45[2]; // [rsp+60h] [rbp-1E0h] BYREF
  __m128i v46; // [rsp+70h] [rbp-1D0h] BYREF
  __int64 v47; // [rsp+80h] [rbp-1C0h]
  _QWORD *v48; // [rsp+90h] [rbp-1B0h] BYREF
  __int16 v49; // [rsp+A0h] [rbp-1A0h]
  __m128 v50; // [rsp+B0h] [rbp-190h] BYREF
  __int64 v51; // [rsp+C0h] [rbp-180h]
  char v52[16]; // [rsp+D0h] [rbp-170h] BYREF
  __int16 v53; // [rsp+E0h] [rbp-160h]
  unsigned __int8 *v54; // [rsp+F0h] [rbp-150h] BYREF
  __int64 v55; // [rsp+F8h] [rbp-148h]
  unsigned __int64 *v56; // [rsp+100h] [rbp-140h]
  __int64 v57; // [rsp+108h] [rbp-138h]
  __int64 v58; // [rsp+110h] [rbp-130h]
  int v59; // [rsp+118h] [rbp-128h]
  __int64 v60; // [rsp+120h] [rbp-120h]
  __int64 v61; // [rsp+128h] [rbp-118h]
  _BYTE *v62; // [rsp+140h] [rbp-100h] BYREF
  __int64 v63; // [rsp+148h] [rbp-F8h]
  _BYTE v64[64]; // [rsp+150h] [rbp-F0h] BYREF
  unsigned __int8 *v65[5]; // [rsp+190h] [rbp-B0h] BYREF
  char *v66; // [rsp+1B8h] [rbp-88h]
  char v67; // [rsp+1C8h] [rbp-78h] BYREF

  if ( !*(_DWORD *)(a1 + 496) || (result = sub_1A3F5B0(a1, a2), (_BYTE)result) )
  {
    v40 = *(_QWORD *)a2;
    result = 0;
    if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 16 )
    {
      v12 = *(_QWORD *)(*(_QWORD *)a2 + 32LL);
      v13 = sub_16498A0(a2);
      v16 = *(unsigned __int8 **)(a2 + 48);
      v54 = 0;
      v57 = v13;
      v17 = *(_QWORD *)(a2 + 40);
      v58 = 0;
      v55 = v17;
      v59 = 0;
      v60 = 0;
      v61 = 0;
      v56 = (unsigned __int64 *)(a2 + 24);
      v65[0] = v16;
      if ( v16 )
      {
        sub_1623A60((__int64)v65, (__int64)v16, 2);
        v54 = v65[0];
        if ( v65[0] )
          sub_1623210((__int64)v65, v65[0], (__int64)&v54);
      }
      sub_1A41500((__int64)v65, (_QWORD *)a1, a2, *(_QWORD *)(a2 - 24), v14, v15);
      v62 = v64;
      v63 = 0x800000000LL;
      v41 = (unsigned int)v12;
      if ( (_DWORD)v12 )
      {
        v22 = v64;
        v23 = v64;
        if ( (unsigned int)v12 > 8uLL )
        {
          sub_16CD150((__int64)&v62, v64, (unsigned int)v12, 8, v18, v19);
          v22 = v62;
          v23 = &v62[8 * (unsigned int)v63];
        }
        for ( i = &v22[8 * (unsigned int)v12]; i != v23; ++v23 )
        {
          if ( v23 )
            *v23 = 0;
        }
        LODWORD(v63) = v12;
      }
      v25 = 0;
      if ( (_DWORD)v12 )
      {
        do
        {
          LODWORD(v48) = v25;
          v49 = 265;
          v45[0] = sub_1649960(a2);
          v45[1] = v32;
          v46.m128i_i64[0] = (__int64)v45;
          v46.m128i_i64[1] = (__int64)".i";
          v27 = v49;
          LOWORD(v47) = 773;
          if ( (_BYTE)v49 )
          {
            if ( (_BYTE)v49 == 1 )
            {
              a3 = (__m128)_mm_loadu_si128(&v46);
              v50 = a3;
              v51 = v47;
            }
            else
            {
              v26 = v48;
              if ( HIBYTE(v49) != 1 )
              {
                v26 = &v48;
                v27 = 2;
              }
              v50.m128_u64[1] = (unsigned __int64)v26;
              LOBYTE(v51) = 2;
              v50.m128_u64[0] = (unsigned __int64)&v46;
              BYTE1(v51) = v27;
            }
          }
          else
          {
            LOWORD(v51) = 256;
          }
          v42 = *(__int64 ***)(v40 + 24);
          v28 = sub_1A3F820((__int64 *)v65, v25);
          v29 = v28;
          v30 = *(unsigned __int8 *)(a2 + 16) - 24;
          v31 = &v62[8 * v25];
          if ( v42 != *(__int64 ***)v28 )
          {
            if ( v28[16] > 0x10u )
            {
              v53 = 257;
              v33 = sub_15FDBD0(v30, (__int64)v28, (__int64)v42, (__int64)v52, 0);
              v29 = (_QWORD *)v33;
              if ( v55 )
              {
                v43 = v56;
                sub_157E9D0(v55 + 40, v33);
                v34 = *v43;
                v35 = v29[3] & 7LL;
                v29[4] = v43;
                v34 &= 0xFFFFFFFFFFFFFFF8LL;
                v29[3] = v34 | v35;
                *(_QWORD *)(v34 + 8) = v29 + 3;
                *v43 = *v43 & 7 | (unsigned __int64)(v29 + 3);
              }
              sub_164B780((__int64)v29, (__int64 *)&v50);
              if ( v54 )
              {
                v44 = v54;
                sub_1623A60((__int64)&v44, (__int64)v54, 2);
                v36 = v29[6];
                v37 = (__int64)(v29 + 6);
                if ( v36 )
                {
                  sub_161E7C0((__int64)(v29 + 6), v36);
                  v37 = (__int64)(v29 + 6);
                }
                v38 = v44;
                v29[6] = v44;
                if ( v38 )
                  sub_1623210((__int64)&v44, v38, v37);
              }
            }
            else
            {
              v29 = (_QWORD *)sub_15A46C0(v30, (__int64 ***)v28, v42, 0);
            }
          }
          *v31 = v29;
          ++v25;
        }
        while ( v25 != v41 );
      }
      sub_1A41120(a1, a2, &v62, a3, a4, a5, a6, v20, v21, a9, a10);
      if ( v62 != v64 )
        _libc_free((unsigned __int64)v62);
      if ( v66 != &v67 )
        _libc_free((unsigned __int64)v66);
      result = 1;
      if ( v54 )
      {
        sub_161E7C0((__int64)&v54, (__int64)v54);
        return 1;
      }
    }
  }
  return result;
}
