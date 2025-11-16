// Function: sub_35CA580
// Address: 0x35ca580
//
void __fastcall sub_35CA580(
        __int64 **a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 *a7,
        const __m128i *a8,
        __int64 *a9,
        __int64 a10)
{
  __int64 v11; // rax
  __int64 v12; // rcx
  __int64 v13; // rdx
  __int64 v14; // rsi
  __m128i v15; // xmm0
  __int8 *v16; // rsi
  size_t v17; // rdx
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  int v22; // ebx
  unsigned __int64 *v23; // r12
  unsigned __int64 *v24; // rbx
  unsigned __int64 *v25; // r12
  unsigned __int64 v26; // rdi
  __int64 v27; // rax
  __int64 v28; // rax
  __m128i *v29; // r14
  __int64 v30; // rsi
  unsigned __int64 *v31; // r15
  unsigned __int64 *v32; // r15
  unsigned __int64 v33; // rdi
  void *v34; // [rsp+10h] [rbp-390h] BYREF
  int v35; // [rsp+18h] [rbp-388h]
  char v36; // [rsp+1Ch] [rbp-384h]
  __int64 v37; // [rsp+20h] [rbp-380h]
  __m128i v38; // [rsp+28h] [rbp-378h]
  char *v39; // [rsp+38h] [rbp-368h]
  __m128i v40; // [rsp+40h] [rbp-360h]
  __m128i v41; // [rsp+50h] [rbp-350h]
  __m128i *v42; // [rsp+60h] [rbp-340h] BYREF
  __int64 v43; // [rsp+68h] [rbp-338h]
  _BYTE v44[320]; // [rsp+70h] [rbp-330h] BYREF
  char v45; // [rsp+1B0h] [rbp-1F0h]
  int v46; // [rsp+1B4h] [rbp-1ECh]
  __int64 v47; // [rsp+1B8h] [rbp-1E8h]
  void *v48; // [rsp+1C0h] [rbp-1E0h] BYREF
  __int64 v49; // [rsp+1C8h] [rbp-1D8h]
  __int64 v50; // [rsp+1D0h] [rbp-1D0h]
  __m128i v51; // [rsp+1D8h] [rbp-1C8h] BYREF
  char *v52; // [rsp+1E8h] [rbp-1B8h]
  __m128i v53; // [rsp+1F0h] [rbp-1B0h] BYREF
  __m128i v54; // [rsp+200h] [rbp-1A0h] BYREF
  unsigned __int64 *v55; // [rsp+210h] [rbp-190h]
  __int64 v56; // [rsp+218h] [rbp-188h]
  _BYTE v57[320]; // [rsp+220h] [rbp-180h] BYREF
  char v58; // [rsp+360h] [rbp-40h]
  int v59; // [rsp+364h] [rbp-3Ch]
  __int64 v60; // [rsp+368h] [rbp-38h]

  v11 = sub_B2BE50(**a1);
  if ( sub_B6EA50(v11)
    || (v27 = sub_B2BE50(**a1),
        v28 = sub_B6F970(v27),
        (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v28 + 48LL))(v28)) )
  {
    v12 = *a7;
    v13 = a7[1];
    v14 = **(_QWORD **)(*a9 + 32);
    v60 = *a9;
    v53.m128i_i64[1] = v13;
    v50 = v14;
    v49 = 0x200000014LL;
    v15 = _mm_loadu_si128(a8);
    v53.m128i_i64[0] = v12;
    v16 = *(__int8 **)a10;
    v17 = *(_QWORD *)(a10 + 8);
    v52 = "shrink-wrap";
    v54.m128i_i8[8] = 0;
    v55 = (unsigned __int64 *)v57;
    v56 = 0x400000000LL;
    v58 = 0;
    v59 = -1;
    v48 = &unk_4A27410;
    v51 = v15;
    sub_B18290((__int64)&v48, v16, v17);
    v43 = 0x400000000LL;
    v35 = v49;
    v22 = v56;
    v38 = _mm_loadu_si128(&v51);
    v36 = BYTE4(v49);
    v40 = _mm_loadu_si128(&v53);
    v37 = v50;
    v41 = _mm_loadu_si128(&v54);
    v34 = &unk_49D9D40;
    v39 = v52;
    v42 = (__m128i *)v44;
    if ( (_DWORD)v56 )
    {
      v29 = (__m128i *)v44;
      v30 = (unsigned int)v56;
      if ( (unsigned int)v56 > 4 )
      {
        sub_11F02D0((__int64)&v42, (unsigned int)v56, v18, v19, v20, v21);
        v29 = v42;
        v30 = (unsigned int)v56;
      }
      v31 = v55;
      v23 = &v55[10 * v30];
      if ( v55 == v23 )
      {
        LODWORD(v43) = v22;
        v45 = v58;
        v46 = v59;
        v47 = v60;
        v34 = &unk_4A27410;
      }
      else
      {
        do
        {
          if ( v29 )
          {
            v29->m128i_i64[0] = (__int64)v29[1].m128i_i64;
            sub_35C9A90(v29->m128i_i64, (_BYTE *)*v31, *v31 + v31[1]);
            v29[2].m128i_i64[0] = (__int64)v29[3].m128i_i64;
            sub_35C9A90(v29[2].m128i_i64, (_BYTE *)v31[4], v31[4] + v31[5]);
            v29[4] = _mm_loadu_si128((const __m128i *)v31 + 4);
          }
          v31 += 10;
          v29 += 5;
        }
        while ( v23 != v31 );
        LODWORD(v43) = v22;
        v23 = v55;
        v45 = v58;
        v46 = v59;
        v32 = &v55[10 * (unsigned int)v56];
        v47 = v60;
        v34 = &unk_4A27410;
        v48 = &unk_49D9D40;
        if ( v32 != v55 )
        {
          do
          {
            v32 -= 10;
            v33 = v32[4];
            if ( (unsigned __int64 *)v33 != v32 + 6 )
              j_j___libc_free_0(v33);
            if ( (unsigned __int64 *)*v32 != v32 + 2 )
              j_j___libc_free_0(*v32);
          }
          while ( v23 != v32 );
          v23 = v55;
        }
      }
    }
    else
    {
      v34 = &unk_4A27410;
      v23 = v55;
      v45 = v58;
      v46 = v59;
      v47 = v60;
    }
    if ( v23 != (unsigned __int64 *)v57 )
      _libc_free((unsigned __int64)v23);
    sub_2EAFC50(a1, (__int64)&v34);
    v24 = (unsigned __int64 *)v42;
    v34 = &unk_49D9D40;
    v25 = (unsigned __int64 *)&v42[5 * (unsigned int)v43];
    if ( v42 != (__m128i *)v25 )
    {
      do
      {
        v25 -= 10;
        v26 = v25[4];
        if ( (unsigned __int64 *)v26 != v25 + 6 )
          j_j___libc_free_0(v26);
        if ( (unsigned __int64 *)*v25 != v25 + 2 )
          j_j___libc_free_0(*v25);
      }
      while ( v24 != v25 );
      v25 = (unsigned __int64 *)v42;
    }
    if ( v25 != (unsigned __int64 *)v44 )
      _libc_free((unsigned __int64)v25);
  }
}
