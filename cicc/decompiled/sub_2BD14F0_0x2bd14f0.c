// Function: sub_2BD14F0
// Address: 0x2bd14f0
//
__int64 __fastcall sub_2BD14F0(__int64 a1, __int64 a2, __m128i a3, __int64 a4, __int64 a5, char a6)
{
  unsigned int v7; // r12d
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 *v12; // r12
  __int64 v13; // r13
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  __m128i v19; // xmm0
  __m128i v20; // xmm1
  __m128i v21; // xmm2
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 *v24; // [rsp+10h] [rbp-4B0h] BYREF
  __int64 v25; // [rsp+18h] [rbp-4A8h]
  _BYTE v26[128]; // [rsp+20h] [rbp-4A0h] BYREF
  unsigned __int64 v27[2]; // [rsp+A0h] [rbp-420h] BYREF
  _BYTE v28[128]; // [rsp+B0h] [rbp-410h] BYREF
  void *v29; // [rsp+130h] [rbp-390h] BYREF
  int v30; // [rsp+138h] [rbp-388h]
  char v31; // [rsp+13Ch] [rbp-384h]
  __int64 v32; // [rsp+140h] [rbp-380h]
  __m128i v33; // [rsp+148h] [rbp-378h]
  __int64 v34; // [rsp+158h] [rbp-368h]
  __m128i v35; // [rsp+160h] [rbp-360h]
  __m128i v36; // [rsp+170h] [rbp-350h]
  _QWORD v37[2]; // [rsp+180h] [rbp-340h] BYREF
  _BYTE v38[324]; // [rsp+190h] [rbp-330h] BYREF
  int v39; // [rsp+2D4h] [rbp-1ECh]
  __int64 v40; // [rsp+2D8h] [rbp-1E8h]
  void *v41; // [rsp+2E0h] [rbp-1E0h] BYREF
  int v42; // [rsp+2E8h] [rbp-1D8h]
  char v43; // [rsp+2ECh] [rbp-1D4h]
  __int64 v44; // [rsp+2F0h] [rbp-1D0h]
  __m128i v45; // [rsp+2F8h] [rbp-1C8h] BYREF
  __int64 v46; // [rsp+308h] [rbp-1B8h]
  __m128i v47; // [rsp+310h] [rbp-1B0h] BYREF
  __m128i v48; // [rsp+320h] [rbp-1A0h] BYREF
  _BYTE v49[8]; // [rsp+330h] [rbp-190h] BYREF
  int v50; // [rsp+338h] [rbp-188h]
  char v51; // [rsp+480h] [rbp-40h]
  int v52; // [rsp+484h] [rbp-3Ch]
  __int64 v53; // [rsp+488h] [rbp-38h]

  v7 = 0;
  if ( (unsigned int)sub_2B2B880(a5, *(_QWORD *)(a2 + 8)) )
  {
    v24 = (__int64 *)v26;
    v27[0] = (unsigned __int64)v28;
    v25 = 0x1000000000LL;
    v27[1] = 0x1000000000LL;
    v7 = sub_2B3C490(a2, (__int64)&v24, (__int64)v27, a5, v9, v10);
    if ( (_BYTE)v7 )
    {
      if ( a6 && (_DWORD)v25 == 2 )
      {
        v12 = *(__int64 **)(a5 + 3352);
        v13 = *v12;
        v14 = sub_B2BE50(*v12);
        if ( sub_B6EA50(v14)
          || (v22 = sub_B2BE50(v13),
              v23 = sub_B6F970(v22),
              (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v23 + 48LL))(v23)) )
        {
          sub_B176B0((__int64)&v41, (__int64)"slp-vectorizer", (__int64)"NotPossible", 11, a2);
          sub_B18290(
            (__int64)&v41,
            "Cannot SLP vectorize list: only 2 elements of buildvalue, trying reduction first.",
            0x51u);
          v19 = _mm_loadu_si128(&v45);
          v20 = _mm_loadu_si128(&v47);
          v30 = v42;
          v21 = _mm_loadu_si128(&v48);
          v33 = v19;
          v31 = v43;
          v35 = v20;
          v32 = v44;
          v29 = &unk_49D9D40;
          v36 = v21;
          v34 = v46;
          v37[0] = v38;
          v37[1] = 0x400000000LL;
          if ( v50 )
            sub_2B44C50((__int64)v37, (__int64)v49, v15, v16, v17, v18);
          v41 = &unk_49D9D40;
          v38[320] = v51;
          v39 = v52;
          v40 = v53;
          v29 = &unk_49D9DB0;
          sub_23FD590((__int64)v49);
          sub_1049740(v12, (__int64)&v29);
          v29 = &unk_49D9D40;
          sub_23FD590((__int64)v37);
        }
        v7 = 0;
      }
      else
      {
        v7 = sub_2BCE070(a1, v24, (unsigned int)v25, a5, a6, a3);
      }
    }
    if ( (_BYTE *)v27[0] != v28 )
      _libc_free(v27[0]);
    if ( v24 != (__int64 *)v26 )
      _libc_free((unsigned __int64)v24);
  }
  return v7;
}
