// Function: sub_267CD60
// Address: 0x267cd60
//
void __fastcall sub_267CD60(__int64 a1, __int64 a2, __int8 *a3, size_t a4)
{
  __int64 *v6; // rax
  __int64 *v7; // r14
  int v8; // eax
  __int64 v9; // r15
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // r9
  __m128i v13; // xmm0
  __m128i v14; // xmm1
  __m128i v15; // xmm2
  __int64 v16; // rcx
  __int64 v17; // r9
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r9
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  void *v26; // [rsp+10h] [rbp-540h] BYREF
  int v27; // [rsp+18h] [rbp-538h]
  char v28; // [rsp+1Ch] [rbp-534h]
  __int64 v29; // [rsp+20h] [rbp-530h]
  __m128i v30; // [rsp+28h] [rbp-528h]
  __int64 v31; // [rsp+38h] [rbp-518h]
  __m128i v32; // [rsp+40h] [rbp-510h]
  __m128i v33; // [rsp+50h] [rbp-500h]
  _QWORD v34[2]; // [rsp+60h] [rbp-4F0h] BYREF
  _BYTE v35[324]; // [rsp+70h] [rbp-4E0h] BYREF
  int v36; // [rsp+1B4h] [rbp-39Ch]
  __int64 v37; // [rsp+1B8h] [rbp-398h]
  void *v38; // [rsp+1C0h] [rbp-390h] BYREF
  int v39; // [rsp+1C8h] [rbp-388h]
  char v40; // [rsp+1CCh] [rbp-384h]
  __int64 v41; // [rsp+1D0h] [rbp-380h]
  __m128i v42; // [rsp+1D8h] [rbp-378h] BYREF
  __int64 v43; // [rsp+1E8h] [rbp-368h]
  __m128i v44; // [rsp+1F0h] [rbp-360h] BYREF
  __m128i v45; // [rsp+200h] [rbp-350h] BYREF
  char *v46; // [rsp+210h] [rbp-340h] BYREF
  __int64 v47; // [rsp+218h] [rbp-338h]
  char v48; // [rsp+220h] [rbp-330h] BYREF
  char v49; // [rsp+360h] [rbp-1F0h]
  int v50; // [rsp+364h] [rbp-1ECh]
  __int64 v51; // [rsp+368h] [rbp-1E8h]
  void *v52; // [rsp+370h] [rbp-1E0h] BYREF
  int v53; // [rsp+378h] [rbp-1D8h]
  char v54; // [rsp+37Ch] [rbp-1D4h]
  __int64 v55; // [rsp+380h] [rbp-1D0h]
  __m128i v56; // [rsp+388h] [rbp-1C8h] BYREF
  __int64 v57; // [rsp+398h] [rbp-1B8h]
  __m128i v58; // [rsp+3A0h] [rbp-1B0h] BYREF
  __m128i v59; // [rsp+3B0h] [rbp-1A0h] BYREF
  char *v60; // [rsp+3C0h] [rbp-190h] BYREF
  __int64 v61; // [rsp+3C8h] [rbp-188h]
  char v62; // [rsp+3D0h] [rbp-180h] BYREF
  char v63; // [rsp+510h] [rbp-40h]
  int v64; // [rsp+514h] [rbp-3Ch]
  __int64 v65; // [rsp+518h] [rbp-38h]

  v6 = (__int64 *)(*(__int64 (__fastcall **)(_QWORD, _QWORD))(a1 + 56))(
                    *(_QWORD *)(a1 + 64),
                    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 72LL));
  v7 = v6;
  if ( a4 <= 2 )
  {
    v9 = *v6;
  }
  else
  {
    if ( *(_WORD *)a3 != 19791 || (v8 = 0, a3[2] != 80) )
      v8 = 1;
    v9 = *v7;
    if ( !v8 )
    {
      v10 = sub_B2BE50(*v7);
      if ( sub_B6EA50(v10)
        || (v24 = sub_B2BE50(v9),
            v25 = sub_B6F970(v24),
            (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v25 + 48LL))(v25)) )
      {
        sub_B174A0((__int64)&v38, (__int64)"openmp-opt", (__int64)a3, a4, a2);
        sub_B18290((__int64)&v38, "Removing parallel region with no side-effects.", 0x2Eu);
        v13 = _mm_loadu_si128(&v42);
        v14 = _mm_loadu_si128(&v44);
        v53 = v39;
        v15 = _mm_loadu_si128(&v45);
        v56 = v13;
        v54 = v40;
        v58 = v14;
        v55 = v41;
        v52 = &unk_49D9D40;
        v59 = v15;
        v57 = v43;
        v60 = &v62;
        v61 = 0x400000000LL;
        if ( (_DWORD)v47 )
          sub_26781A0((__int64)&v60, (__int64)&v46, v11, (unsigned int)v47, (__int64)&v60, v12);
        v63 = v49;
        v64 = v50;
        v65 = v51;
        v52 = &unk_49D9D78;
        sub_B18290((__int64)&v52, " [", 2u);
        sub_B18290((__int64)&v52, a3, a4);
        sub_B18290((__int64)&v52, "]", 1u);
        v27 = v53;
        v30 = _mm_loadu_si128(&v56);
        v28 = v54;
        v32 = _mm_loadu_si128(&v58);
        v29 = v55;
        v26 = &unk_49D9D40;
        v33 = _mm_loadu_si128(&v59);
        v31 = v57;
        v34[0] = v35;
        v34[1] = 0x400000000LL;
        if ( (_DWORD)v61 )
          sub_26781A0((__int64)v34, (__int64)&v60, (unsigned int)v61, v16, (__int64)&v60, v17);
        v52 = &unk_49D9D40;
        v35[320] = v63;
        v36 = v64;
        v37 = v65;
        v26 = &unk_49D9D78;
        sub_23FD590((__int64)&v60);
        v38 = &unk_49D9D40;
        sub_23FD590((__int64)&v46);
        sub_1049740(v7, (__int64)&v26);
        v26 = &unk_49D9D40;
        sub_23FD590((__int64)v34);
      }
      return;
    }
  }
  v18 = sub_B2BE50(v9);
  if ( sub_B6EA50(v18)
    || (v22 = sub_B2BE50(v9),
        v23 = sub_B6F970(v22),
        (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v23 + 48LL))(v23)) )
  {
    sub_B174A0((__int64)&v52, (__int64)"openmp-opt", (__int64)a3, a4, a2);
    sub_B18290((__int64)&v52, "Removing parallel region with no side-effects.", 0x2Eu);
    v39 = v53;
    v42 = _mm_loadu_si128(&v56);
    v40 = v54;
    v44 = _mm_loadu_si128(&v58);
    v41 = v55;
    v38 = &unk_49D9D40;
    v45 = _mm_loadu_si128(&v59);
    v43 = v57;
    v46 = &v48;
    v47 = 0x400000000LL;
    if ( (_DWORD)v61 )
      sub_26781A0((__int64)&v46, (__int64)&v60, v19, v20, (__int64)&v60, v21);
    v52 = &unk_49D9D40;
    v49 = v63;
    v50 = v64;
    v51 = v65;
    v38 = &unk_49D9D78;
    sub_23FD590((__int64)&v60);
    sub_1049740(v7, (__int64)&v38);
    v38 = &unk_49D9D40;
    sub_23FD590((__int64)&v46);
  }
}
