// Function: sub_200E3C0
// Address: 0x200e3c0
//
unsigned __int64 __fastcall sub_200E3C0(
        __int64 **a1,
        __int64 a2,
        unsigned __int64 a3,
        __int64 a4,
        const void **a5,
        __int64 a6,
        __m128i a7,
        double a8,
        __m128i a9,
        unsigned int a10,
        const void **a11,
        _QWORD *a12)
{
  __int64 v16; // rsi
  int v17; // eax
  __int64 *v18; // rdi
  int v19; // edx
  __int64 v20; // rax
  unsigned __int8 v21; // si
  __int64 v22; // rcx
  int v23; // eax
  __int64 v24; // rcx
  __int64 v25; // rdi
  unsigned int v26; // eax
  __int64 v27; // rax
  unsigned __int8 v28; // si
  unsigned int v29; // eax
  __int64 v30; // rcx
  __int64 v31; // r10
  __int128 v32; // rax
  __int64 *v33; // rax
  __int64 v34; // rcx
  const void **v35; // r8
  int v36; // edx
  __int64 v37; // rax
  _QWORD *v38; // rbx
  __int64 v39; // rsi
  int v40; // edx
  unsigned __int64 result; // rax
  unsigned __int64 v42; // r8
  unsigned __int64 v43; // r8
  unsigned int v44; // eax
  __int128 v45; // rax
  int v46; // edx
  __int64 *v47; // r15
  __int128 v48; // rax
  __int64 *v49; // rax
  _QWORD *v50; // rbx
  unsigned int v51; // edx
  __int64 v52; // rax
  __int128 v53; // [rsp-10h] [rbp-F0h]
  unsigned __int64 v54; // [rsp-10h] [rbp-F0h]
  __int64 (__fastcall *v55)(__int64 *, __int64, _QWORD, __int64); // [rsp+0h] [rbp-E0h]
  __int64 *v56; // [rsp+8h] [rbp-D8h]
  __int64 (__fastcall *v57)(__int64 *, __int64, _QWORD, __int64); // [rsp+8h] [rbp-D8h]
  __int64 v58; // [rsp+10h] [rbp-D0h]
  __int64 *v59; // [rsp+10h] [rbp-D0h]
  unsigned int v60; // [rsp+18h] [rbp-C8h]
  __int64 v61; // [rsp+18h] [rbp-C8h]
  unsigned int v62; // [rsp+18h] [rbp-C8h]
  __int64 v63; // [rsp+18h] [rbp-C8h]
  __int64 *v65; // [rsp+20h] [rbp-C0h]
  __int64 *v66; // [rsp+20h] [rbp-C0h]
  int v67; // [rsp+28h] [rbp-B8h]
  __int64 v68; // [rsp+28h] [rbp-B8h]
  __int64 *v69; // [rsp+28h] [rbp-B8h]
  __int64 v70; // [rsp+80h] [rbp-60h] BYREF
  const void **v71; // [rsp+88h] [rbp-58h]
  __int64 v72; // [rsp+90h] [rbp-50h] BYREF
  int v73; // [rsp+98h] [rbp-48h]
  unsigned __int8 v74[8]; // [rsp+A0h] [rbp-40h] BYREF
  __int64 v75; // [rsp+A8h] [rbp-38h]

  v16 = *(_QWORD *)(a2 + 72);
  v70 = a4;
  v71 = a5;
  v60 = a3;
  v72 = v16;
  if ( v16 )
    sub_1623A60((__int64)&v72, v16, 2);
  v73 = *(_DWORD *)(a2 + 64);
  if ( (_BYTE)v70 )
    v67 = sub_200D0E0(v70);
  else
    v67 = sub_1F58D40((__int64)&v70);
  if ( (_BYTE)a10 )
    v17 = sub_200D0E0(a10);
  else
    v17 = sub_1F58D40((__int64)&a10);
  v18 = a1[1];
  if ( v17 == v67 && ((v17 - 16) & 0xFFFFFFEF) == 0 )
  {
    v69 = a1[1];
    *(_QWORD *)&v45 = sub_1D38E70((__int64)v18, 0, (__int64)&v72, 0, a7, a8, a9);
    *(_QWORD *)a6 = sub_1D332F0(
                      v69,
                      49,
                      (__int64)&v72,
                      (unsigned int)v70,
                      v71,
                      0,
                      *(double *)a7.m128i_i64,
                      a8,
                      a9,
                      a2,
                      a3,
                      v45);
    *(_DWORD *)(a6 + 8) = v46;
    v47 = a1[1];
    *(_QWORD *)&v48 = sub_1D38E70((__int64)v47, 1, (__int64)&v72, 0, a7, a8, a9);
    v49 = sub_1D332F0(v47, 49, (__int64)&v72, a10, a11, 0, *(double *)a7.m128i_i64, a8, a9, a2, a3, v48);
    v50 = a12;
    v39 = v72;
    *a12 = v49;
    result = v51;
    *((_DWORD *)v50 + 2) = v51;
    if ( v39 )
      return sub_161E7C0((__int64)&v72, v39);
  }
  else
  {
    *((_QWORD *)&v53 + 1) = a3;
    *(_QWORD *)&v53 = a2;
    *(_QWORD *)a6 = sub_1D309E0(
                      v18,
                      145,
                      (__int64)&v72,
                      (unsigned int)v70,
                      v71,
                      0,
                      *(double *)a7.m128i_i64,
                      a8,
                      *(double *)a9.m128i_i64,
                      v53);
    *(_DWORD *)(a6 + 8) = v19;
    v68 = 16LL * v60;
    v20 = *(_QWORD *)(a2 + 40) + v68;
    v21 = *(_BYTE *)v20;
    v22 = *(_QWORD *)(v20 + 8);
    v74[0] = v21;
    v75 = v22;
    if ( v21 )
    {
      v23 = sub_200D0E0(v21);
    }
    else
    {
      v61 = v22;
      v23 = sub_1F58D40((__int64)v74);
      v24 = v61;
      v21 = 0;
    }
    v25 = a1[1][4];
    v26 = v23 - 1;
    if ( v26 )
    {
      _BitScanReverse(&v26, v26);
      v55 = *(__int64 (__fastcall **)(__int64 *, __int64, _QWORD, __int64))(**a1 + 40);
      v56 = *a1;
      v58 = v24;
      v62 = 32 - (v26 ^ 0x1F);
      v27 = sub_1E0A0C0(v25);
      v28 = v55(v56, v27, v21, v58);
      if ( (unsigned int)sub_200D0E0(v28) < v62 )
      {
        v42 = ((v62 | ((unsigned __int64)v62 >> 1)) >> 2) | v62 | ((unsigned __int64)v62 >> 1);
        v43 = (v42 >> 4) | v42;
        v44 = v43 + 1;
        if ( v43 == 15 )
        {
          v28 = 4;
        }
        else if ( v44 <= 0x10 )
        {
          v28 = 3 * (v44 == 8);
        }
        else
        {
          v28 = 5;
          if ( (_DWORD)v43 != 31 )
          {
            v28 = 0;
            if ( (_DWORD)v43 == 63 )
              v28 = 6;
          }
        }
      }
    }
    else
    {
      v57 = *(__int64 (__fastcall **)(__int64 *, __int64, _QWORD, __int64))(**a1 + 40);
      v59 = *a1;
      v63 = v24;
      v52 = sub_1E0A0C0(v25);
      v28 = v57(v59, v52, v21, v63);
    }
    if ( (_BYTE)v70 )
    {
      v29 = sub_200D0E0(v70);
    }
    else
    {
      v66 = a1[1];
      v29 = sub_1F58D40((__int64)&v70);
      v30 = v28;
      v31 = (__int64)v66;
    }
    v65 = (__int64 *)v31;
    *(_QWORD *)&v32 = sub_1D38BB0(v31, v29, (__int64)&v72, v30, 0, 0, a7, a8, a9, 0);
    v33 = sub_1D332F0(
            v65,
            124,
            (__int64)&v72,
            *(unsigned __int8 *)(*(_QWORD *)(a2 + 40) + v68),
            *(const void ***)(*(_QWORD *)(a2 + 40) + v68 + 8),
            0,
            *(double *)a7.m128i_i64,
            a8,
            a9,
            a2,
            a3,
            v32);
    v34 = (__int64)a12;
    v35 = a11;
    *a12 = v33;
    *(_DWORD *)(v34 + 8) = v36;
    v37 = sub_1D309E0(
            a1[1],
            145,
            (__int64)&v72,
            a10,
            v35,
            0,
            *(double *)a7.m128i_i64,
            a8,
            *(double *)a9.m128i_i64,
            *(_OWORD *)v34);
    v38 = a12;
    v39 = v72;
    *a12 = v37;
    *((_DWORD *)v38 + 2) = v40;
    result = v54;
    if ( v39 )
      return sub_161E7C0((__int64)&v72, v39);
  }
  return result;
}
