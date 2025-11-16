// Function: sub_1D3BB30
// Address: 0x1d3bb30
//
__int64 *__fastcall sub_1D3BB30(
        __int64 *a1,
        __int64 a2,
        unsigned __int64 a3,
        __int64 a4,
        __int64 a5,
        const void **a6,
        __m128i a7,
        double a8,
        __m128i a9)
{
  char v11; // di
  unsigned __int8 *v13; // rax
  __int64 v14; // r8
  const void **v15; // rax
  __int128 v16; // rax
  __int64 v18; // rdx
  __int64 v19; // rcx
  unsigned int v20; // ebx
  __int64 v21; // r8
  __int64 v22; // r9
  unsigned int v23; // eax
  unsigned int v24; // eax
  __int128 v25; // [rsp-10h] [rbp-70h]
  unsigned __int8 v26; // [rsp+Fh] [rbp-51h]
  __int64 v27; // [rsp+10h] [rbp-50h] BYREF
  const void **v28; // [rsp+18h] [rbp-48h]
  char v29[8]; // [rsp+20h] [rbp-40h] BYREF
  const void **v30; // [rsp+28h] [rbp-38h]

  v11 = a5;
  v13 = (unsigned __int8 *)(*(_QWORD *)(a2 + 40) + 16LL * (unsigned int)a3);
  v27 = a5;
  v14 = *v13;
  v15 = (const void **)*((_QWORD *)v13 + 1);
  v28 = a6;
  v29[0] = v14;
  v30 = v15;
  if ( (_BYTE)v14 == v11 )
  {
    if ( (_BYTE)v14 || v15 == a6 )
      goto LABEL_3;
LABEL_10:
    v26 = v14;
    v24 = sub_1F58D40(&v27, a2, a3, a4, v14, a6);
    v21 = v26;
    v20 = v24;
    if ( !v26 )
      goto LABEL_11;
LABEL_6:
    v23 = sub_1D13440(v21);
    goto LABEL_7;
  }
  if ( !v11 )
    goto LABEL_10;
  v20 = sub_1D13440(v11);
  if ( (_BYTE)v21 )
    goto LABEL_6;
LABEL_11:
  v23 = sub_1F58D40(v29, a2, v18, v19, v21, v22);
LABEL_7:
  if ( v23 >= v20 )
  {
LABEL_3:
    *(_QWORD *)&v16 = sub_1D38E70((__int64)a1, 0, a4, 0, a7, a8, a9);
    return sub_1D332F0(a1, 154, a4, (unsigned int)v27, v28, 0, *(double *)a7.m128i_i64, a8, a9, a2, a3, v16);
  }
  *((_QWORD *)&v25 + 1) = a3;
  *(_QWORD *)&v25 = a2;
  return (__int64 *)sub_1D309E0(
                      a1,
                      157,
                      a4,
                      (unsigned int)v27,
                      v28,
                      0,
                      *(double *)a7.m128i_i64,
                      a8,
                      *(double *)a9.m128i_i64,
                      v25);
}
