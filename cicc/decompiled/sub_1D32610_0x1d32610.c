// Function: sub_1D32610
// Address: 0x1d32610
//
__int64 __fastcall sub_1D32610(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        const void **a6,
        double a7,
        double a8,
        double a9,
        __int128 a10)
{
  char v11; // di
  unsigned __int8 *v14; // rax
  __int64 v15; // r8
  const void **v16; // rax
  __int64 v18; // rdx
  __int64 v19; // rcx
  unsigned int v20; // ebx
  __int64 v21; // r8
  __int64 v22; // r9
  unsigned int v23; // eax
  __m128i v24; // xmm0
  _DWORD *v25; // rdx
  char v26; // bl
  int v27; // eax
  unsigned int v28; // eax
  char v29; // al
  __int128 v30; // [rsp-10h] [rbp-70h]
  __int128 v31; // [rsp-10h] [rbp-70h]
  _DWORD *v32; // [rsp+0h] [rbp-60h]
  unsigned __int8 v33; // [rsp+8h] [rbp-58h]
  __int64 v34; // [rsp+10h] [rbp-50h] BYREF
  const void **v35; // [rsp+18h] [rbp-48h]
  __m128i v36; // [rsp+20h] [rbp-40h] BYREF

  v11 = a5;
  v14 = (unsigned __int8 *)(*(_QWORD *)(a2 + 40) + 16LL * (unsigned int)a3);
  v34 = a5;
  v15 = *v14;
  v16 = (const void **)*((_QWORD *)v14 + 1);
  v35 = a6;
  v36.m128i_i8[0] = v15;
  v36.m128i_i64[1] = (__int64)v16;
  if ( (_BYTE)v15 == v11 )
  {
    if ( (_BYTE)v15 || v16 == a6 )
      goto LABEL_3;
LABEL_15:
    v33 = v15;
    v28 = sub_1F58D40(&v34, a2, a3, a4, v15, a6);
    v21 = v33;
    v20 = v28;
    if ( !v33 )
      goto LABEL_16;
LABEL_6:
    v23 = sub_1D13440(v21);
    goto LABEL_7;
  }
  if ( !v11 )
    goto LABEL_15;
  v20 = sub_1D13440(v11);
  if ( (_BYTE)v21 )
    goto LABEL_6;
LABEL_16:
  v23 = sub_1F58D40(&v36, a2, v18, v19, v21, v22);
LABEL_7:
  if ( v23 >= v20 )
  {
LABEL_3:
    *((_QWORD *)&v30 + 1) = a3;
    *(_QWORD *)&v30 = a2;
    return sub_1D309E0(a1, 145, a4, (unsigned int)v34, v35, 0, a7, a8, a9, v30);
  }
  v24 = _mm_loadu_si128((const __m128i *)&a10);
  v25 = (_DWORD *)a1[2];
  v36 = v24;
  if ( (_BYTE)a10 )
  {
    if ( (unsigned __int8)(a10 - 14) > 0x5Fu )
    {
      v26 = (unsigned __int8)(a10 - 86) <= 0x17u || (unsigned __int8)(a10 - 8) <= 5u;
      goto LABEL_11;
    }
  }
  else
  {
    v32 = v25;
    v26 = sub_1F58CD0(&v36);
    v29 = sub_1F58D20(&v36);
    v25 = v32;
    if ( !v29 )
    {
LABEL_11:
      if ( v26 )
        v27 = v25[16];
      else
        v27 = v25[15];
      goto LABEL_13;
    }
  }
  v27 = v25[17];
LABEL_13:
  *((_QWORD *)&v31 + 1) = a3;
  *(_QWORD *)&v31 = a2;
  return sub_1D309E0(
           a1,
           (unsigned int)(144 - v27),
           a4,
           (unsigned int)v34,
           v35,
           0,
           *(double *)v24.m128i_i64,
           a8,
           a9,
           v31);
}
