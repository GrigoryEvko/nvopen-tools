// Function: sub_1F973B0
// Address: 0x1f973b0
//
__int64 __fastcall sub_1F973B0(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        unsigned int a4,
        const void **a5,
        _BYTE *a6,
        __m128i a7,
        double a8,
        __m128i a9)
{
  __int64 v14; // rsi
  __int16 v15; // ax
  __int64 v16; // rcx
  __int64 v17; // rax
  __int64 v18; // r13
  __int64 v20; // rax
  char v21; // di
  __int64 v22; // rax
  char v23; // al
  const void **v24; // r8
  char v25; // si
  __int64 v26; // rax
  unsigned __int64 v27; // rdx
  __int64 *v28; // rax
  unsigned __int64 v29; // rdx
  unsigned __int8 v30; // si
  __int64 v31; // rdx
  __int64 v32; // rax
  _QWORD *v33; // rdi
  int v34; // esi
  __int128 v35; // [rsp-10h] [rbp-70h]
  __int128 v36; // [rsp-10h] [rbp-70h]
  _BYTE *v37; // [rsp+0h] [rbp-60h]
  const void **v38; // [rsp+8h] [rbp-58h]
  const void **v39; // [rsp+8h] [rbp-58h]
  const void **v40; // [rsp+8h] [rbp-58h]
  const void **v41; // [rsp+8h] [rbp-58h]
  __int64 v42; // [rsp+10h] [rbp-50h] BYREF
  int v43; // [rsp+18h] [rbp-48h]
  char v44[8]; // [rsp+20h] [rbp-40h] BYREF
  __int64 v45; // [rsp+28h] [rbp-38h]

  *a6 = 0;
  v14 = *(_QWORD *)(a2 + 72);
  v42 = v14;
  if ( v14 )
  {
    v37 = a6;
    v38 = a5;
    sub_1623A60((__int64)&v42, v14, 2);
    a6 = v37;
    a5 = v38;
  }
  v43 = *(_DWORD *)(a2 + 64);
  v15 = *(_WORD *)(a2 + 24);
  if ( v15 != 185 )
  {
    switch ( v15 )
    {
      case 4:
        v41 = a5;
        v28 = sub_1F972B0(
                a1,
                **(_QWORD **)(a2 + 32),
                *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL),
                a4,
                (__int64)a5,
                a7,
                a8,
                a9);
        a5 = v41;
        if ( v28 )
        {
          v18 = (__int64)sub_1D332F0(
                           (__int64 *)*a1,
                           4,
                           (__int64)&v42,
                           a4,
                           v41,
                           0,
                           *(double *)a7.m128i_i64,
                           a8,
                           a9,
                           (__int64)v28,
                           v29,
                           *(_OWORD *)(*(_QWORD *)(a2 + 32) + 40LL));
          goto LABEL_10;
        }
        break;
      case 10:
        v20 = *(_QWORD *)(a2 + 40) + 16LL * (unsigned int)a3;
        v21 = *(_BYTE *)v20;
        v22 = *(_QWORD *)(v20 + 8);
        v44[0] = v21;
        v45 = v22;
        if ( v21 )
        {
          v25 = sub_1F6C8D0(v21);
        }
        else
        {
          v39 = a5;
          v23 = sub_1F58D40((__int64)v44);
          v24 = v39;
          v25 = v23;
        }
        *((_QWORD *)&v35 + 1) = a3;
        *(_QWORD *)&v35 = a2;
        v18 = sub_1D309E0(
                (__int64 *)*a1,
                142 - ((unsigned int)((v25 & 7) == 0) - 1),
                (__int64)&v42,
                a4,
                v24,
                0,
                *(double *)a7.m128i_i64,
                a8,
                *(double *)a9.m128i_i64,
                v35);
        goto LABEL_10;
      case 3:
        v40 = a5;
        v26 = sub_1F97690(a1, **(_QWORD **)(a2 + 32), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL), a4, a5, a6);
        a5 = v40;
        if ( v26 )
        {
          v18 = (__int64)sub_1D332F0(
                           (__int64 *)*a1,
                           3,
                           (__int64)&v42,
                           a4,
                           v40,
                           0,
                           *(double *)a7.m128i_i64,
                           a8,
                           a9,
                           v26,
                           v27,
                           *(_OWORD *)(*(_QWORD *)(a2 + 32) + 40LL));
          goto LABEL_10;
        }
        break;
    }
    goto LABEL_7;
  }
  if ( (*(_WORD *)(a2 + 26) & 0x380) != 0 )
  {
LABEL_7:
    v16 = a1[1];
    v17 = 1;
    if ( ((_BYTE)a4 == 1 || (_BYTE)a4 && (v17 = (unsigned __int8)a4, *(_QWORD *)(v16 + 8LL * (unsigned __int8)a4 + 120)))
      && !*(_BYTE *)(v16 + 259 * v17 + 2566) )
    {
      *((_QWORD *)&v36 + 1) = a3;
      *(_QWORD *)&v36 = a2;
      v18 = sub_1D309E0(
              (__int64 *)*a1,
              144,
              (__int64)&v42,
              a4,
              a5,
              0,
              *(double *)a7.m128i_i64,
              a8,
              *(double *)a9.m128i_i64,
              v36);
    }
    else
    {
      v18 = 0;
    }
    goto LABEL_10;
  }
  v30 = *(_BYTE *)(a2 + 27);
  v31 = *(_QWORD *)(a2 + 96);
  v32 = *(unsigned __int8 *)(a2 + 88);
  *a6 = 1;
  v33 = (_QWORD *)*a1;
  v34 = (v30 >> 2) & 3;
  if ( !v34 )
    LOBYTE(v34) = 1;
  v18 = sub_1D2B590(
          v33,
          v34,
          (__int64)&v42,
          a4,
          (__int64)a5,
          *(_QWORD *)(a2 + 104),
          *(_OWORD *)*(_QWORD *)(a2 + 32),
          *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL),
          *(_QWORD *)(*(_QWORD *)(a2 + 32) + 48LL),
          v32,
          v31);
LABEL_10:
  if ( v42 )
    sub_161E7C0((__int64)&v42, v42);
  return v18;
}
