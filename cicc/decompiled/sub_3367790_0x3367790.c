// Function: sub_3367790
// Address: 0x3367790
//
_QWORD *__fastcall sub_3367790(__int64 *a1, __int32 a2, _QWORD *a3, char a4)
{
  __int64 v5; // r14
  _QWORD *v6; // r13
  __int64 v8; // r14
  _WORD *v9; // r14
  __int64 v10; // r13
  __int64 v11; // [rsp+0h] [rbp-C0h]
  const __m128i *v12; // [rsp+8h] [rbp-B8h]
  __int64 v13; // [rsp+10h] [rbp-B0h]
  __int64 v14; // [rsp+10h] [rbp-B0h]
  unsigned __int8 *v16; // [rsp+28h] [rbp-98h] BYREF
  unsigned __int64 v17[2]; // [rsp+30h] [rbp-90h] BYREF
  _QWORD v18[2]; // [rsp+40h] [rbp-80h] BYREF
  const __m128i *v19; // [rsp+50h] [rbp-70h] BYREF
  __int64 v20; // [rsp+58h] [rbp-68h]
  unsigned __int64 v21; // [rsp+60h] [rbp-60h] BYREF
  __int32 v22; // [rsp+68h] [rbp-58h]
  __int64 v23; // [rsp+70h] [rbp-50h]
  __int64 v24; // [rsp+78h] [rbp-48h]
  __int64 v25; // [rsp+80h] [rbp-40h]

  if ( a2 < 0 && (unsigned __int8)sub_2E799E0(*a1) )
  {
    v8 = *(_QWORD *)(*(_QWORD *)a1[1] + 8LL);
    v19 = (const __m128i *)&v21;
    v20 = 0x100000001LL;
    v9 = (_WORD *)(v8 - 640);
    v22 = a2;
    v23 = 0;
    v24 = 0;
    v25 = 0;
    v21 = v21 & 0xFFFFFFF000000000LL | 0x800000000LL;
    if ( a4 )
      a3 = (_QWORD *)sub_B0DAC0(a3, 1, 0);
    v17[0] = (unsigned __int64)v18;
    v17[1] = 0x200000002LL;
    v18[0] = 4101;
    v18[1] = 0;
    v10 = sub_B0D8A0(a3, (__int64)v17, 0, 0);
    v12 = v19;
    v14 = (unsigned int)v20;
    v11 = *(_QWORD *)a1[3];
    sub_B10CB0(&v16, *(_QWORD *)a1[2]);
    v6 = sub_2E908B0((_QWORD *)*a1, &v16, v9, 0, v12, v14, v11, v10);
    if ( v16 )
      sub_B91220((__int64)&v16, (__int64)v16);
    if ( (_QWORD *)v17[0] != v18 )
      _libc_free(v17[0]);
    if ( v19 != (const __m128i *)&v21 )
      _libc_free((unsigned __int64)v19);
  }
  else
  {
    v5 = *(_QWORD *)(*(_QWORD *)a1[1] + 8LL) - 560LL;
    v13 = *(_QWORD *)a1[3];
    sub_B10CB0(&v19, *(_QWORD *)a1[2]);
    v6 = sub_2E8FEC0((_QWORD *)*a1, (unsigned __int8 **)&v19, v5, a4, a2, v13, (__int64)a3);
    if ( v19 )
      sub_B91220((__int64)&v19, (__int64)v19);
  }
  return v6;
}
