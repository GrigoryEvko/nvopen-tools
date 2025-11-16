// Function: sub_1D35F20
// Address: 0x1d35f20
//
_QWORD *__fastcall sub_1D35F20(
        __int64 *a1,
        __int64 a2,
        const void **a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        double a7,
        double a8,
        __m128i a9)
{
  bool v9; // zf
  unsigned int v13; // r8d
  __int64 v14; // r9
  _BYTE *v15; // rax
  __int64 v16; // rdx
  _QWORD *v17; // r12
  __int128 v19; // [rsp-10h] [rbp-170h]
  unsigned int v20; // [rsp+4h] [rbp-15Ch]
  __int64 v21; // [rsp+8h] [rbp-158h]
  __int64 v22; // [rsp+10h] [rbp-150h] BYREF
  const void **v23; // [rsp+18h] [rbp-148h]
  _BYTE *v24; // [rsp+20h] [rbp-140h] BYREF
  __int64 v25; // [rsp+28h] [rbp-138h]
  _BYTE v26[304]; // [rsp+30h] [rbp-130h] BYREF

  v9 = *(_WORD *)(a5 + 24) == 48;
  v22 = a2;
  v23 = a3;
  if ( v9 )
  {
    v24 = 0;
    LODWORD(v25) = 0;
    v17 = sub_1D2B300(a1, 0x30u, (__int64)&v24, v22, (__int64)a3, a6);
    if ( v24 )
      sub_161E7C0((__int64)&v24, (__int64)v24);
  }
  else
  {
    if ( (_BYTE)v22 )
      v13 = word_42E7700[(unsigned __int8)(v22 - 14)];
    else
      v13 = sub_1F58D30(&v22);
    v14 = v13;
    v25 = 0x1000000000LL;
    v15 = v26;
    v24 = v26;
    if ( v13 > 0x10 )
    {
      v20 = v13;
      v21 = v13;
      sub_16CD150((__int64)&v24, v26, v13, 16, v13, v13);
      v15 = v24;
      v13 = v20;
      v14 = v21;
    }
    LODWORD(v25) = v13;
    v16 = (__int64)&v15[16 * v14];
    if ( v15 != (_BYTE *)v16 )
    {
      do
      {
        if ( v15 )
        {
          *(_QWORD *)v15 = a5;
          *((_QWORD *)v15 + 1) = a6;
        }
        v15 += 16;
      }
      while ( (_BYTE *)v16 != v15 );
      v16 = (__int64)v24;
      v14 = (unsigned int)v25;
    }
    *((_QWORD *)&v19 + 1) = v14;
    *(_QWORD *)&v19 = v16;
    v17 = sub_1D359D0(a1, 104, a4, (unsigned int)v22, v23, 0, a7, a8, a9, v19);
    if ( v24 != v26 )
      _libc_free((unsigned __int64)v24);
  }
  return v17;
}
