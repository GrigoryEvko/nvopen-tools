// Function: sub_1D24AE0
// Address: 0x1d24ae0
//
__int64 *__fastcall sub_1D24AE0(
        _QWORD *a1,
        __int64 a2,
        int a3,
        unsigned __int8 a4,
        __int64 a5,
        __int64 a6,
        __int64 *a7,
        __int64 a8,
        __int64 a9)
{
  __int64 *v11; // r15
  __int64 *v12; // rbx
  __int64 v13; // rsi
  __int64 v14; // rsi
  int v15; // r11d
  unsigned __int16 v16; // r15
  int v17; // eax
  __int64 *v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  __int64 *v23; // r12
  int v25; // r13d
  __int128 v26; // [rsp-20h] [rbp-1A0h]
  __int128 v27; // [rsp-20h] [rbp-1A0h]
  _BYTE *v28; // [rsp+10h] [rbp-170h]
  unsigned __int8 *v33; // [rsp+48h] [rbp-138h] BYREF
  __int64 *v34[3]; // [rsp+50h] [rbp-130h] BYREF
  char v35; // [rsp+6Ah] [rbp-116h]
  char v36; // [rsp+6Bh] [rbp-115h]
  __int64 v37[5]; // [rsp+98h] [rbp-E8h] BYREF
  unsigned __int64 v38[2]; // [rsp+C0h] [rbp-C0h] BYREF
  _BYTE v39[176]; // [rsp+D0h] [rbp-B0h] BYREF

  v11 = a7;
  v28 = v39;
  v38[0] = (unsigned __int64)v39;
  v12 = &a7[2 * a8];
  v38[1] = 0x2000000000LL;
  sub_16BD430((__int64)v38, 237);
  sub_16BD4C0((__int64)v38, a2);
  if ( v12 != a7 )
  {
    do
    {
      v13 = *v11;
      v11 += 2;
      sub_16BD4C0((__int64)v38, v13);
      sub_16BD430((__int64)v38, *((_DWORD *)v11 - 2));
    }
    while ( v12 != v11 );
  }
  v14 = a4;
  if ( !a4 )
    v14 = a5;
  sub_16BD4D0((__int64)v38, v14);
  *((_QWORD *)&v26 + 1) = a5;
  v15 = *(_DWORD *)(a6 + 8);
  v33 = 0;
  *(_QWORD *)&v26 = a4;
  sub_1D189E0((__int64)v34, 237, v15, &v33, a2, a3, v26, a9);
  HIBYTE(v16) = v36;
  LOBYTE(v16) = v35 & 0xFA;
  if ( v37[0] )
    sub_161E7C0((__int64)v37, v37[0]);
  if ( v33 )
    sub_161E7C0((__int64)&v33, (__int64)v33);
  sub_16BD3E0((__int64)v38, v16);
  v17 = sub_1E340A0(a9);
  sub_16BD430((__int64)v38, v17);
  v34[0] = 0;
  v18 = sub_1D17920((__int64)a1, (__int64)v38, a6, (__int64 *)v34);
  v23 = v18;
  if ( v18 )
  {
    sub_1E34340(v18[13], a9, v19, v20, v21, v22, a8, a7, v39);
  }
  else
  {
    v25 = *(_DWORD *)(a6 + 8);
    v23 = (__int64 *)a1[26];
    if ( v23 )
      a1[26] = *v23;
    else
      v23 = (__int64 *)sub_145CBF0(a1 + 27, 112, 8);
    *((_QWORD *)&v27 + 1) = a5;
    *(_QWORD *)&v27 = a4;
    sub_1D189E0((__int64)v23, 237, v25, (unsigned __int8 **)a6, a2, a3, v27, a9);
    sub_1D23B60((__int64)a1, (__int64)v23, (__int64)a7, a8);
    sub_16BDA20(a1 + 40, v23, v34[0]);
    sub_1D172A0((__int64)a1, (__int64)v23);
  }
  if ( (_BYTE *)v38[0] != v28 )
    _libc_free(v38[0]);
  return v23;
}
