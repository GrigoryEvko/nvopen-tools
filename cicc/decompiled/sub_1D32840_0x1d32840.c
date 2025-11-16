// Function: sub_1D32840
// Address: 0x1d32840
//
__int64 __fastcall sub_1D32840(
        __int64 *a1,
        unsigned int a2,
        const void **a3,
        __int64 a4,
        __int64 a5,
        double a6,
        double a7,
        double a8)
{
  __int64 *v8; // r10
  __int64 v11; // r12
  __int64 v13; // rax
  __int64 v15; // rsi
  __int128 v16; // [rsp-10h] [rbp-60h]
  __int64 v17; // [rsp+8h] [rbp-48h]
  __int64 v18; // [rsp+10h] [rbp-40h] BYREF
  int v19; // [rsp+18h] [rbp-38h]

  v8 = a1;
  v11 = a4;
  v13 = *(_QWORD *)(a4 + 40) + 16LL * (unsigned int)a5;
  if ( (_BYTE)a2 != *(_BYTE *)v13 || !(_BYTE)a2 && *(const void ***)(v13 + 8) != a3 )
  {
    v15 = *(_QWORD *)(a4 + 72);
    v18 = v15;
    if ( v15 )
    {
      v17 = a4;
      sub_1623A60((__int64)&v18, v15, 2);
      v8 = a1;
      a4 = v17;
    }
    *((_QWORD *)&v16 + 1) = a5;
    *(_QWORD *)&v16 = v11;
    v19 = *(_DWORD *)(a4 + 64);
    v11 = sub_1D309E0(v8, 158, (__int64)&v18, a2, a3, 0, a6, a7, a8, v16);
    if ( v18 )
      sub_161E7C0((__int64)&v18, v18);
  }
  return v11;
}
