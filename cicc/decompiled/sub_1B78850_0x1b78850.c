// Function: sub_1B78850
// Address: 0x1b78850
//
void __fastcall sub_1B78850(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  _BYTE *v12; // rdi
  __int64 v13; // r12
  _BYTE *v14; // rbx
  __int64 v15; // rdx
  unsigned int v16; // esi
  __int64 v17; // r12
  double v18; // xmm4_8
  double v19; // xmm5_8
  _BYTE *v20; // [rsp+8h] [rbp-D8h]
  __int64 v21; // [rsp+10h] [rbp-D0h] BYREF
  char v22; // [rsp+18h] [rbp-C8h]
  _BYTE *v23; // [rsp+20h] [rbp-C0h] BYREF
  __int64 v24; // [rsp+28h] [rbp-B8h]
  _BYTE v25[176]; // [rsp+30h] [rbp-B0h] BYREF

  v23 = v25;
  v24 = 0x800000000LL;
  sub_1626D60(a2, (__int64)&v23);
  sub_161FB70(a2);
  v12 = v23;
  v13 = 16LL * (unsigned int)v24;
  v20 = &v23[v13];
  if ( &v23[v13] != v23 )
  {
    v14 = v23;
    do
    {
      v17 = *((_QWORD *)v14 + 1);
      sub_1B76840((__int64)&v21, a1, v17, *(double *)a3.m128_u64, a4, a5);
      if ( v22 )
        v15 = v21;
      else
        v15 = sub_1B785E0(a1, v17, a3, a4, a5, a6, v18, v19, a9, a10);
      v16 = *(_DWORD *)v14;
      v14 += 16;
      sub_16267C0(a2, v16, v15);
    }
    while ( v20 != v14 );
    v12 = v23;
  }
  if ( v12 != v25 )
    _libc_free((unsigned __int64)v12);
}
