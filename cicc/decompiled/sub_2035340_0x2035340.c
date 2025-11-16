// Function: sub_2035340
// Address: 0x2035340
//
__int64 *__fastcall sub_2035340(
        __int64 a1,
        __int64 a2,
        double a3,
        double a4,
        __m128i a5,
        __int64 a6,
        __int64 a7,
        int a8,
        int a9)
{
  _BYTE *v9; // rax
  unsigned __int64 v12; // rbx
  int v13; // r12d
  __int64 v14; // rbx
  _BYTE *i; // rdx
  __int64 v16; // rax
  __int64 v17; // r12
  __int64 v18; // rbx
  __int64 v19; // r15
  __int64 v20; // rax
  unsigned __int64 v21; // rsi
  __int64 v22; // rdx
  __int64 v23; // rax
  int v24; // edx
  int v25; // edi
  __int64 v26; // rdx
  unsigned __int64 v27; // rax
  __int64 v28; // rsi
  __int64 *v29; // r12
  unsigned __int64 v30; // r13
  __int64 v31; // rbx
  __int64 *v32; // r12
  __int128 v34; // [rsp-10h] [rbp-100h]
  __int64 v35; // [rsp+20h] [rbp-D0h] BYREF
  int v36; // [rsp+28h] [rbp-C8h]
  _BYTE *v37; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v38; // [rsp+38h] [rbp-B8h]
  _BYTE v39[176]; // [rsp+40h] [rbp-B0h] BYREF

  v9 = v39;
  v12 = *(unsigned int *)(a2 + 56);
  v37 = v39;
  v38 = 0x800000000LL;
  v13 = v12;
  if ( (unsigned int)v12 > 8 )
  {
    sub_16CD150((__int64)&v37, v39, v12, 16, a8, a9);
    v9 = v37;
  }
  v14 = 16 * v12;
  LODWORD(v38) = v13;
  for ( i = &v9[v14]; i != v9; v9 += 16 )
  {
    if ( v9 )
    {
      *(_QWORD *)v9 = 0;
      *((_DWORD *)v9 + 2) = 0;
    }
  }
  v16 = *(unsigned int *)(a2 + 56);
  if ( (_DWORD)v16 )
  {
    v17 = 0;
    v18 = 0;
    v19 = 40 * v16;
    do
    {
      v20 = *(_QWORD *)(a2 + 32);
      v21 = *(_QWORD *)(v20 + v18);
      v22 = *(_QWORD *)(v20 + v18 + 8);
      v18 += 40;
      v23 = sub_2032580(a1, v21, v22);
      v25 = v24;
      v26 = v23;
      v27 = (unsigned __int64)v37;
      *(_QWORD *)&v37[v17] = v26;
      *(_DWORD *)(v27 + v17 + 8) = v25;
      v17 += 16;
    }
    while ( v19 != v18 );
  }
  v28 = *(_QWORD *)(a2 + 72);
  v29 = *(__int64 **)(a1 + 8);
  v30 = (unsigned __int64)v37;
  v31 = (unsigned int)v38;
  v35 = v28;
  if ( v28 )
    sub_1623A60((__int64)&v35, v28, 2);
  v36 = *(_DWORD *)(a2 + 64);
  *((_QWORD *)&v34 + 1) = v31;
  *(_QWORD *)&v34 = v30;
  v32 = sub_1D359D0(
          v29,
          104,
          (__int64)&v35,
          **(unsigned __int8 **)(a2 + 40),
          *(const void ***)(*(_QWORD *)(a2 + 40) + 8LL),
          0,
          a3,
          a4,
          a5,
          v34);
  if ( v35 )
    sub_161E7C0((__int64)&v35, v35);
  if ( v37 != v39 )
    _libc_free((unsigned __int64)v37);
  return v32;
}
