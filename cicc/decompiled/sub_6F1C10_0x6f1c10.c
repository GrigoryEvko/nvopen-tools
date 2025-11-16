// Function: sub_6F1C10
// Address: 0x6f1c10
//
__int64 __fastcall sub_6F1C10(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __m128i *a4,
        unsigned int a5,
        __int64 *a6,
        int *a7,
        unsigned int *a8)
{
  _QWORD *v11; // rax
  unsigned int v12; // r12d
  __int64 v15; // [rsp+20h] [rbp-50h] BYREF
  __int64 v16; // [rsp+28h] [rbp-48h]
  __int64 v17; // [rsp+30h] [rbp-40h]

  v17 = 0;
  v16 = 1;
  v15 = sub_823970(24);
  v11 = (_QWORD *)v15;
  if ( v15 )
  {
    *(_BYTE *)(v15 + 16) &= 0xF0u;
    *v11 = a3;
    v11[1] = a2;
  }
  v17 = 1;
  v12 = sub_6F0CB0(a1, (unsigned __int64)&v15, a4, a5, a6, a7, a8);
  sub_823A00(v15, 24 * v16);
  return v12;
}
