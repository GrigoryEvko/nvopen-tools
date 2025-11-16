// Function: sub_1494B40
// Address: 0x1494b40
//
__int64 __fastcall sub_1494B40(
        __int64 a1,
        _QWORD *a2,
        __int64 a3,
        _QWORD *a4,
        __int64 a5,
        unsigned __int8 a6,
        __m128i a7,
        __m128i a8,
        unsigned __int8 a9,
        unsigned __int8 a10)
{
  unsigned int v14; // ebx
  unsigned __int64 v15; // rdi
  unsigned __int64 v17; // [rsp+40h] [rbp-A0h] BYREF
  __int64 v18; // [rsp+48h] [rbp-98h]
  char v19; // [rsp+50h] [rbp-90h]
  _BYTE v20[8]; // [rsp+58h] [rbp-88h] BYREF
  __int64 v21; // [rsp+60h] [rbp-80h]
  unsigned __int64 v22; // [rsp+68h] [rbp-78h]
  char v23; // [rsp+A0h] [rbp-40h]

  v14 = a6;
  sub_14576D0((__int64)&v17, a3, (__int64)a4, a5, a6, a9);
  if ( !v23 )
  {
    sub_1494580((__int64)&v17, a2, a3, a4, a5, v14, a7, a8, a9, a10);
    sub_1469CF0(a3, (__int64)a4, a5, v14, a9, a10, &v17);
    *(_QWORD *)a1 = v17;
    *(_QWORD *)(a1 + 8) = v18;
    *(_BYTE *)(a1 + 16) = v19;
    sub_16CCEE0(a1 + 24, a1 + 64, 4, v20);
    v15 = v22;
    if ( v22 == v21 )
      return a1;
    goto LABEL_4;
  }
  *(_QWORD *)a1 = v17;
  *(_QWORD *)(a1 + 8) = v18;
  *(_BYTE *)(a1 + 16) = v19;
  sub_16CCCB0(a1 + 24, a1 + 64, v20);
  if ( v23 )
  {
    v15 = v22;
    if ( v22 != v21 )
LABEL_4:
      _libc_free(v15);
  }
  return a1;
}
