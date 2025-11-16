// Function: sub_1C95350
// Address: 0x1c95350
//
__int64 __fastcall sub_1C95350(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        _QWORD *a4,
        __int64 a5,
        __int64 a6,
        __m128 a7,
        double a8,
        double a9,
        double a10,
        double a11,
        double a12,
        double a13,
        __m128 a14,
        __int64 a15)
{
  unsigned int v15; // r14d
  __int64 v16; // rbx
  __int64 v17; // rbx
  __int64 v18; // r12
  int v19; // r8d
  int v20; // r9d
  __int64 v21; // rax
  int v22; // eax
  void *v23; // rdi
  double v24; // xmm4_8
  double v25; // xmm5_8
  double v27; // xmm4_8
  double v28; // xmm5_8
  char v31; // [rsp+1Ch] [rbp-D4h]
  __int64 v33; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v34; // [rsp+38h] [rbp-B8h]
  __int64 v35; // [rsp+40h] [rbp-B0h]
  __int64 v36; // [rsp+48h] [rbp-A8h]
  void *src; // [rsp+50h] [rbp-A0h] BYREF
  __int64 v38; // [rsp+58h] [rbp-98h]
  _QWORD v39[2]; // [rsp+60h] [rbp-90h] BYREF
  int v40; // [rsp+70h] [rbp-80h]
  _BYTE v41[120]; // [rsp+78h] [rbp-78h] BYREF

  a1[2] = a6;
  *a1 = a2;
  v15 = (unsigned __int8)byte_4FBDBA0;
  v31 = a3;
  if ( byte_4FBDBA0 )
  {
    v16 = *(_QWORD *)(a2 + 80);
    if ( !v16 )
    {
      src = v39;
      v38 = 0x400000000LL;
      BUG();
    }
    src = v39;
    v17 = v16 + 16;
    v38 = 0x400000000LL;
    v18 = *(_QWORD *)(v17 + 8);
    if ( v18 == v17 )
      goto LABEL_15;
    do
    {
      if ( !v18 )
        BUG();
      if ( *(_BYTE *)(v18 - 8) == 53 )
      {
        v33 = 0;
        v34 = 0;
        v35 = 0;
        v36 = 0;
        if ( (unsigned __int8)sub_1B33D90(v18 - 24, (__int64)&v33, a3, (__int64)a4, a5, a6) )
        {
          v21 = (unsigned int)v38;
          if ( (unsigned int)v38 >= HIDWORD(v38) )
          {
            sub_16CD150((__int64)&src, v39, 0, 8, v19, v20);
            v21 = (unsigned int)v38;
          }
          *((_QWORD *)src + v21) = v18 - 24;
          LODWORD(v38) = v38 + 1;
        }
        j___libc_free_0(v34);
      }
      v18 = *(_QWORD *)(v18 + 8);
    }
    while ( v17 != v18 );
    if ( (_DWORD)v38 )
      sub_1B3B3D0(src, (unsigned int)v38, a4, 0, 1, a7, a8, a9, a10, a11, a12, a13, a14);
    else
LABEL_15:
      v15 = 0;
    v22 = sub_1CF9D40(a4, a5, a15);
    v23 = src;
    v15 |= v22;
    if ( !v31 )
    {
      if ( src == v39 )
        return v15;
      goto LABEL_23;
    }
    if ( src != v39 )
      _libc_free((unsigned __int64)src);
  }
  if ( (unsigned __int8)sub_1C2F070(a2) )
  {
    src = 0;
    v38 = (__int64)v41;
    v39[0] = v41;
    v39[1] = 8;
    v40 = 0;
    v33 = 0;
    v34 = 0;
    v35 = 0;
    v36 = 0;
    sub_1C8F750((__int64)a1, (__int64)&src, (__int64)&v33, a7, a8, a9, a10, v24, v25, a13, a14);
    v15 |= sub_1C94ED0(a1, (__int64)&src, (__int64)&v33, a7, a8, a9, a10, v27, v28, a13, a14);
    j___libc_free_0(v34);
    v23 = (void *)v39[0];
    if ( v39[0] != v38 )
LABEL_23:
      _libc_free((unsigned __int64)v23);
  }
  return v15;
}
