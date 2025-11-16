// Function: sub_1AEB420
// Address: 0x1aeb420
//
__int64 __fastcall sub_1AEB420(
        __int64 ***a1,
        __int64 *a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v12; // rbx
  _QWORD *v13; // r15
  unsigned int v14; // r12d
  __int64 ****v16; // rax
  char v17; // dl
  __int64 ****v18; // rsi
  __int64 ****v19; // rcx
  __int64 v20; // rax
  double v21; // xmm4_8
  double v22; // xmm5_8
  __int64 v23; // rdi
  __int64 v24; // [rsp+0h] [rbp-80h] BYREF
  __int64 ****v25; // [rsp+8h] [rbp-78h]
  __int64 ****v26; // [rsp+10h] [rbp-70h]
  __int64 v27; // [rsp+18h] [rbp-68h]
  int v28; // [rsp+20h] [rbp-60h]
  _BYTE v29[88]; // [rsp+28h] [rbp-58h] BYREF

  v24 = 0;
  v25 = (__int64 ****)v29;
  v26 = (__int64 ****)v29;
  v27 = 4;
  v28 = 0;
  while ( 1 )
  {
    v12 = (__int64)a1[1];
    if ( v12 )
    {
      v13 = sub_1648700((__int64)a1[1]);
      while ( 1 )
      {
        v12 = *(_QWORD *)(v12 + 8);
        if ( !v12 )
          break;
        if ( v13 != sub_1648700(v12) )
          goto LABEL_7;
      }
    }
    if ( (unsigned __int8)sub_15F3040((__int64)a1) || sub_15F3330((__int64)a1) )
    {
LABEL_7:
      v14 = 0;
      goto LABEL_8;
    }
    if ( !a1[1] )
      break;
    v16 = v25;
    if ( v26 != v25 )
      goto LABEL_14;
    v18 = &v25[HIDWORD(v27)];
    if ( v25 != v18 )
    {
      v19 = 0;
      while ( a1 != *v16 )
      {
        if ( *v16 == (__int64 ***)-2LL )
          v19 = v16;
        if ( v18 == ++v16 )
        {
          if ( !v19 )
            goto LABEL_25;
          *v19 = a1;
          --v28;
          ++v24;
          goto LABEL_15;
        }
      }
LABEL_22:
      v20 = sub_1599EF0(*a1);
      sub_164D160((__int64)a1, v20, a3, a4, a5, a6, v21, v22, a9, a10);
      v23 = (__int64)a1;
      v14 = 1;
      sub_1AEB370(v23, a2);
      goto LABEL_8;
    }
LABEL_25:
    if ( HIDWORD(v27) < (unsigned int)v27 )
    {
      ++HIDWORD(v27);
      *v18 = a1;
      ++v24;
    }
    else
    {
LABEL_14:
      sub_16CCBA0((__int64)&v24, (__int64)a1);
      if ( !v17 )
        goto LABEL_22;
    }
LABEL_15:
    a1 = (__int64 ***)sub_1648700((__int64)a1[1]);
  }
  v14 = sub_1AEB370((__int64)a1, a2);
LABEL_8:
  if ( v26 != v25 )
    _libc_free((unsigned __int64)v26);
  return v14;
}
