// Function: sub_259B8C0
// Address: 0x259b8c0
//
__int64 __fastcall sub_259B8C0(__int64 a1, __int64 a2, __m128i *a3, int a4, _BYTE *a5, char a6, __int64 *a7)
{
  unsigned int v11; // eax
  unsigned int v12; // r12d
  unsigned __int8 *v14; // rax
  __int64 v15; // rax
  _BYTE *v16; // r15
  unsigned int v17; // ebx
  unsigned __int64 v18; // r12
  _BYTE *v19; // r15
  __int64 *v20; // rdi
  __int64 *v21; // rax
  __int64 v22; // [rsp+10h] [rbp-80h]
  unsigned __int8 v23; // [rsp+1Bh] [rbp-75h]
  __int64 v27; // [rsp+38h] [rbp-58h] BYREF
  _BYTE *v28; // [rsp+40h] [rbp-50h] BYREF
  __int64 v29; // [rsp+48h] [rbp-48h]
  _BYTE v30[64]; // [rsp+50h] [rbp-40h] BYREF

  *a5 = 0;
  LODWORD(v28) = 39;
  v11 = sub_2516400(a1, a3, (__int64)&v28, 1, a6, 39);
  if ( (_BYTE)v11 )
  {
LABEL_2:
    v12 = 1;
    *a5 = 1;
    return v12;
  }
  v12 = v11;
  v14 = sub_250CBE0(a3->m128i_i64, (__int64)a3);
  v22 = (__int64)v14;
  if ( v14 && !(unsigned __int8)sub_B2D610((__int64)v14, 6) )
  {
    v28 = v30;
    v29 = 0x200000000LL;
    LODWORD(v27) = 92;
    sub_2515D00(a1, a3, (int *)&v27, 1, (__int64)&v28, a6);
    v16 = v28;
    if ( v28 != &v28[8 * (unsigned int)v29] )
    {
      v23 = v12;
      v17 = 255;
      v18 = (unsigned __int64)v28;
      v19 = &v28[8 * (unsigned int)v29];
      do
      {
        v20 = (__int64 *)v18;
        v18 += 8LL;
        v17 &= sub_A71E40(v20);
      }
      while ( v19 != (_BYTE *)v18 );
      v12 = v23;
      v16 = v28;
      if ( !(v17 & 2 | (v17 >> 6) & 2 | ((unsigned __int8)(v17 >> 2) | (unsigned __int8)(v17 >> 4)) & 2) )
      {
        v21 = (__int64 *)sub_B2BE50(v22);
        v27 = sub_A778C0(v21, 39, 0);
        sub_2516380(a1, a3->m128i_i64, (__int64)&v27, 1, 0);
        if ( v28 != v30 )
          _libc_free((unsigned __int64)v28);
        goto LABEL_2;
      }
    }
    if ( v16 != v30 )
      _libc_free((unsigned __int64)v16);
  }
  if ( a2 )
  {
    v15 = sub_252BBE0(a1, a3->m128i_i64[0], a3->m128i_i64[1], a2, a4, 0, 1);
    if ( a7 )
      *a7 = v15;
    if ( v15 )
    {
      v12 = *(unsigned __int8 *)(v15 + 97);
      if ( (_BYTE)v12 )
        *a5 = *(_BYTE *)(v15 + 96);
    }
  }
  return v12;
}
