// Function: sub_259AB90
// Address: 0x259ab90
//
__int64 __fastcall sub_259AB90(__int64 a1, __int64 a2, __m128i *a3, int a4, _BYTE *a5, char a6, __int64 *a7)
{
  unsigned int v10; // r14d
  __int64 *v11; // rbx
  unsigned int v12; // r15d
  __int64 *v13; // rdi
  char v14; // cl
  unsigned __int64 v15; // rdi
  __int64 *v16; // rax
  __int64 v18; // rax
  __int64 *v21; // [rsp+28h] [rbp-68h]
  int v22; // [rsp+3Ch] [rbp-54h] BYREF
  __int64 *v23; // [rsp+40h] [rbp-50h] BYREF
  __int64 v24; // [rsp+48h] [rbp-48h]
  _BYTE v25[64]; // [rsp+50h] [rbp-40h] BYREF

  *a5 = 0;
  if ( (unsigned __int8)(*(_BYTE *)sub_250D070(a3) - 12) <= 1u
    || *(_BYTE *)sub_250D070(a3) == 13
    || (LODWORD(v23) = 76, v10 = sub_2516400(a1, a3, (__int64)&v23, 1, a6, 76), (_BYTE)v10) )
  {
LABEL_15:
    v10 = 1;
    *a5 = 1;
    return v10;
  }
  LODWORD(v23) = 19;
  if ( (unsigned __int8)sub_2516400(a1, a3, (__int64)&v23, 1, 0, 0) )
  {
    v23 = (__int64 *)v25;
    v24 = 0x200000000LL;
    v22 = 92;
    sub_2515D00(a1, a3, &v22, 1, (__int64)&v23, 0);
    v21 = &v23[(unsigned int)v24];
    if ( v23 == v21 )
    {
      v12 = 255;
    }
    else
    {
      v11 = v23;
      v12 = 255;
      do
      {
        v13 = v11++;
        v12 &= sub_A71E40(v13);
      }
      while ( v21 != v11 );
      v21 = v23;
    }
    v14 = v12 & 2 | (v12 >> 6) & 2 | ((v12 >> 4) | (v12 >> 2)) & 2;
    if ( v21 != (__int64 *)v25 )
    {
      _libc_free((unsigned __int64)v21);
      v14 = v12 & 2 | (v12 >> 6) & 2 | ((v12 >> 4) | (v12 >> 2)) & 2;
    }
    if ( !v14 )
    {
      v15 = a3->m128i_i64[0] & 0xFFFFFFFFFFFFFFFCLL;
      if ( (a3->m128i_i64[0] & 3) == 3 )
        v15 = *(_QWORD *)(v15 + 24);
      v16 = (__int64 *)sub_BD5C60(v15);
      v23 = (__int64 *)sub_A778C0(v16, 76, 0);
      sub_2516380(a1, a3->m128i_i64, (__int64)&v23, 1, 0);
      goto LABEL_15;
    }
  }
  if ( a2 )
  {
    v18 = sub_259A730(a1, a3->m128i_i64[0], a3->m128i_i64[1], a2, a4, 0, 1);
    if ( a7 )
      *a7 = v18;
    if ( v18 )
    {
      v10 = *(unsigned __int8 *)(v18 + 97);
      if ( (_BYTE)v10 )
        *a5 = *(_BYTE *)(v18 + 96);
    }
  }
  return v10;
}
