// Function: sub_7CE770
// Address: 0x7ce770
//
_BYTE **__fastcall sub_7CE770(__int64 a1, unsigned __int8 a2, __int64 *a3)
{
  int v3; // r12d
  unsigned __int64 v4; // r14
  int v5; // esi
  __int64 *v6; // r13
  __int64 v7; // rdi
  __int64 v8; // rax
  __int64 v9; // rcx
  char v10; // cl
  unsigned __int64 v11; // rbx
  __m128i *v12; // rax
  size_t v13; // r15
  char *v14; // rdi
  __int64 v16; // [rsp+8h] [rbp-68h]
  _BYTE *v17; // [rsp+10h] [rbp-60h]
  int v18; // [rsp+20h] [rbp-50h]
  _QWORD *v20; // [rsp+28h] [rbp-48h]
  int v21; // [rsp+34h] [rbp-3Ch] BYREF
  _QWORD v22[7]; // [rsp+38h] [rbp-38h] BYREF

  v17 = qword_4F06460;
  if ( !a3 )
  {
    a3 = *(__int64 **)(a1 + 8);
    v3 = a2;
    v4 = qword_4F06B40[a2];
    if ( a2 <= 4u )
    {
      if ( !a3 )
        BUG();
      goto LABEL_3;
    }
LABEL_32:
    sub_721090();
  }
  v3 = a2;
  v4 = qword_4F06B40[a2];
  if ( a2 > 4u )
    goto LABEL_32;
LABEL_3:
  v5 = 0;
  v6 = 0;
  v7 = 0;
  do
  {
    while ( *((_BYTE *)a3 + 26) == 3 )
    {
      a3 = (__int64 *)*a3;
      if ( !a3 )
        goto LABEL_16;
    }
    v9 = a3[6];
    if ( !v6 )
      v6 = a3;
    if ( !*(_BYTE *)(v9 + 173) )
    {
      v16 = v6[6];
      goto LABEL_28;
    }
    v8 = *(_QWORD *)(v9 + 176);
    v10 = *(_BYTE *)(v9 + 168);
    if ( (v10 & 7) != v3 )
    {
      if ( (v10 & 7) != 0 )
        v5 = 1;
      else
        v8 *= v4;
    }
    a3 = (__int64 *)*a3;
    if ( a3 )
      v8 -= v4;
    v7 += v8;
  }
  while ( a3 );
LABEL_16:
  v16 = v6[6];
  if ( v5 )
  {
LABEL_28:
    sub_72C970(v16);
    goto LABEL_26;
  }
  v11 = 0;
  v18 = dword_3C19980[a2] | 0x10;
  v20 = sub_724830(v7);
  do
  {
    if ( *((_BYTE *)v6 + 26) != 3 )
    {
      v12 = (__m128i *)v6[6];
      if ( (v12[10].m128i_i8[8] & 7) != v3 )
      {
        sub_7CE2C0(
          v12[11].m128i_u64[1],
          (_BYTE *)(v12[11].m128i_i64[1] + v12[11].m128i_i64[0] - 1),
          v18,
          v12[11].m128i_i64[0] - 1,
          &v21,
          v22,
          1);
        v12 = xmmword_4F06300;
      }
      v13 = v12[11].m128i_u64[0];
      v14 = (char *)v20 + v11;
      if ( *v6 )
        v13 = v12[11].m128i_i64[0] - v4;
      v11 += v13;
      memcpy(v14, (const void *)v12[11].m128i_i64[1], v13);
    }
    v6 = (__int64 *)*v6;
  }
  while ( v6 );
  sub_724C70(v16, 2);
  *(_QWORD *)(v16 + 176) = v11;
  *(_QWORD *)(v16 + 184) = v20;
  *(_QWORD *)(v16 + 128) = sub_73C8D0(v3, v11 / v4);
  *(_BYTE *)(v16 + 168) = *(_BYTE *)(v16 + 168) & 0xF8 | a2 & 7;
LABEL_26:
  qword_4F06460 = v17;
  return &qword_4F06460;
}
