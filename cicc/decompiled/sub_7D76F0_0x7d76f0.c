// Function: sub_7D76F0
// Address: 0x7d76f0
//
void __fastcall sub_7D76F0(const __m128i *a1)
{
  _QWORD *v2; // r14
  __int64 v3; // r15
  __int64 v4; // r13
  _QWORD *v5; // r12
  char v6; // al
  _BYTE *v7; // rax
  void *v8; // r13
  __m128i *v9; // r12
  _QWORD *v10; // rax
  __int64 v11; // rdi
  _QWORD *v12; // rdi
  _BYTE *v13; // rax
  __int64 i; // [rsp-40h] [rbp-40h]

  if ( !a1[4].m128i_i8[8] && (*(_BYTE *)(a1[5].m128i_i64[0] + 173) & 4) != 0 )
  {
    v2 = sub_72BA30(unk_4F06A60);
    v3 = *(_QWORD *)(a1[5].m128i_i64[0] + 120);
    for ( i = a1[5].m128i_i64[0]; *(_BYTE *)(v3 + 140) == 12; v3 = *(_QWORD *)(v3 + 160) )
      ;
    v4 = 1;
    v5 = 0;
    *(_QWORD *)(i + 248) = sub_7E7CA0(v2);
    do
    {
      if ( (*(_BYTE *)(v3 + 169) & 2) != 0 )
      {
        v10 = sub_72D900(v3);
        v11 = v10[6];
        if ( v11 )
          v12 = sub_73E830(v11);
        else
          v12 = (_QWORD *)sub_7E8090(v10[2], 1);
        v13 = sub_73E130(v12, (__int64)v2);
        if ( v5 )
        {
          v5[2] = v13;
          v5 = sub_73DBF0(0x29u, (__int64)v2, (__int64)v5);
        }
        else
        {
          v5 = v13;
        }
      }
      else
      {
        v4 *= *(_QWORD *)(v3 + 176);
      }
      do
      {
        v3 = *(_QWORD *)(v3 + 160);
        v6 = *(_BYTE *)(v3 + 140);
      }
      while ( v6 == 12 );
    }
    while ( v6 == 8 );
    if ( v4 != 1 )
    {
      v5[2] = sub_73A8E0(v4, unk_4F06A60);
      v5 = sub_73DBF0(0x29u, (__int64)v2, (__int64)v5);
    }
    v7 = sub_731250(*(_QWORD *)(i + 248));
    *((_QWORD *)v7 + 2) = v5;
    v8 = sub_73DBF0(0x49u, (__int64)v2, (__int64)v7);
    v9 = (__m128i *)sub_726B30(22);
    sub_732B40(a1, v9);
    v9[1].m128i_i64[0] = a1[1].m128i_i64[0];
    a1[1].m128i_i64[0] = (__int64)v9;
    a1[2].m128i_i8[8] = 0;
    a1[3].m128i_i64[0] = (__int64)v8;
    sub_7304E0((__int64)v8);
  }
}
