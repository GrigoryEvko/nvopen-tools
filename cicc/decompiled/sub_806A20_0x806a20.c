// Function: sub_806A20
// Address: 0x806a20
//
void __fastcall sub_806A20(__m128i *a1, __int64 a2, _DWORD *a3)
{
  __int64 v5; // rbx
  char v6; // al
  _BYTE *v7; // r14
  __m128i v8; // xmm3
  __int64 *v9; // rbx
  __int64 v10; // rax
  __int64 v11; // rdi
  __m128i *v12; // r14
  __int64 v13; // r13
  __int64 *v14; // r12
  __m128i *v15; // rax
  __m128i v16; // [rsp+0h] [rbp-40h] BYREF
  __m128i v17[3]; // [rsp+10h] [rbp-30h] BYREF

  *a3 = 0;
  if ( a1->m128i_i32[0] )
    goto LABEL_4;
  v5 = a1->m128i_i64[1];
  v6 = *(_BYTE *)(v5 + 40);
  if ( v6 == 11 )
  {
    if ( (unsigned int)sub_7F64C0(a1->m128i_i64[1], a1->m128i_i64[1]) )
      sub_7E1720(*(_QWORD *)(v5 + 16), (__int64)a1);
    if ( a1->m128i_i32[0] )
      goto LABEL_4;
    v5 = a1->m128i_i64[1];
    if ( *(_BYTE *)(v5 + 40) != 8 )
      goto LABEL_4;
  }
  else if ( v6 != 8 )
  {
LABEL_4:
    v7 = sub_726B30(8);
    v16 = _mm_loadu_si128(a1);
    v17[0] = _mm_loadu_si128(a1 + 1);
    sub_7E6810((__int64)v7, (__int64)a1, 1);
    v8 = _mm_loadu_si128(v17);
    *a1 = _mm_loadu_si128(&v16);
    a1[1] = v8;
    sub_7E17A0((__int64)v7);
    v9 = (__int64 *)*qword_4D03F60;
    *qword_4D03F60 = 0;
    if ( !v9 )
      return;
    v15 = sub_806990((__int64)a1);
    v15[7].m128i_i8[8] &= ~1u;
    v12 = v15;
    goto LABEL_11;
  }
  v10 = *(_QWORD *)(a2 + 72);
  if ( v5 == v10 )
  {
    sub_7E1740(a2, (__int64)a1);
  }
  else
  {
    do
    {
      v11 = v10;
      v10 = *(_QWORD *)(v10 + 16);
    }
    while ( v5 != v10 );
    sub_7E1720(v11, (__int64)a1);
  }
  v9 = (__int64 *)*qword_4D03F60;
  *qword_4D03F60 = 0;
  if ( v9 )
  {
    v12 = sub_806990((__int64)a1);
LABEL_11:
    *a3 = 1;
    do
    {
      v13 = v9[1];
      v14 = v9;
      v9 = (__int64 *)*v9;
      sub_7268E0(v13, 6);
      *(_QWORD *)(v13 + 72) = v12;
      *v14 = 0;
      sub_7E17F0(v14);
    }
    while ( v9 );
  }
}
