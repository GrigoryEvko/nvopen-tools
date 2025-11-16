// Function: sub_7F87B0
// Address: 0x7f87b0
//
_QWORD *__fastcall sub_7F87B0(__int64 a1, __m128i *a2, _QWORD *a3, int a4, __m128i *a5)
{
  __int64 v8; // rsi
  __m128i *v9; // r15
  _QWORD *v10; // rax
  __int64 v11; // rax
  _QWORD *v12; // rax
  __int64 v13; // rcx
  __int64 v14; // r8
  _QWORD *v15; // r13
  _QWORD *v16; // r9
  __m128i *v17; // rax
  __int64 v19; // [rsp+0h] [rbp-50h]
  _QWORD *v20; // [rsp+0h] [rbp-50h]
  _QWORD *v22; // [rsp+8h] [rbp-48h]
  _DWORD v23[13]; // [rsp+1Ch] [rbp-34h] BYREF

  v8 = dword_4F06968;
  if ( dword_4F06968 && a2 )
  {
    v9 = a2;
    do
    {
      sub_7F8570(v9, v8);
      v9 = (__m128i *)v9[1].m128i_i64[0];
    }
    while ( v9 );
  }
  v10 = sub_7312D0(a1);
  *(_BYTE *)(a1 + 193) |= 0x40u;
  v10[2] = a2;
  v19 = (__int64)v10;
  sub_7EAF80(*(_QWORD *)(a1 + 152), (__m128i *)v8);
  v11 = sub_7F8700(*(_QWORD *)(a1 + 152));
  v12 = sub_73DBF0(0x69u, v11, v19);
  v15 = v12;
  v16 = v12;
  if ( a3 )
    v16 = (_QWORD *)sub_698020(a3, 73, (__int64)v12, v13, v14, (__int64)v12);
  if ( a5 )
  {
    v20 = v16;
    v17 = (__m128i *)sub_7E6A50(v16, a5->m128i_i32);
    v16 = v20;
    a5 = v17;
  }
  if ( a4 )
  {
    *((_BYTE *)v15 + 59) |= 0x40u;
  }
  else
  {
    *(_BYTE *)(a1 + 88) |= 4u;
    v22 = v16;
    sub_825720(v15);
    v16 = v22;
    if ( dword_4D04380 )
    {
      if ( v22 == v15 )
      {
        sub_76EF80(v15, a5, v23);
        v16 = v22;
        if ( v23[0] )
          return 0;
      }
    }
  }
  return v16;
}
