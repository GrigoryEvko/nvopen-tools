// Function: sub_6FC7D0
// Address: 0x6fc7d0
//
void __fastcall sub_6FC7D0(__int64 a1, __m128i *a2, __m128i *a3)
{
  char v4; // dl
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax

  v4 = *(_BYTE *)(a1 + 140);
  if ( v4 == 12 )
  {
    v5 = a1;
    do
    {
      v5 = *(_QWORD *)(v5 + 160);
      v4 = *(_BYTE *)(v5 + 140);
    }
    while ( v4 == 12 );
  }
  if ( v4 )
  {
    if ( a2 )
    {
      v6 = a2->m128i_i64[0];
      if ( a2->m128i_i64[0] != a1 )
      {
        if ( !v6 || !dword_4F07588 || (v7 = *(_QWORD *)(v6 + 32), *(_QWORD *)(a1 + 32) != v7) || !v7 )
          sub_6FC3F0(a1, a2, 1u);
      }
    }
    v8 = a3->m128i_i64[0];
    if ( a3->m128i_i64[0] != a1 )
    {
      if ( !v8 || !dword_4F07588 || (v9 = *(_QWORD *)(v8 + 32), *(_QWORD *)(a1 + 32) != v9) || !v9 )
        sub_6FC3F0(a1, a3, 1u);
    }
  }
}
