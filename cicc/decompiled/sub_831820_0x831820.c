// Function: sub_831820
// Address: 0x831820
//
void __fastcall sub_831820(__m128i *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r14
  __int64 v6; // rax
  char i; // dl
  char v8; // dl
  __int64 v9; // rax
  int v10; // eax
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // rdx

  v5 = a1->m128i_i64[0];
  if ( a1->m128i_i64[0] != a2 && !(unsigned int)sub_8D97D0(a1->m128i_i64[0], a2, 0, a4, a5) && a1[1].m128i_i8[0] )
  {
    v6 = a1->m128i_i64[0];
    for ( i = *(_BYTE *)(a1->m128i_i64[0] + 140); i == 12; i = *(_BYTE *)(v6 + 140) )
      v6 = *(_QWORD *)(v6 + 160);
    if ( i )
    {
      v8 = *(_BYTE *)(a2 + 140);
      if ( v8 == 12 )
      {
        v9 = a2;
        do
        {
          v9 = *(_QWORD *)(v9 + 160);
          v8 = *(_BYTE *)(v9 + 140);
        }
        while ( v8 == 12 );
      }
      if ( v8 )
      {
        v10 = sub_8DED30(v5, a2, 3);
        v13 = 0;
        if ( !v10 )
          v13 = sub_8D5CE0(v5, a2);
        sub_831640(a1, (_DWORD *)a2, v13, v11, v12);
      }
      else
      {
        sub_6E6840((__int64)a1);
      }
    }
  }
}
