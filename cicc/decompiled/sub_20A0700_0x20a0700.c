// Function: sub_20A0700
// Address: 0x20a0700
//
__int64 __fastcall sub_20A0700(__int64 a1, __int64 a2, __m128i a3, __m128i a4, __m128i a5)
{
  __int64 v5; // r13
  __int64 v6; // r14
  __int64 v7; // rax
  unsigned int v8; // edx
  unsigned __int8 v9; // al
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // rdx
  __int64 v13; // [rsp-10h] [rbp-30h]

  v5 = *(_QWORD *)(a1 + 552);
  v6 = *(_QWORD *)(v5 + 16);
  v7 = sub_1E0A0C0(*(_QWORD *)(v5 + 32));
  v8 = 8 * sub_15A9520(v7, 0);
  if ( v8 == 32 )
  {
    v9 = 5;
  }
  else if ( v8 > 0x20 )
  {
    v9 = 6;
    if ( v8 != 64 )
    {
      v9 = 0;
      if ( v8 == 128 )
        v9 = 7;
    }
  }
  else
  {
    v9 = 3;
    if ( v8 != 8 )
      v9 = 4 * (v8 == 16);
  }
  v10 = sub_1D27640(v5, *(char **)(v6 + 77784), v9, 0);
  sub_209FCA0(a1, a2 | 4, v10, v11, 0, 1, a3, a4, a5, 1);
  return v13;
}
