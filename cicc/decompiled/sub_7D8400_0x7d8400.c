// Function: sub_7D8400
// Address: 0x7d8400
//
__int64 __fastcall sub_7D8400(__int64 a1)
{
  char v2; // al
  char v3; // bl
  __int64 v4; // rdi
  char v5; // bl
  __int64 v6; // rdx
  __int64 v7; // rcx
  const __m128i *v8; // rsi
  __int64 v9; // r8
  __int64 v10; // r9
  _QWORD *v12; // rax
  const __m128i *v13; // rax

  v2 = *(_BYTE *)(a1 + 56);
  v3 = *(_BYTE *)(a1 + 25);
  v4 = *(_QWORD *)(a1 + 72);
  v5 = v3 & 1;
  if ( v2 == 33 )
  {
    v12 = (_QWORD *)sub_7D7B30(v4);
    v8 = (const __m128i *)sub_73DCD0(v12);
    if ( v5 )
      return sub_730620(a1, v8);
  }
  else
  {
    if ( v2 != 34 )
      sub_721090();
    v8 = (const __m128i *)sub_7D7C20(v4);
    if ( v5 )
      return sub_730620(a1, v8);
  }
  v13 = (const __m128i *)sub_731370((__int64)v8, (__int64)v8, v6, v7, v9, v10);
  return sub_730620(a1, v13);
}
