// Function: sub_839CB0
// Address: 0x839cb0
//
void __fastcall sub_839CB0(__int64 a1, __int64 a2, __int64 a3, __m128i *a4, __int64 a5)
{
  char v6; // al

  if ( a3 )
  {
    if ( !a2 )
    {
LABEL_5:
      sub_8399C0(a1, 0, (*(_BYTE *)(a1 + 18) & 2) != 0, a4, a3, a5);
      return;
    }
  }
  else
  {
    a3 = *(_QWORD *)(a2 + 152);
  }
  v6 = *(_BYTE *)(a2 + 174);
  if ( (unsigned __int8)(v6 - 1) > 1u && v6 != 7 )
    goto LABEL_5;
  sub_82D850(a5);
  *(_DWORD *)(a5 + 8) = 0;
  *(_BYTE *)(a5 + 15) = 1;
}
