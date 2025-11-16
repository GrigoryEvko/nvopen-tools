// Function: sub_CB2940
// Address: 0xcb2940
//
char __fastcall sub_CB2940(__int64 a1, __int64 a2, int a3)
{
  size_t v5; // rdx
  const char *v6; // rsi

  sub_CB20A0(a1, 0);
  v5 = 2;
  v6 = "''";
  if ( *(_QWORD *)(a2 + 8) )
  {
    sub_CB1CC0(a1, *(char **)a2, *(_QWORD *)(a2 + 8), a3);
    v6 = byte_3F871B3;
    v5 = 0;
  }
  return sub_CB27D0(a1, v6, v5);
}
