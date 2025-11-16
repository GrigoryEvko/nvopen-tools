// Function: sub_CB1E90
// Address: 0xcb1e90
//
char *__fastcall sub_CB1E90(__int64 a1, char *a2, size_t a3)
{
  int v4; // eax

  v4 = sub_C2FE50(a2, a3, 0);
  sub_CB1CC0(a1, a2, a3, v4);
  sub_CB1B10(a1, ":", 1u);
  if ( a3 > 0xF )
  {
    *(_QWORD *)(a1 + 104) = 1;
    *(_QWORD *)(a1 + 96) = " ";
    return " ";
  }
  else
  {
    *(_QWORD *)(a1 + 96) = &asc_3F6A94C[a3];
    *(_QWORD *)(a1 + 104) = 16 - a3;
    return (char *)(16 - a3);
  }
}
