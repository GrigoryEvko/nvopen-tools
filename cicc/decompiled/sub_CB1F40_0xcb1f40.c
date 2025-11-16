// Function: sub_CB1F40
// Address: 0xcb1f40
//
void *__fastcall sub_CB1F40(__int64 a1, char *a2, size_t a3)
{
  int v4; // eax
  int v5; // eax
  int v7; // eax
  int i; // ebx

  if ( *(_DWORD *)(*(_QWORD *)(a1 + 32) + 4LL * *(unsigned int *)(a1 + 40) - 4) == 7 )
    sub_CB1B10(a1, ", ", 2u);
  v4 = *(_DWORD *)(a1 + 24);
  if ( v4 && v4 < *(_DWORD *)(a1 + 80) )
  {
    sub_CB1B10(a1, "\n", 1u);
    v7 = *(_DWORD *)(a1 + 88);
    if ( v7 > 0 )
    {
      for ( i = 0; i < v7; ++i )
      {
        sub_CB1B10(a1, " ", 1u);
        v7 = *(_DWORD *)(a1 + 88);
      }
    }
    *(_DWORD *)(a1 + 80) = v7;
    sub_CB1B10(a1, "  ", 2u);
  }
  v5 = sub_C2FE50(a2, a3, 0);
  sub_CB1CC0(a1, a2, a3, v5);
  return sub_CB1B10(a1, ": ", 2u);
}
