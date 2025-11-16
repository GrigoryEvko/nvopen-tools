// Function: sub_16E54F0
// Address: 0x16e54f0
//
void *__fastcall sub_16E54F0(__int64 a1, const char *a2, size_t a3)
{
  int v4; // eax
  int v6; // eax
  int i; // ebx

  if ( *(_DWORD *)(*(_QWORD *)(a1 + 32) + 4LL * *(unsigned int *)(a1 + 40) - 4) == 5 )
    sub_16E4B40(a1, ", ", 2u);
  v4 = *(_DWORD *)(a1 + 24);
  if ( v4 && v4 < *(_DWORD *)(a1 + 80) )
  {
    sub_16E4B40(a1, "\n", 1u);
    v6 = *(_DWORD *)(a1 + 88);
    if ( v6 > 0 )
    {
      for ( i = 0; i < v6; ++i )
      {
        sub_16E4B40(a1, " ", 1u);
        v6 = *(_DWORD *)(a1 + 88);
      }
    }
    *(_DWORD *)(a1 + 80) = v6;
    sub_16E4B40(a1, "  ", 2u);
  }
  sub_16E4B40(a1, a2, a3);
  return sub_16E4B40(a1, ": ", 2u);
}
