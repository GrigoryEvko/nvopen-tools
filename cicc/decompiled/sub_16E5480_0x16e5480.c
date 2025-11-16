// Function: sub_16E5480
// Address: 0x16e5480
//
void *__fastcall sub_16E5480(__int64 a1, const char *a2, size_t a3)
{
  const char *v4; // rsi
  size_t v5; // rdx

  sub_16E4B40(a1, a2, a3);
  sub_16E4B40(a1, ":", 1u);
  v4 = " ";
  v5 = 1;
  if ( a3 <= 0xF )
  {
    v5 = 16 - a3;
    v4 = &asc_3F6A94C[a3];
  }
  return sub_16E4B40(a1, v4, v5);
}
