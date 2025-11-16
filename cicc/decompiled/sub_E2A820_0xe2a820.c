// Function: sub_E2A820
// Address: 0xe2a820
//
void __fastcall sub_E2A820(__int64 *a1, char a2, char a3, char a4)
{
  unsigned __int64 v6; // r12
  unsigned __int64 v7; // rsi
  unsigned __int64 v8; // rax
  void *v9; // rdi
  unsigned __int64 v10; // rsi
  unsigned __int64 v11; // rax
  __int64 v12; // rax

  v6 = a1[1];
  if ( (a2 & 1) != 0 )
    a3 = sub_E2A620(a1, 1, a3);
  if ( (a2 & 2) != 0 )
    a3 = sub_E2A620(a1, 2, a3);
  if ( (a2 & 0x20) != 0 )
    sub_E2A620(a1, 32, a3);
  v7 = a1[1];
  if ( v6 < v7 && a4 )
  {
    v8 = a1[2];
    v9 = (void *)*a1;
    if ( v7 + 1 > v8 )
    {
      v10 = v7 + 993;
      v11 = 2 * v8;
      if ( v10 > v11 )
        a1[2] = v10;
      else
        a1[2] = v11;
      v12 = realloc(v9);
      *a1 = v12;
      v9 = (void *)v12;
      if ( !v12 )
        abort();
      v7 = a1[1];
    }
    *((_BYTE *)v9 + v7) = 32;
    ++a1[1];
  }
}
