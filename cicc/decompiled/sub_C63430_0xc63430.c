// Function: sub_C63430
// Address: 0xc63430
//
void __fastcall sub_C63430(__int64 *a1)
{
  __int64 v1; // rax
  __int64 v2; // rbx
  __int64 v3; // r13
  char **v4; // rsi

  v1 = a1[22];
  if ( a1[23] != v1 )
    a1[23] = v1;
  v2 = a1[18];
  v3 = a1[19];
  while ( v3 != v2 )
  {
    v4 = (char **)(v2 + 8);
    v2 += 48;
    sub_C62320(a1[17], v4);
  }
}
