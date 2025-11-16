// Function: sub_232E070
// Address: 0x232e070
//
__int64 __fastcall sub_232E070(_QWORD *a1, const void *a2, size_t a3)
{
  unsigned int v3; // r13d
  size_t v4; // rbx
  size_t v5; // r14

  v3 = 0;
  v4 = a1[1];
  if ( v4 >= a3 )
  {
    v5 = v4 - a3;
    if ( !a3 || !memcmp((const void *)(v5 + *a1), a2, a3) )
    {
      v3 = 1;
      if ( v4 > v5 )
        v4 = v5;
      a1[1] = v4;
    }
  }
  return v3;
}
