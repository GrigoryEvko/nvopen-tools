// Function: sub_16F4B60
// Address: 0x16f4b60
//
__int64 __fastcall sub_16F4B60(__int64 a1, char *a2, size_t a3)
{
  size_t v3; // r13
  size_t v4; // r14
  char *v5; // rbx
  __int64 result; // rax
  size_t v7; // r13
  char *v8; // r13
  char *v9; // rsi
  _BYTE *v10; // rax

  v3 = a3;
  v4 = (unsigned int)a3
     - ((unsigned int)((a3 - 1) / 3)
      + (((0xAAAAAAAAAAAAAAABLL * (unsigned __int128)(a3 - 1)) >> 64) & 0xFFFFFFFE));
  if ( v4 <= a3 )
    a3 = (unsigned int)a3
       - ((unsigned int)((a3 - 1) / 3)
        + (((0xAAAAAAAAAAAAAAABLL * (unsigned __int128)(a3 - 1)) >> 64) & 0xFFFFFFFE));
  v5 = &a2[v4];
  result = sub_16E7EE0(a1, a2, a3);
  v7 = v3 - v4;
  if ( v7 )
  {
    v8 = &v5[v7];
    do
    {
      v10 = *(_BYTE **)(a1 + 24);
      if ( (unsigned __int64)v10 < *(_QWORD *)(a1 + 16) )
      {
        *(_QWORD *)(a1 + 24) = v10 + 1;
        *v10 = 44;
      }
      else
      {
        sub_16E7DE0(a1, 44);
      }
      v9 = v5;
      v5 += 3;
      result = sub_16E7EE0(a1, v9, 3u);
    }
    while ( v5 != v8 );
  }
  return result;
}
