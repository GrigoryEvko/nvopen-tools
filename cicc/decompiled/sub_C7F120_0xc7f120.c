// Function: sub_C7F120
// Address: 0xc7f120
//
__int64 __fastcall sub_C7F120(__int64 a1, __int64 a2, unsigned __int64 a3)
{
  unsigned __int64 v3; // r13
  unsigned __int64 v4; // r14
  __int64 v5; // rbx
  __int64 result; // rax
  unsigned __int64 v7; // r13
  __int64 v8; // r13
  __int64 v9; // rsi
  _BYTE *v10; // rax

  v3 = a3;
  v4 = (unsigned int)a3
     - ((unsigned int)((a3 - 1) / 3)
      + (((0xAAAAAAAAAAAAAAABLL * (unsigned __int128)(a3 - 1)) >> 64) & 0xFFFFFFFE));
  if ( v4 <= a3 )
    a3 = (unsigned int)a3
       - ((unsigned int)((a3 - 1) / 3)
        + (((0xAAAAAAAAAAAAAAABLL * (unsigned __int128)(a3 - 1)) >> 64) & 0xFFFFFFFE));
  v5 = v4 + a2;
  result = sub_CB6200(a1, a2, a3);
  v7 = v3 - v4;
  if ( v7 )
  {
    v8 = v5 + v7;
    do
    {
      v10 = *(_BYTE **)(a1 + 32);
      if ( (unsigned __int64)v10 < *(_QWORD *)(a1 + 24) )
      {
        *(_QWORD *)(a1 + 32) = v10 + 1;
        *v10 = 44;
      }
      else
      {
        sub_CB5D20(a1, 44);
      }
      v9 = v5;
      v5 += 3;
      result = sub_CB6200(a1, v9, 3);
    }
    while ( v5 != v8 );
  }
  return result;
}
