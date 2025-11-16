// Function: sub_831640
// Address: 0x831640
//
void __fastcall sub_831640(__m128i *a1, _DWORD *a2, __int64 a3, __int64 a4, __int64 a5)
{
  _DWORD *v6; // rdi
  __int8 v7; // al

  if ( a3 )
  {
    sub_6F7270(a1, a3, a2, 1, 0, 1, 0, 1);
  }
  else
  {
    v6 = (_DWORD *)a1->m128i_i64[0];
    if ( a2 != v6 && !(unsigned int)sub_8D97D0(v6, a2, 0, a4, a5) )
    {
      v7 = a1[1].m128i_i8[1];
      if ( v7 == 1 )
      {
        sub_6F7690(a1, (__int64)a2);
      }
      else if ( v7 == 2 )
      {
        sub_6F7980(a1, (__int64)a2);
      }
    }
  }
}
