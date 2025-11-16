// Function: sub_2C535E0
// Address: 0x2c535e0
//
void __fastcall sub_2C535E0(__int64 a1, unsigned __int8 *a2, __int64 a3)
{
  __int64 v4; // r12
  __int64 i; // rbx

  v4 = a1 + 200;
  sub_BD84D0((__int64)a2, a3);
  if ( *(_BYTE *)a3 > 0x1Cu )
  {
    sub_BD6B90((unsigned __int8 *)a3, a2);
    for ( i = *(_QWORD *)(a3 + 16); i; i = *(_QWORD *)(i + 8) )
      sub_F15FC0(v4, *(_QWORD *)(i + 24));
    if ( *(_BYTE *)a3 > 0x1Cu )
      sub_F15FC0(v4, a3);
  }
  if ( *a2 > 0x1Cu )
    sub_F15FC0(v4, (__int64)a2);
}
