// Function: sub_2857DF0
// Address: 0x2857df0
//
char __fastcall sub_2857DF0(
        __int64 *a1,
        __int64 *a2,
        __int64 a3,
        char a4,
        __int64 a5,
        char a6,
        unsigned int a7,
        __int64 a8,
        unsigned int a9,
        __int64 a10,
        unsigned __int8 a11)
{
  char result; // al
  char v16; // dl
  unsigned __int64 v17; // r12
  char v18; // [rsp+13h] [rbp-4Dh]
  unsigned __int8 v19; // [rsp+14h] [rbp-4Ch]
  __int64 v20; // [rsp+20h] [rbp-40h]

  v19 = a11;
  if ( sub_D968A0(a10) )
    return 1;
  v20 = sub_28579B0((__int64)&a10, a2);
  v18 = v16;
  v17 = sub_28569D0(&a10, a2);
  result = sub_D968A0(a10);
  if ( result )
  {
    if ( !(v20 | v17) )
      return 1;
    result = 0;
    if ( !v18 )
      return sub_2850670(a1, a3, a4, a5, a6, a7, a8, a9, v17, v20, 0, v19, 2LL * (a7 != 3) - 1);
  }
  return result;
}
