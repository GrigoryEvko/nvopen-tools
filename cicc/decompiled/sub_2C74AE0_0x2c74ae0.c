// Function: sub_2C74AE0
// Address: 0x2c74ae0
//
__int64 __fastcall sub_2C74AE0(__int64 a1, __int64 *a2)
{
  if ( (unsigned __int8)sub_AE4360(a1, 0) != 3 || sub_22416F0(a2, "p6", 0, 2u) != -1 )
    return 0;
  if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - a2[1]) <= 8 )
    sub_4262D8((__int64)"basic_string::append");
  sub_2241490((unsigned __int64 *)a2, "-p6:32:32", 9u);
  return 1;
}
